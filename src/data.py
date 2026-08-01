import hashlib
import os
import random
import re

from datasets import load_dataset, load_from_disk

from prompts import get_solver_prompt


GPQA_DATASET_ID = "Idavidrein/gpqa"
GPQA_SUBSET = "gpqa_diamond"


def _select_split(dataset, preferred_split: str):
    if hasattr(dataset, "keys"):
        if preferred_split in dataset:
            return dataset[preferred_split]
        for candidate in ("test", "train", "validation"):
            if candidate in dataset:
                return dataset[candidate]
    return dataset


def _load_local_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.isdir(path):
        return _select_split(load_from_disk(path), "train")
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return _select_split(load_dataset("csv", data_files={"train": path}), "train")
    raise ValueError(f"Unsupported local dataset format: {path}")


def _load_named_dataset(name_or_path: str, dataset_id: str, split: str, config: str | None = None):
    if os.path.exists(name_or_path):
        return _load_local_dataset(name_or_path)
    if config is None:
        return _select_split(load_dataset(dataset_id), split)
    return _select_split(load_dataset(dataset_id, config), split)


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _find_column(column_names: list[str], *aliases: str) -> str:
    normalized = {_normalized_name(name): name for name in column_names}
    for alias in aliases:
        if _normalized_name(alias) in normalized:
            return normalized[_normalized_name(alias)]
    raise ValueError(f"Missing required GPQA column. Tried {aliases}; found {column_names}")


def _gpqa_columns(column_names: list[str]) -> dict[str, str]:
    return {
        "question": _find_column(column_names, "Question"),
        "correct": _find_column(column_names, "Correct Answer"),
        "incorrect_1": _find_column(column_names, "Incorrect Answer 1"),
        "incorrect_2": _find_column(column_names, "Incorrect Answer 2"),
        "incorrect_3": _find_column(column_names, "Incorrect Answer 3"),
        "record_id": _find_column(column_names, "Record ID"),
        "domain": _find_column(column_names, "High-level domain", "High Level Domain"),
        "subdomain": _find_column(column_names, "Subdomain"),
    }


def _load_gpqa(name_or_path: str):
    if os.path.exists(name_or_path):
        return _load_local_dataset(name_or_path)
    try:
        return _select_split(load_dataset(GPQA_DATASET_ID, GPQA_SUBSET), "train")
    except Exception as exc:
        raise RuntimeError(
            "Unable to load GPQA Diamond. Accept the dataset terms and configure "
            "Hugging Face authentication."
        ) from exc


def _gpqa_shuffle_seed(seed: int, record_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _format_gpqa_question(question: str, choices: list[tuple[str, str]]) -> str:
    rendered = " ".join(f"({label}) {text}" for label, text in choices)
    return f"{question.strip()}\nAnswer Choices: {rendered}"


def get_dataset(
    data_name_or_path: str,
    *,
    tokenizer,
    prompt_idx: int,
    seed: int,
):
    lower_name = data_name_or_path.lower()
    if "gsm8k" in lower_name:
        dataset = _load_named_dataset(data_name_or_path, "openai/gsm8k", "test", "socratic")
        dataset_type = "gsm8k"
        question_column = "question"
        answer_column = "answer"
        gpqa_columns = None
    elif "math" in lower_name:
        dataset = _load_named_dataset(data_name_or_path, "HuggingFaceH4/MATH-500", "test")
        dataset_type = "math"
        question_column = "problem"
        answer_column = "answer"
        gpqa_columns = None
    elif "gpqa" in lower_name:
        dataset = _load_gpqa(data_name_or_path)
        dataset_type = "gpqa"
        gpqa_columns = _gpqa_columns(list(dataset.column_names))
        question_column = gpqa_columns["question"]
        answer_column = gpqa_columns["correct"]
    else:
        raise ValueError(
            "Unsupported dataset. Expected GSM8K, MATH-500, or GPQA Diamond."
        )

    def preprocess(examples, indices):
        formatted = []
        questions = []
        answers = []
        metadata = {
            "gpqa_subset": [],
            "gpqa_record_id": [],
            "gpqa_high_level_domain": [],
            "gpqa_subdomain": [],
            "gpqa_shuffle_seed": [],
            "gpqa_choice_order": [],
            "gpqa_choice_a": [],
            "gpqa_choice_b": [],
            "gpqa_choice_c": [],
            "gpqa_choice_d": [],
        }

        for index, raw_question in enumerate(examples[question_column]):
            question = str(raw_question)
            answer = examples[answer_column][index]
            if dataset_type == "gpqa":
                record_id = str(examples[gpqa_columns["record_id"]][index]).strip() or str(
                    indices[index]
                )
                shuffle_seed = _gpqa_shuffle_seed(seed, record_id)
                choices = [
                    ("correct", str(examples[gpqa_columns["correct"]][index]).strip()),
                    ("incorrect_1", str(examples[gpqa_columns["incorrect_1"]][index]).strip()),
                    ("incorrect_2", str(examples[gpqa_columns["incorrect_2"]][index]).strip()),
                    ("incorrect_3", str(examples[gpqa_columns["incorrect_3"]][index]).strip()),
                ]
                random.Random(shuffle_seed).shuffle(choices)
                labeled_choices = list(zip("ABCD", [text for _, text in choices]))
                question = _format_gpqa_question(question, labeled_choices)
                answer = next(
                    label for label, (origin, _) in zip("ABCD", choices) if origin == "correct"
                )
                metadata["gpqa_subset"].append(GPQA_SUBSET)
                metadata["gpqa_record_id"].append(record_id)
                metadata["gpqa_high_level_domain"].append(
                    str(examples[gpqa_columns["domain"]][index]).strip()
                )
                metadata["gpqa_subdomain"].append(
                    str(examples[gpqa_columns["subdomain"]][index]).strip()
                )
                metadata["gpqa_shuffle_seed"].append(str(shuffle_seed))
                metadata["gpqa_choice_order"].append([origin for origin, _ in choices])
                for label, choice_text in labeled_choices:
                    metadata[f"gpqa_choice_{label.lower()}"].append(choice_text)

            messages = get_solver_prompt(question, dataset_type, prompt_idx)
            formatted.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            questions.append(question)
            answers.append(answer)

        result = {"formatted": formatted, "question": questions, "answer": answers}
        if dataset_type == "gpqa":
            result.update(metadata)
        return result

    return dataset.map(
        preprocess,
        batched=True,
        with_indices=True,
        load_from_cache_file=False,
    )

