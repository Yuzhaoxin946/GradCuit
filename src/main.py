import argparse
import json
import logging
import os
import random
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as transformers_logging

from data import get_dataset
from evaluation import extract_answer, extract_true_answer, judge_answer
from gradcuit import GradCuit
from rewards import SingleVerifierRewardModel


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GradCuit with a single external answer verifier."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--insert_prefix_text", required=True)
    parser.add_argument("--solver_prompt_idx", required=True, type=int, choices=[0, 1])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument(
        "--optimizer",
        required=True,
        choices=["adam", "sgd", "muon"],
    )
    parser.add_argument("--max_num_steps", required=True, type=int)
    parser.add_argument("--max_new_tokens", required=True, type=int)
    parser.add_argument("--optimize_layer_idx", required=True, type=int)
    parser.add_argument(
        "--grad_clip",
        required=True,
        type=float,
        help="Gradient norm limit. Use 0 to disable clipping.",
    )
    parser.add_argument("--reward_threshold", required=True, type=float)
    parser.add_argument("--start_data_idx", required=True, type=int)
    parser.add_argument(
        "--num_data",
        required=True,
        type=int,
        help="Number of examples to process. Use -1 for all remaining examples.",
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--vllm_base_url", required=True)
    parser.add_argument("--vllm_verifier_model_name", required=True)
    parser.add_argument("--vllm_timeout", required=True, type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--show_warnings", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_num_steps <= 0:
        raise ValueError("max_num_steps must be positive.")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if args.optimize_layer_idx < 0:
        raise ValueError("optimize_layer_idx must be non-negative.")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be non-negative.")
    if args.start_data_idx < 0:
        raise ValueError("start_data_idx must be non-negative.")
    if args.num_data != -1 and args.num_data <= 0:
        raise ValueError("num_data must be -1 or a positive integer.")
    if args.vllm_timeout <= 0:
        raise ValueError("vllm_timeout must be positive.")


def configure_warnings(show_warnings: bool) -> None:
    if show_warnings:
        warnings.resetwarnings()
        transformers_logging.set_verbosity_warning()
        logging.getLogger("transformers").setLevel(logging.WARNING)
        return
    warnings.filterwarnings("ignore")
    transformers_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}")


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sanitize_tag(text: str, max_length: int = 32) -> str:
    sanitized = "".join(character if character.isalnum() else "_" for character in text)
    sanitized = "_".join(filter(None, sanitized.split("_"))) or "value"
    return sanitized[:max_length]


def sanitized_run_arguments(args: argparse.Namespace) -> dict:
    result = vars(args).copy()
    for key in (
        "dataset",
        "model_name_or_path",
        "output_dir",
        "vllm_verifier_model_name",
    ):
        value = os.path.normpath(str(result[key]))
        result[key] = os.path.basename(value)
    result["vllm_base_url"] = "<redacted>"
    return result


def save_run_arguments(output_dir: str, args: argparse.Namespace) -> None:
    path = os.path.join(output_dir, "run_args.json")
    if os.path.exists(path):
        return
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(sanitized_run_arguments(args), file, indent=2, sort_keys=True)
        file.write("\n")


def build_output_subdir(args: argparse.Namespace) -> str:
    model_name = sanitize_tag(os.path.basename(os.path.normpath(args.model_name_or_path)))
    dataset_name = sanitize_tag(os.path.basename(os.path.normpath(args.dataset)))
    prefix_name = sanitize_tag(args.insert_prefix_text)
    return (
        f"{model_name}-{dataset_name}-prefix{prefix_name}-lr{args.lr}-"
        f"{args.optimizer}-prompt{args.solver_prompt_idx}-layer{args.optimize_layer_idx}"
    )


def find_resume_directory(output_root: str, output_subdir: str) -> str:
    candidates: list[tuple[float, str]] = []
    if os.path.isdir(output_root):
        for root, directory_names, _ in os.walk(output_root):
            for directory_name in directory_names:
                if directory_name.startswith(f"{output_subdir}-"):
                    path = os.path.join(root, directory_name)
                    candidates.append((os.path.getmtime(path), path))
    if not candidates:
        raise FileNotFoundError(
            f"No prior run found for resume with prefix {output_subdir}."
        )
    return max(candidates, key=lambda item: item[0])[1]


def resolve_output_directory(args: argparse.Namespace) -> str:
    output_subdir = build_output_subdir(args)
    if args.resume:
        return find_resume_directory(args.output_dir, output_subdir)
    now = datetime.now(timezone.utc)
    return os.path.join(
        args.output_dir,
        now.strftime("%y%m%d"),
        f"{output_subdir}-{now.strftime('%Y%m%d-%H%M%S')}",
    )


def benchmark_metadata(example: dict) -> dict | None:
    if "gpqa_subset" not in example:
        return None
    return {
        "type": "gpqa",
        "subset": example["gpqa_subset"],
        "record_id": example["gpqa_record_id"],
        "high_level_domain": example["gpqa_high_level_domain"],
        "subdomain": example["gpqa_subdomain"],
        "shuffle_seed": example["gpqa_shuffle_seed"],
        "choice_order": example["gpqa_choice_order"],
        "choices": {
            "A": example["gpqa_choice_a"],
            "B": example["gpqa_choice_b"],
            "C": example["gpqa_choice_c"],
            "D": example["gpqa_choice_d"],
        },
    }


def load_components(args: argparse.Namespace) -> tuple[GradCuit, object]:
    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    reward_model = SingleVerifierRewardModel(
        data_name=args.dataset,
        prompt_idx=args.solver_prompt_idx,
        model_name=args.model_name_or_path,
        request_model_name=args.vllm_verifier_model_name,
        vllm_base_url=args.vllm_base_url,
        vllm_timeout=args.vllm_timeout,
    )
    gradcuit = GradCuit(
        model=model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        insert_prefix_text=args.insert_prefix_text,
        optimize_layer_idx=args.optimize_layer_idx,
        log_fn=log,
        raw_log_fn=print,
    )
    dataset = get_dataset(
        args.dataset,
        tokenizer=tokenizer,
        prompt_idx=args.solver_prompt_idx,
        seed=args.seed,
    )
    log(f"Loaded dataset size: {len(dataset)}")
    return gradcuit, dataset


def evaluate_dataset(
    args: argparse.Namespace,
    *,
    gradcuit: GradCuit,
    dataset,
    output_dir: str,
) -> dict:
    counters = {
        "original_correct": 0,
        "optimized_correct": 0,
        "total": 0,
        "update_count": 0,
        "original_length": 0,
        "optimized_length": 0,
        "fitten_length": 0,
    }
    start_index = args.start_data_idx
    if args.resume:
        logistics = torch.load(
            os.path.join(output_dir, "logistics.pt"),
            weights_only=True,
        )
        start_index = int(logistics.pop("start_idx"))
        for key in counters:
            counters[key] = int(logistics[key])

    remaining = len(dataset) - start_index
    if args.num_data > 0:
        run_budget = args.num_data - counters["total"] if args.resume else args.num_data
        remaining = min(remaining, max(0, run_budget))
    end_index = start_index + remaining
    if remaining <= 0:
        raise ValueError("No examples remain for the requested range.")

    save_run_arguments(output_dir, args)
    sample_output_dir = os.path.join(output_dir, "test")
    os.makedirs(sample_output_dir, exist_ok=True)
    run_start = time.time()

    for processed_count, index in enumerate(range(start_index, end_index), start=1):
        example_start = time.time()
        example = dataset[index]
        true_answer = extract_true_answer(example["answer"], args.dataset)
        if true_answer is None:
            raise ValueError(f"Unable to extract ground-truth answer for index {index}.")

        original_output, latents, _ = gradcuit.original_generation(
            input_text=example["formatted"]
        )
        (
            optimized_output,
            reward_history,
            output_history,
            reward_details_history,
            latent_delta_history,
            original_length,
            optimized_length,
            update_length,
        ) = gradcuit.optimized_generation(
            question=example["question"],
            input_text=example["formatted"],
            original_output=original_output,
            original_latents_list=latents,
            max_num_steps=args.max_num_steps,
            lr=args.lr,
            optimizer_name=args.optimizer,
            grad_clip=args.grad_clip,
            reward_threshold=args.reward_threshold,
        )

        original_answer = extract_answer(
            original_output,
            args.dataset,
            args.solver_prompt_idx,
            model_name=args.model_name_or_path,
        )
        optimized_answer = extract_answer(
            optimized_output,
            args.dataset,
            args.solver_prompt_idx,
            model_name=args.model_name_or_path,
        )
        original_correct = judge_answer(
            original_output,
            true_answer,
            args.dataset,
            args.solver_prompt_idx,
        )
        optimized_correct = judge_answer(
            optimized_output,
            true_answer,
            args.dataset,
            args.solver_prompt_idx,
        )

        counters["original_correct"] += int(original_correct)
        counters["optimized_correct"] += int(optimized_correct)
        counters["total"] += 1
        counters["update_count"] += len(latent_delta_history)
        counters["original_length"] += original_length
        counters["optimized_length"] += optimized_length
        if latent_delta_history:
            counters["fitten_length"] += optimized_length - update_length

        optimized_steps = []
        for history_index, (output, reward, details) in enumerate(
            zip(output_history, reward_history, reward_details_history)
        ):
            latent_delta = (
                latent_delta_history[history_index - 1]
                if 0 < history_index <= len(latent_delta_history)
                else {}
            )
            optimized_steps.append(
                {
                    "output": output,
                    "reward": float(reward),
                    "reward_details": details,
                    "latent_delta": latent_delta,
                }
            )

        record = {
            "question": example["question"],
            "prompt": example["formatted"],
            "insert_prefix_text": args.insert_prefix_text,
            "optimize_layer_idx": args.optimize_layer_idx,
            "optimize_target_type": gradcuit.optimize_target_type,
            "original_output": original_output,
            "original_answer": original_answer,
            "optimized_steps": optimized_steps,
            "seek_step_count": len(latent_delta_history),
            "groundtruth": true_answer,
            "final_answer": optimized_answer,
            "final_correct": optimized_correct,
        }
        metadata = benchmark_metadata(example)
        if metadata is not None:
            record["benchmark_metadata"] = metadata
        with open(
            os.path.join(sample_output_dir, f"{index}_data.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(record, file, ensure_ascii=False, indent=2)

        torch.save(
            {**counters, "start_idx": index + 1},
            os.path.join(output_dir, "logistics.pt"),
        )
        elapsed = time.time() - run_start
        average = elapsed / processed_count
        examples_left = remaining - processed_count
        log(
            f"Example {processed_count}/{remaining} finished in "
            f"{time.time() - example_start:.2f}s; ETA {format_duration(average * examples_left)}"
        )

    total = counters["total"]
    result = {
        **counters,
        "output_dir": output_dir,
        "original_accuracy": counters["original_correct"] / total,
        "optimized_accuracy": counters["optimized_correct"] / total,
        "average_update_length": counters["update_count"] / total,
        "average_original_length": counters["original_length"] / total,
        "average_optimized_length": counters["optimized_length"] / total,
        "average_fitten_length": counters["fitten_length"] / total,
    }
    log(f"Original accuracy: {result['original_accuracy']:.4f}")
    log(f"Optimized accuracy: {result['optimized_accuracy']:.4f}")
    return result


def main(args: argparse.Namespace) -> dict:
    validate_args(args)
    gradcuit, dataset = load_components(args)
    output_dir = resolve_output_directory(args)
    return evaluate_dataset(
        args,
        gradcuit=gradcuit,
        dataset=dataset,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parsed_args = parse_args()
    configure_warnings(parsed_args.show_warnings)
    log("Starting GradCuit.")
    main(parsed_args)
