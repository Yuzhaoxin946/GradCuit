import json
import math
import re

try:
    from math_verify import parse, verify
except ModuleNotFoundError:
    parse = None
    verify = None

try:
    from .grader import math_equal_process
except ModuleNotFoundError:
    math_equal_process = None
from .math_equivalent_MATH import is_equiv
from .parse_utils_qwen import extract_answer as extract_fn


SCIBENCH_REL_TOL = 0.05
GPQA_VALID_CHOICES = {"A", "B", "C", "D"}


def _require_math500_judge_dependencies() -> None:
    missing_dependencies = []
    if parse is None or verify is None:
        missing_dependencies.append("math-verify")
    if math_equal_process is None:
        missing_dependencies.append("latex2sympy2-based math grader")

    if missing_dependencies:
        joined = ", ".join(missing_dependencies)
        raise ImportError(
            "Math-500 judging dependencies are unavailable: "
            f"{joined}. Install the evaluation dependencies before running Math-500 experiments."
        )


def _is_scibench_name(name: str) -> bool:
    return "scibench" in name.lower()


def _is_gpqa_name(name: str) -> bool:
    lowered = name.lower()
    return lowered == "gpqa_diamond" or "gpqa_diamond" in lowered or "/gpqa" in lowered or lowered.endswith("gpqa")


def _serialize_scibench_label(answer_number, unit=None):
    payload = {
        "answer_number": "" if answer_number is None else str(answer_number).strip(),
        "unit": "" if unit is None else str(unit).strip(),
    }
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _parse_scibench_label(label):
    if isinstance(label, dict):
        return (
            str(label.get("answer_number", "")).strip(),
            str(label.get("unit", "")).strip(),
        )

    label_text = "" if label is None else str(label).strip()
    if not label_text:
        return "", ""

    try:
        payload = json.loads(label_text)
    except (TypeError, json.JSONDecodeError):
        return label_text, ""

    if isinstance(payload, dict):
        return (
            str(payload.get("answer_number", "")).strip(),
            str(payload.get("unit", "")).strip(),
        )
    return label_text, ""


def _last_boxed_only_string(text):
    idx = text.rfind("oxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None

    right_brace_idx = None
    num_left_braces_open = 0
    cursor = idx
    while cursor < len(text):
        if text[cursor] == "{":
            num_left_braces_open += 1
        elif text[cursor] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = cursor
                break
        cursor += 1

    if right_brace_idx is None:
        return None
    return text[idx:right_brace_idx + 1]


def _remove_boxed(text):
    left = "oxed{"
    if text is None:
        return None
    try:
        assert text[:len(left)] == left
        assert text[-1] == "}"
        answer = text[len(left):-1]
        if "=" in answer:
            answer = answer.split("=")[-1].lstrip(" ")
        return answer
    except Exception:
        return None


def _cleanup_scibench_candidate(text):
    if text is None:
        return None

    candidate = str(text).strip().strip("`").strip()
    if not candidate:
        return None

    candidate = candidate.replace("\\,", "")
    candidate = candidate.replace(",", "")
    candidate = candidate.replace("−", "-")
    candidate = candidate.replace("–", "-")
    candidate = re.sub(r"\\text\{([^}]*)\}", r"\1", candidate)
    candidate = re.sub(r"^\$+|\$+$", "", candidate)
    if "=" in candidate:
        candidate = candidate.split("=")[-1].strip()
    candidate = candidate.strip().rstrip(".,;:")
    return candidate or None


def _extract_json_final_answer(text):
    if text is None:
        return None

    def extract_payload(payload):
        if isinstance(payload, dict) and set(payload.keys()) == {"answer"}:
            value = payload["answer"]
            return value if isinstance(value, str) else str(value)
        return None

    stripped = text.strip("` \n")
    try:
        extracted = extract_payload(json.loads(stripped))
        if extracted is not None:
            return extracted
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL):
        try:
            extracted = extract_payload(json.loads(match.group(0)))
        except json.JSONDecodeError:
            continue
        if extracted is not None:
            return extracted
    return None


def _extract_scibench_answer(text):
    if text is None:
        return None

    boxed = _remove_boxed(_last_boxed_only_string(text))
    if boxed is not None:
        return _cleanup_scibench_candidate(boxed)

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if lines:
        return _cleanup_scibench_candidate(lines[-1])
    return None


def _normalize_gpqa_choice(choice):
    if choice is None:
        return None

    text = str(choice).strip().strip("`").strip()
    if not text:
        return None

    text = text.strip().strip("\"'").strip()
    text = re.sub(r"^[\(\[\{<\s]+|[\)\]\}>\s\.,;:]+$", "", text)
    text = text.upper()
    return text if text in GPQA_VALID_CHOICES else None


def _extract_gpqa_choice(text):
    if text is None:
        return None

    candidates = []

    def _remember(value):
        normalized = _normalize_gpqa_choice(value)
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    json_answer = _extract_json_final_answer(text)
    _remember(json_answer)

    boxed = _remove_boxed(_last_boxed_only_string(text))
    _remember(boxed)

    explicit_answer_match = re.search(
        r"(?:final answer|final_answer|my answer|correct answer|answer|choice|option)\s*(?:is|:)\s*([^\n]+)",
        str(text),
        flags=re.IGNORECASE,
    )
    if explicit_answer_match:
        explicit_segment = explicit_answer_match.group(1)
        explicit_letters = {
            letter.upper()
            for letter in re.findall(r"\b([ABCD])\b", explicit_segment, flags=re.IGNORECASE)
        }
        if len(explicit_letters) > 1:
            return None

    explicit_patterns = (
        r"(?:final answer|final_answer|my answer|correct answer|answer|choice|option)\s*(?:is|:)\s*[\"']?\(?([ABCD])\)?[\"']?\b",
        r"^\s*\(?([ABCD])\)?\s*$",
    )
    for pattern in explicit_patterns:
        matches = re.findall(pattern, str(text), flags=re.IGNORECASE | re.MULTILINE)
        for match in matches:
            _remember(match)

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if lines:
        _remember(lines[-1])

    if len(candidates) == 1:
        return candidates[0]
    return None


def _extract_scibench_unit_scale(unit):
    if unit is None:
        return None
    match = re.search(r"10\^\{?\s*([+-]?\d+)\s*\}?", str(unit))
    if not match:
        return None
    return float(match.group(1))


def _parse_scibench_numeric_value(text):
    candidate = _cleanup_scibench_candidate(text)
    if not candidate:
        return None, False

    scientific_pattern = re.compile(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:\\times|×|\*)\s*10\^\{?\s*([+-]?\d+)\s*\}?",
        flags=re.IGNORECASE,
    )
    match = scientific_pattern.search(candidate)
    if match:
        coefficient = float(match.group(1))
        exponent = float(match.group(2))
        return coefficient * (10 ** exponent), True

    fragments = [candidate]
    if candidate.split():
        fragments.append(candidate.split()[0].rstrip(".,;:"))

    numeric_match = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", candidate)
    if numeric_match:
        fragments.append(numeric_match.group(0))

    for fragment in fragments:
        try:
            return float(fragment), ("e" in fragment.lower())
        except (TypeError, ValueError):
            continue
    return None, False


def _scibench_candidate_values(text, unit):
    parsed_value, has_explicit_scale = _parse_scibench_numeric_value(text)
    if parsed_value is None:
        return []

    values = {parsed_value}
    unit_scale = _extract_scibench_unit_scale(unit)
    if unit_scale is not None and not has_explicit_scale:
        values.add(parsed_value * (10 ** unit_scale))
    return list(values)


def _scibench_gold_values(answer_number, unit):
    gold_value, _ = _parse_scibench_numeric_value(answer_number)
    if gold_value is None:
        return []

    values = {gold_value}
    unit_scale = _extract_scibench_unit_scale(unit)
    if unit_scale is not None:
        values.add(gold_value * (10 ** unit_scale))
    return list(values)


def _judge_scibench_answer(input_value, label, extract=True, prompt_idx=0):
    answer_number, unit = _parse_scibench_label(label)
    if extract:
        input_value = extract_answer(input_value, data_name="scibench", prompt_idx=prompt_idx)

    prediction_values = _scibench_candidate_values(input_value, unit)
    gold_values = _scibench_gold_values(answer_number, unit)
    for prediction in prediction_values:
        for gold in gold_values:
            if math.isclose(prediction, gold, rel_tol=SCIBENCH_REL_TOL):
                return True
    return False


def extract_true_answer(text, name="gsm8k", unit=None):
    """
    Extract answer from text.

    Args:
        text: input text
        name: name of the dataset
        unit: optional unit field for SciBench labels

    Returns:
        answer: extracted answer
    """
    lower_name = name.lower()
    if "gsm8k" in lower_name:
        label = text.split("#### ")[1]
        return label
    elif "math-500" in lower_name:
        return text
    elif "aime_2024" in lower_name:
        return text
    elif _is_gpqa_name(lower_name):
        return _normalize_gpqa_choice(text)
    elif _is_scibench_name(lower_name):
        answer_number, parsed_unit = _parse_scibench_label(text)
        final_unit = parsed_unit if unit is None else unit
        if final_unit:
            return _serialize_scibench_label(answer_number, final_unit)
        return answer_number
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def judge_answer(input, label, data_name="gsm8k", extract=True, prompt_idx=0):
    """Score.

    Judge whether the answer is correct or not.

    Args:
        input (str): model response
        label (str): ground truth
        data_name (str): dataset name
        extract (bool): whether to extract answer from model response
        prompt_idx (int): index of the solver prompt (different format)

    Returns:
        bool: True if the answer is correct, False otherwise
    """
    lower_name = data_name.lower()
    if "gsm8k" in lower_name:
        if extract:
            input = extract_answer(input, data_name="gsm8k", prompt_idx=prompt_idx)
        return input == label
    elif "math-500" in lower_name:
        _require_math500_judge_dependencies()
        if extract:
            input = extract_answer(input, data_name="MATH-500", prompt_idx=prompt_idx)

        hf_input = parse(input)
        hf_verifier_judge = verify(label, hf_input)
        if hf_verifier_judge:
            return True

        qwen_verifier_judge = math_equal_process((label, input))
        if qwen_verifier_judge:
            return True

        exact_judge = str(input) == str(label)
        if exact_judge:
            return True

        math_500_judge = is_equiv(str(label), str(input))
        if math_500_judge:
            return True
        return False

    elif "aime_2024" in lower_name:
        if extract:
            input = extract_answer(input, data_name="AIME_2024", prompt_idx=prompt_idx)
            input = str(input)
            label = str(label)
        return input == label

    elif _is_gpqa_name(lower_name):
        if extract:
            input = extract_answer(input, data_name="gpqa_diamond", prompt_idx=prompt_idx)
        return _normalize_gpqa_choice(input) == _normalize_gpqa_choice(label)

    elif _is_scibench_name(lower_name):
        return _judge_scibench_answer(input, label, extract=extract, prompt_idx=prompt_idx)

    else:
        raise ValueError(f"Unknown dataset name: {data_name} for judge answer")


def extract_answer(text, data_name="gsm8k", prompt_idx=0, model_name="Qwen2.5-7B-Instruct"):
    """
    Extract answer from model response.

    Args:
        text: Raw response string from the language model
        data_name: dataset name
        prompt_idx: index of the solver prompt (different format)

    Returns:
        answer: extracted answer
    """
    lower_name = data_name.lower()
    if "gsm8k" in lower_name:
        if prompt_idx == 0:
            if "qwen2.5-1.5b-instruct" in model_name.lower():
                return _extract_qwen25_1_5B_answer(text)
            return _extract_answer(text)

        elif prompt_idx == 1:
            final_answer = _extract_json_final_answer(text)
            if final_answer is not None:
                return _extract_answer(final_answer)
            return None

        else:
            raise ValueError(f"Unknown prompt index: {prompt_idx} for extract answer")

    elif "math-500" in lower_name:
        if prompt_idx == 0:
            return extract_fn(text, data_name="math")

        elif prompt_idx == 1:
            final_answer = _extract_json_final_answer(text)
            if final_answer is None:
                return None
            final_answer = final_answer.replace("\n", "")
            final_answer = final_answer.replace("\"", "")
            final_answer = final_answer.replace("\'", "")
            return final_answer

    elif "aime_2024" in lower_name:
        if prompt_idx == 0:
            return _extract_answer(text)

        elif prompt_idx == 1:
            final_answer = _extract_json_final_answer(text)
            if final_answer is not None:
                return _extract_answer(final_answer)
            return None

        else:
            raise ValueError(f"Unknown prompt index: {prompt_idx} for extract answer")

    elif _is_gpqa_name(lower_name):
        if prompt_idx == 0:
            return _extract_gpqa_choice(text)

        elif prompt_idx == 1:
            final_answer = _extract_json_final_answer(text)
            if final_answer is not None:
                parsed_final = _extract_gpqa_choice(final_answer)
                if parsed_final is not None:
                    return parsed_final
            return None

        else:
            raise ValueError(f"Unknown prompt index: {prompt_idx} for extract answer")

    elif _is_scibench_name(lower_name):
        if prompt_idx == 0:
            return _extract_scibench_answer(text)

        elif prompt_idx == 1:
            final_answer = _extract_json_final_answer(text)
            if final_answer is not None:
                return _cleanup_scibench_candidate(final_answer)
            return None

        else:
            raise ValueError(f"Unknown prompt index: {prompt_idx} for extract answer")

    else:
        raise ValueError(f"Unknown dataset name: {data_name} for extract answer")



######################
#       MATH         #
######################

def extract_MATH_solution(solution_str: str):
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        extracted final answer
    """""
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        processed_str = solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>.*?(\\boxed{.*}).*?</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))

    if not matches:
        answer_pattern = r'\\boxed{(.*)}'
        matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        print("[Error] No valid answer tags found")
        return None
    final_answer = matches[-1].group(1).strip()
    return final_answer


def _extract_answer(text):
    """
    Extract numerical answer from generated text.
    handling various edge cases.
    
    Args:
        text (str): Generated text to extract answer from.
    
    Returns:
        str or None: Extracted numerical answer, or None if not found.
    """
    if text is None:
        return None
    
    text = text.strip()

    def clean_number(num_str):
        """Remove currency symbols, commas, and whitespace."""
        num_str = re.sub(r'[$€£¥]', '', num_str)
        num_str = re.sub(r',', '', num_str)
        num_str = re.sub(r'\s', '', num_str)
        return num_str

    ### Several Corner Cases ###
    # 1. \boxed{}
    boxed_pattern = r"\\boxed\{\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*\}"
    match = re.search(boxed_pattern, text, re.IGNORECASE)
    if match:
        return clean_number(match.group(1))
    
    # 2. Answer:
    answer_pattern = r"Answer:\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    match = re.search(answer_pattern, text, re.IGNORECASE)
    if match:
        return clean_number(match.group(1))
    
    # 3. = (use the last match to avoid early intermediate equations)
    #equals_pattern = r"=\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    #matches = re.findall(equals_pattern, text)
    #if matches:
    #    return clean_number(matches[-1])

    # 4. With currency unit
    currency_pattern = r"is\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*(?:dollars|euros|pounds|yen)"
    match = re.search(currency_pattern, text, re.IGNORECASE)
    if match:
        return clean_number(match.group(1))

    # 5. Search from the last line of the text upwards, matching independent numbers
    lines = text.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line:
            final_num_pattern = r"([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*$"
            match = re.search(final_num_pattern, line)
            if match:
                return clean_number(match.group(1))

    # 6. Returns the last matching number in the text
    number_pattern = r"([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    matches = re.findall(number_pattern, text)
    if matches:
        return clean_number(matches[-1])

    return None


def _extract_qwen25_1_5B_answer(text):
    """
    Extract numerical answer from generated text for Qwen-2.5 1.5B model.
    handling various edge cases.

    Args:
        text (str): Generated text to extract answer from.

    Returns:
        str or None: Extracted numerical answer, or None if not found.
    """
    if text is None:
        return None

    text = text.strip()

    def clean_number(num_str):
        """Remove currency symbols, commas, and whitespace."""
        num_str = re.sub(r'[$€£¥]', '', num_str)
        num_str = re.sub(r',', '', num_str)
        num_str = re.sub(r'\s', '', num_str)
        return num_str

    ### Several Corner Cases ###
    # 1. \boxed{}
    boxed_pattern = r"\\boxed\{\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)\s*\}"
    match = re.search(boxed_pattern, text, re.IGNORECASE)
    if match:
        return clean_number(match.group(1))

    # 2. he answer is
    answer_pattern = r"he answer is\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    match = re.search(answer_pattern, text, re.IGNORECASE)
    if match:
        return clean_number(match.group(1))

    # 3. final answer is
    answer_pattern = r"final answer is\s*([$€£¥]?\s*-?\s*[\d,]+(?:\.\d+)?)"
    match = re.search(answer_pattern, text, re.IGNORECASE)
    if match:
        return clean_number(match.group(1))

    # 4. Returns the last matching number in the text
    number_pattern = r'\d+(?:,\d+)*(?:\.\d+)?'
    matches = re.findall(number_pattern, text)
    if matches:
        return clean_number(matches[-1])

    return None
