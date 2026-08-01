BOXED_MATH_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
JSON_MATH_INSTRUCTION = (
    "Please reason step by step, and put your final answer in a json dict with exactly "
    'one key "answer", for example {"answer": "1.234"}.'
)
BOXED_GPQA_INSTRUCTION = (
    "Please reason step by step. This is a multiple-choice question with answer choices "
    "labeled A, B, C, and D. Your final answer must be a single uppercase letter and must "
    "be placed within \\boxed{}, for example \\boxed{B}."
)
JSON_GPQA_INSTRUCTION = (
    "Please reason step by step. This is a multiple-choice question with answer choices "
    "labeled A, B, C, and D. Your final answer must be a single uppercase letter and must "
    'be placed in a json dict with exactly one key "answer", for example {"answer": "B"}.'
)


def get_solver_prompt(question: str, data_name: str, prompt_idx: int) -> list[dict[str, str]]:
    lower_name = data_name.lower()
    is_gpqa = "gpqa" in lower_name
    if prompt_idx not in (0, 1):
        raise ValueError(f"Unknown solver_prompt_idx: {prompt_idx}")

    if is_gpqa:
        instruction = BOXED_GPQA_INSTRUCTION if prompt_idx == 0 else JSON_GPQA_INSTRUCTION
    else:
        instruction = BOXED_MATH_INSTRUCTION if prompt_idx == 0 else JSON_MATH_INSTRUCTION
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]
