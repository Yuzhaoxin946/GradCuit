_BOXED_VERDICT_INSTRUCTION = (
    "First verify carefully step by step. On the final line, put the final verification verdict inside "
    "\\boxed{} using exactly \\boxed{True} or \\boxed{False}.\n"
)


def is_gpqa_dataset(data_name: str | None) -> bool:
    if not data_name:
        return False
    lowered = data_name.lower()
    return (
        lowered == "gpqa_diamond"
        or "gpqa_diamond" in lowered
        or "/gpqa" in lowered
        or lowered.endswith("gpqa")
    )


def get_single_verifier_prompt(
    question: str,
    extracted_answer: str,
    data_name: str,
) -> str:
    if is_gpqa_dataset(data_name):
        return (
            "You are a critical verifier for graduate-level scientific multiple-choice questions. "
            "You will be given the original question, which already contains answer choices labeled A, B, C, and D, "
            "plus one extracted final answer letter.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"FINAL ANSWER:\n{extracted_answer}\n\n"
            "INSTRUCTIONS:\n"
            "1. Treat this as a single-choice question. Exactly one option should be correct.\n"
            "2. Read the FINAL ANSWER as a letter and map it to the corresponding answer choice in the QUESTION.\n"
            "3. Verify only whether that selected choice is correct. Do not score missing reasoning steps.\n"
            "4. If the letter does not map cleanly to one of the provided answer choices, the verdict is False.\n"
            "5. If the selected option is the correct answer, the verdict is True.\n"
            "6. If the selected option is incorrect, ambiguous, or not uniquely justified by the question, the verdict is False.\n"
            f"7. {_BOXED_VERDICT_INSTRUCTION}"
        )

    return (
        "You are a critical verifier for mathematical questions. "
        "You will be given the original question and one final answer. "
        "Decide whether that answer is correct for the question.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"FINAL ANSWER:\n{extracted_answer}\n\n"
        "INSTRUCTIONS:\n"
        "1. Verify only the final answer. Do not evaluate any missing reasoning steps.\n"
        "2. Do not solve the problem independently from scratch unless a tiny auxiliary calculation is strictly "
        "necessary for verification.\n"
        "3. Prefer reverse verification methods such as substitution, plugging the answer back into the original "
        "conditions, checking algebraic consistency, checking boundary cases, or other direct validation targeted "
        "at the proposed answer.\n"
        "4. Accept mathematically equivalent forms when they represent the same final answer.\n"
        "5. If the final answer is correct, the verdict is True.\n"
        "6. If the final answer is incorrect, the verdict is False.\n"
        f"7. {_BOXED_VERDICT_INSTRUCTION}"
    )
