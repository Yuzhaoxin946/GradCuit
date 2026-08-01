"""Compatibility layer for the original GradCuit experiment evaluator.

The extraction and judging implementation lives in ``extract_judge_answer`` and
is migrated from LatentSeekPro.  These wrappers preserve the compact call
signatures used by this repository's main and reward modules.
"""

from extract_judge_answer import (
    extract_answer as _extract_answer,
    extract_true_answer as _extract_true_answer,
    judge_answer as _judge_answer,
)


def extract_true_answer(text: str, data_name: str) -> str | None:
    return _extract_true_answer(text, name=data_name)


def extract_answer(
    text: str,
    data_name: str,
    prompt_idx: int,
    model_name: str = "Qwen2.5-7B-Instruct",
) -> str | None:
    return _extract_answer(
        text,
        data_name=data_name,
        prompt_idx=prompt_idx,
        model_name=model_name,
    )


def judge_answer(
    response: str,
    true_answer: str,
    data_name: str,
    prompt_idx: int,
) -> bool:
    return bool(
        _judge_answer(
            response,
            true_answer,
            data_name=data_name,
            prompt_idx=prompt_idx,
        )
    )
