import os
import re
from typing import Any

from openai import OpenAI

from evaluation import extract_answer
from prompts.single_verifier_prompts import get_single_verifier_prompt


class SingleVerifierRewardModel:
    def __init__(
        self,
        *,
        data_name: str,
        prompt_idx: int,
        model_name: str,
        request_model_name: str,
        vllm_base_url: str,
        vllm_timeout: float,
    ):
        self.data_name = data_name
        self.prompt_idx = prompt_idx
        self.model_name = model_name
        self.request_model_name = request_model_name
        self.vllm_base_url = vllm_base_url
        self.vllm_timeout = float(vllm_timeout)
        self.vllm_api_key = os.getenv("VLLM_API_KEY", "EMPTY")

    def _backend_error_message(self) -> str:
        return (
            "Single verifier request failed. Ensure the OpenAI-compatible verifier "
            f"server is reachable at {self.vllm_base_url} and exposes "
            f"model {self.request_model_name}."
        )

    @staticmethod
    def _is_timeout_error(exc: BaseException) -> bool:
        seen = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if current.__class__.__name__ in {"APITimeoutError", "ReadTimeout", "TimeoutError"}:
                return True
            if "timed out" in str(current).lower():
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _response_text(response: Any) -> str:
        if not getattr(response, "choices", None):
            return ""
        content = response.choices[0].message.content
        if isinstance(content, list):
            return "".join(
                part.text if hasattr(part, "text") else str(part.get("text", ""))
                for part in content
            )
        return "" if content is None else str(content)

    @staticmethod
    def _extract_verifier_answer_fragment(verifier_response: Any) -> str | None:
        if verifier_response is None:
            return None

        text = str(verifier_response)
        matches = list(
            re.finditer(r"(?:\\+)?(?:boxed|fbox)\s*\{", text, flags=re.IGNORECASE)
        )
        for match in reversed(matches):
            left_brace_idx = text.find("{", match.start(), match.end())
            if left_brace_idx < 0:
                continue

            depth = 1
            cursor = left_brace_idx + 1
            while cursor < len(text):
                if text[cursor] == "{":
                    depth += 1
                elif text[cursor] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[left_brace_idx + 1:cursor].strip()
                cursor += 1
        return None

    @staticmethod
    def _normalize_verifier_answer(answer: Any) -> bool | None:
        if answer is None:
            return None

        normalized = str(answer).replace("*", "").strip().strip("$").strip()
        if not normalized:
            return None

        for _ in range(3):
            match = re.fullmatch(
                r"\\(?:text|textrm|textbf|mathrm|mathbf|operatorname)\s*\{(.*)\}",
                normalized,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if not match:
                break
            normalized = match.group(1).strip()

        words = re.findall(r"[a-zA-Z]+", normalized.casefold())
        if not words:
            return None
        if words[0] == "true":
            return True
        if words[0] == "false":
            return False
        return None

    @classmethod
    def _boxed_verdict(cls, text: str) -> bool | None:
        answer = cls._extract_verifier_answer_fragment(text)
        return cls._normalize_verifier_answer(answer)

    def _verify(self, prompt: str) -> tuple[str, bool | None]:
        try:
            with OpenAI(
                base_url=self.vllm_base_url,
                api_key=self.vllm_api_key,
                timeout=self.vllm_timeout,
            ) as client:
                response = client.chat.completions.create(
                    model=self.request_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
        except Exception as exc:
            if self._is_timeout_error(exc):
                raise TimeoutError(self._backend_error_message()) from exc
            raise RuntimeError(self._backend_error_message()) from exc
        response_text = self._response_text(response)
        return response_text, self._boxed_verdict(response_text)

    def get_reward(self, question: str, solution: str) -> tuple[float, dict[str, Any]]:
        extracted = extract_answer(
            solution,
            self.data_name,
            self.prompt_idx,
            model_name=self.model_name,
        )
        details: dict[str, Any] = {
            "mode": "single_verifier",
            "extracted_answer": extracted,
            "extraction_succeeded": extracted is not None and bool(str(extracted).strip()),
        }
        if not details["extraction_succeeded"]:
            details["verifier"] = {
                "prompt": "",
                "output": "",
                "approved": False,
                "format_valid": False,
                "reward": -1.0,
                "skipped": True,
                "timed_out": False,
            }
            return -1.0, details

        prompt = get_single_verifier_prompt(question, str(extracted), self.data_name)
        try:
            output, verdict = self._verify(prompt)
        except TimeoutError as exc:
            details["verifier"] = {
                "prompt": prompt,
                "output": "",
                "approved": False,
                "format_valid": False,
                "reward": -1.0,
                "skipped": False,
                "timed_out": True,
                "error": str(exc),
            }
            return -1.0, details

        approved = verdict is True
        reward = 0.0 if approved else -1.0
        details["verifier"] = {
            "prompt": prompt,
            "output": output,
            "approved": approved,
            "format_valid": verdict is not None,
            "reward": reward,
            "skipped": False,
            "timed_out": False,
        }
        return reward, details
