from contextlib import contextmanager
from typing import Callable

import torch


class GradCuit:
    def __init__(
        self,
        *,
        model,
        reward_model,
        tokenizer,
        max_new_tokens: int,
        device: str,
        insert_prefix_text: str,
        optimize_layer_idx: int,
        log_fn: Callable[[str], None] | None = None,
        raw_log_fn: Callable[[str], None] | None = None,
    ):
        self.model = model
        self.embeddings = model.get_input_embeddings()
        self.model.requires_grad_(False)
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.max_new_tokens = int(max_new_tokens)
        self.device = device
        self.log_fn = log_fn or print
        self.raw_log_fn = raw_log_fn or print
        self.optimize_layer_idx = int(optimize_layer_idx)

        if self.reward_model is None:
            raise ValueError("reward_model is required.")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if self.optimize_layer_idx < 0:
            raise ValueError("optimize_layer_idx must be non-negative.")
        if insert_prefix_text is None or not str(insert_prefix_text).strip():
            raise ValueError("insert_prefix_text must be a non-empty string.")

        self.optimize_target_type = (
            "embedding" if self.optimize_layer_idx == 0 else "block_input_hidden_state"
        )
        self.insert_prefix_text = insert_prefix_text
        self.prefix_token_ids = self._build_prefix_token_ids(insert_prefix_text)
        self.decoder_layers = None
        if self.optimize_layer_idx > 0:
            self.decoder_layers = self._resolve_decoder_layers()
            if self.optimize_layer_idx >= len(self.decoder_layers):
                raise ValueError(
                    "optimize_layer_idx is out of range for the model decoder layers: "
                    f"{self.optimize_layer_idx} >= {len(self.decoder_layers)}."
                )

        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0

    def _build_prefix_token_ids(self, prefix_text: str) -> torch.Tensor:
        token_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        if not token_ids:
            raise ValueError("insert_prefix_text tokenized to an empty sequence.")
        return torch.tensor(token_ids, dtype=torch.long)

    @staticmethod
    def _resolve_nested_attr(root, attr_path: str):
        value = root
        for attribute in attr_path.split("."):
            if not hasattr(value, attribute):
                return None
            value = getattr(value, attribute)
        return value

    def _resolve_decoder_layers(self):
        for attr_path in ("model.layers", "model.decoder.layers", "transformer.h"):
            layers = self._resolve_nested_attr(self.model, attr_path)
            if layers is None:
                continue
            try:
                if len(layers) > 0:
                    return layers
            except TypeError:
                continue
        raise ValueError(
            "Unable to locate decoder layers. Tried model.layers, "
            "model.decoder.layers, and transformer.h."
        )

    def _target_layer(self):
        if self.decoder_layers is None:
            self.decoder_layers = self._resolve_decoder_layers()
        return self.decoder_layers[self.optimize_layer_idx]

    @contextmanager
    def _temporary_prefix_hidden_override(
        self,
        *,
        target_prefix_state: torch.Tensor,
        prefix_start: int,
        prefix_end: int,
    ):
        if self.optimize_layer_idx == 0:
            yield
            return

        def hook(_module, args, kwargs):
            kwargs = kwargs or {}
            if args:
                hidden_states = args[0]
                rest_args = args[1:]
                from_args = True
            else:
                hidden_states = kwargs.get("hidden_states")
                rest_args = ()
                from_args = False

            if hidden_states is None or hidden_states.ndim < 3:
                return args, kwargs
            if hidden_states.size(1) < prefix_end:
                return args, kwargs

            replacement = target_prefix_state.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if replacement.size(0) != hidden_states.size(0):
                if replacement.size(0) != 1:
                    raise ValueError(
                        "target_prefix_state batch size must be 1 or match the model batch size."
                    )
                replacement = replacement.expand(hidden_states.size(0), -1, -1)

            updated = hidden_states.clone()
            updated[:, prefix_start:prefix_end, :] = replacement
            if from_args:
                return (updated, *rest_args), kwargs
            updated_kwargs = dict(kwargs)
            updated_kwargs["hidden_states"] = updated
            return args, updated_kwargs

        handle = self._target_layer().register_forward_pre_hook(hook, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()

    @staticmethod
    def _prefix_state_to_list(prefix_state: torch.Tensor) -> list[torch.Tensor]:
        return [prefix_state[:, index].clone() for index in range(prefix_state.shape[1])]

    @staticmethod
    def _build_optimizer(name: str, parameters, lr: float):
        normalized = name.lower()
        if normalized == "adam":
            return torch.optim.Adam(parameters, lr=lr)
        if normalized == "sgd":
            return torch.optim.SGD(parameters, lr=lr)
        if normalized == "muon":
            muon_class = getattr(torch.optim, "Muon", None)
            if muon_class is None:
                raise ValueError(
                    "optimizer='muon' requires a PyTorch build that provides torch.optim.Muon."
                )
            return muon_class(parameters, lr=lr)
        raise ValueError(f"Unknown optimizer: {name}. Expected adam, sgd, or muon.")

    def _extract_initial_prefix_state(
        self,
        *,
        prompt_ids: torch.Tensor,
        prefix_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.optimize_layer_idx == 0:
            with torch.no_grad():
                return self.embeddings(prefix_ids).detach()

        full_input_ids, full_attention_mask = self._full_prompt_prefix_ids_and_mask(
            prompt_ids=prompt_ids,
            prefix_ids=prefix_ids,
            prompt_attention_mask=prompt_attention_mask,
        )
        with torch.no_grad():
            output = self.model(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise ValueError(
                "Model did not return hidden_states for hidden-layer prefix optimization."
            )
        if self.optimize_layer_idx >= len(hidden_states):
            raise ValueError(
                "Model returned too few hidden states for optimize_layer_idx "
                f"{self.optimize_layer_idx}."
            )
        prompt_length = prompt_ids.shape[1]
        prefix_length = prefix_ids.shape[1]
        return hidden_states[self.optimize_layer_idx][
            :, prompt_length:prompt_length + prefix_length, :
        ].detach()

    @staticmethod
    def _full_prompt_prefix_ids_and_mask(
        *,
        prompt_ids: torch.Tensor,
        prefix_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_attention_mask is None:
            prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        prefix_attention = torch.ones_like(
            prefix_ids,
            dtype=prompt_attention_mask.dtype,
            device=prefix_ids.device,
        )
        return (
            torch.cat([prompt_ids, prefix_ids], dim=1),
            torch.cat([prompt_attention_mask, prefix_attention], dim=1),
        )

    def _log_output(self, text: str) -> None:
        separator = "*" * 80
        self.raw_log_fn(separator)
        self.raw_log_fn(text or "")
        self.raw_log_fn(separator)

    def generate_continuation_from_latents(
        self,
        *,
        prompt_ids: torch.Tensor,
        prefix_state: torch.Tensor,
    ):
        prompt_state = self.embeddings(prompt_ids)
        combined_state = torch.cat([prompt_state, prefix_state], dim=1)
        attention_mask = torch.ones(
            combined_state.shape[:2],
            dtype=torch.long,
            device=combined_state.device,
        )
        return self.model.generate(
            inputs_embeds=combined_state,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            pad_token_id=self.pad_token_id,
        )

    def generate_continuation_from_hidden_state(
        self,
        *,
        prompt_ids: torch.Tensor,
        prefix_ids: torch.Tensor,
        prefix_state: torch.Tensor,
    ) -> torch.Tensor:
        full_input_ids, attention_mask = self._full_prompt_prefix_ids_and_mask(
            prompt_ids=prompt_ids,
            prefix_ids=prefix_ids,
        )
        prefix_start = prompt_ids.shape[1]
        prefix_end = prefix_start + prefix_ids.shape[1]
        with self._temporary_prefix_hidden_override(
            target_prefix_state=prefix_state,
            prefix_start=prefix_start,
            prefix_end=prefix_end,
        ):
            response = self.model.generate(
                input_ids=full_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                pad_token_id=self.pad_token_id,
            )
        return response.sequences[:, full_input_ids.shape[1]:]

    def original_generation(
        self,
        *,
        input_text: str,
    ) -> tuple[str, list[torch.Tensor], torch.Tensor]:
        prompt = self.tokenizer(
            [input_text],
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        prefix_ids = self.prefix_token_ids.to(self.device).unsqueeze(0)
        full_input_ids, attention_mask = self._full_prompt_prefix_ids_and_mask(
            prompt_ids=prompt.input_ids,
            prefix_ids=prefix_ids,
            prompt_attention_mask=prompt.attention_mask,
        )
        response = self.model.generate(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            pad_token_id=self.pad_token_id,
        )
        prompt_prefix_length = full_input_ids.shape[1]
        answer = self.tokenizer.decode(
            response.sequences[0][prompt_prefix_length:],
            skip_special_tokens=True,
        )
        prefix_state = self._extract_initial_prefix_state(
            prompt_ids=prompt.input_ids,
            prefix_ids=prefix_ids,
            prompt_attention_mask=prompt.attention_mask,
        )
        return answer, self._prefix_state_to_list(prefix_state), response.sequences

    def optimized_generation(
        self,
        *,
        question: str,
        input_text: str,
        original_output: str,
        original_latents_list: list[torch.Tensor],
        max_num_steps: int,
        lr: float,
        optimizer_name: str,
        grad_clip: float,
        reward_threshold: float,
    ):
        reward_history: list[float] = []
        output_history: list[str] = []
        reward_details_history: list[dict] = []
        latent_delta_history: list[dict[str, float]] = []

        original_generation_length = len(
            self.tokenizer.encode(original_output, add_special_tokens=False)
        )
        prompt = self.tokenizer(
            [input_text],
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        prompt_ids = prompt.input_ids.clone()
        prompt_length = prompt_ids.shape[1]
        prefix_ids = self.prefix_token_ids.to(self.device).unsqueeze(0)

        update_length = len(original_latents_list)
        expected_length = int(self.prefix_token_ids.numel())
        if update_length <= 0 or update_length != expected_length:
            raise ValueError(
                "Original prefix state length does not match insert_prefix_text token length."
            )

        initial_prefix_state = torch.stack(
            [state.clone().detach() for state in original_latents_list],
            dim=1,
        )
        use_muon_shape = optimizer_name.lower() == "muon"
        if use_muon_shape:
            latent_delta = torch.nn.Parameter(torch.zeros_like(initial_prefix_state[0]))
        else:
            latent_delta = torch.nn.Parameter(torch.zeros_like(initial_prefix_state))
        optimizer = self._build_optimizer(optimizer_name, [latent_delta], lr)

        def current_prefix_state() -> torch.Tensor:
            delta = latent_delta.unsqueeze(0) if use_muon_shape else latent_delta
            return initial_prefix_state + delta

        new_answer = original_output
        generated_ids = self.tokenizer.encode(
            new_answer,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        optimized_length = prompt_length + update_length + original_generation_length

        current_reward, reward_details = self.reward_model.get_reward(question, new_answer)
        reward_history.append(float(current_reward))
        output_history.append(new_answer)
        reward_details_history.append(reward_details)

        for step_index in range(1, max_num_steps + 1):
            if current_reward > reward_threshold:
                generated_length = len(
                    self.tokenizer.encode(new_answer, add_special_tokens=False)
                )
                optimized_length = prompt_length + update_length + generated_length
                self.log_fn(
                    f"Early stop at step {step_index}/{max_num_steps}: "
                    f"reward {current_reward} > threshold {reward_threshold}"
                )
                self._log_output(new_answer)
                break
            if generated_ids.size(1) == 0:
                self.log_fn("Generated continuation is empty. Stop optimization.")
                break

            optimizer.zero_grad()
            if self.optimize_layer_idx == 0:
                prompt_state = self.embeddings(prompt_ids).detach()
                prompt_prefix_state = torch.cat(
                    [prompt_state, current_prefix_state()],
                    dim=1,
                )
                continuation_state = self.embeddings(generated_ids[:, :-1]).detach()
                model_input_state = torch.cat(
                    [prompt_prefix_state, continuation_state],
                    dim=1,
                )
                output = self.model(inputs_embeds=model_input_state, use_cache=False)
                logits_start = prompt_prefix_state.size(1) - 1
            else:
                full_input_ids = torch.cat(
                    [prompt_ids, prefix_ids, generated_ids[:, :-1]],
                    dim=1,
                )
                attention_mask = torch.ones_like(full_input_ids, dtype=torch.long)
                prefix_start = prompt_length
                prefix_end = prefix_start + update_length
                with self._temporary_prefix_hidden_override(
                    target_prefix_state=current_prefix_state(),
                    prefix_start=prefix_start,
                    prefix_end=prefix_end,
                ):
                    output = self.model(
                        input_ids=full_input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                logits_start = prefix_end - 1

            logits = output.logits[
                :, logits_start:logits_start + generated_ids.size(1), :
            ]
            generated_log_probs = torch.log_softmax(logits, dim=-1).gather(
                -1,
                generated_ids.unsqueeze(-1),
            ).squeeze(-1)
            reward_tensor = torch.as_tensor(
                current_reward,
                device=generated_log_probs.device,
                dtype=generated_log_probs.dtype,
            )
            loss = -(reward_tensor * generated_log_probs.sum())
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([latent_delta], grad_clip)
            optimizer.step()

            if self.optimize_layer_idx == 0:
                response = self.generate_continuation_from_latents(
                    prompt_ids=prompt_ids,
                    prefix_state=current_prefix_state(),
                )
                generated_ids = response.sequences
            else:
                generated_ids = self.generate_continuation_from_hidden_state(
                    prompt_ids=prompt_ids,
                    prefix_ids=prefix_ids,
                    prefix_state=current_prefix_state(),
                )
            new_answer = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            )
            current_reward, reward_details = self.reward_model.get_reward(
                question,
                new_answer,
            )
            reward_history.append(float(current_reward))
            output_history.append(new_answer)
            reward_details_history.append(reward_details)

            with torch.no_grad():
                state = current_prefix_state().detach()
                delta = state - initial_prefix_state
                base_norm = torch.norm(initial_prefix_state).item()
                delta_norm = torch.norm(delta).item()
                flat_state = state.reshape(-1)
                flat_initial = initial_prefix_state.reshape(-1)
                denominator = flat_state.norm() * flat_initial.norm()
                token_delta = torch.norm(delta, dim=-1)
                latent_delta_history.append(
                    {
                        "delta_l2": delta_norm,
                        "delta_l2_rel": delta_norm / (base_norm + 1e-12),
                        "cosine_sim": (
                            torch.dot(flat_state, flat_initial) / (denominator + 1e-12)
                        ).item(),
                        "delta_linf": delta.abs().max().item(),
                        "per_token_delta_l2_mean": token_delta.mean().item(),
                        "per_token_delta_l2_max": token_delta.max().item(),
                        "changed_token_ratio": (
                            (token_delta > 1e-6).float().mean().item()
                        ),
                    }
                )

            self.log_fn(f"Optimization step: {step_index}/{max_num_steps}")
            self.log_fn(f"Current reward: {current_reward}")
            self._log_output(new_answer)
            generated_length = len(
                self.tokenizer.encode(new_answer, add_special_tokens=False)
            )
            optimized_length = prompt_length + update_length + generated_length

        return (
            new_answer,
            reward_history,
            output_history,
            reward_details_history,
            latent_delta_history,
            original_generation_length,
            optimized_length,
            update_length,
        )

