# GradCuit

[![Project Page](https://img.shields.io/badge/Project-Page-BA8E9F?style=for-the-badge)](https://yuzhaoxin946.github.io/GradCuit/)
[![Paper](https://img.shields.io/badge/arXiv-2608.02585-B31B1B?style=for-the-badge)](https://arxiv.org/abs/2608.02585)

**Circuit-like Gradient Flow for Test-Time Instance-Level Latent Reasoning**

GradCuit (*gradient through circuit*) is a test-time latent reasoning method for frozen
autoregressive transformers. It places a small set of continuous latent tokens directly in the
transformer's input embedding sequence, alongside the embedded prompt. Generated tokens attend to
these latents through standard self-attention, while token-probability gradients flow backward
through the same attention connectivity to assign credit to each latent.

For each problem, GradCuit evaluates the generated answer with a single verifier and applies a
policy-gradient-style update to the latent tokens. The language model parameters remain frozen;
only the instance-specific latents are updated before regenerating the answer.

This implementation supports GSM8K, MATH-500, and GPQA-Diamond.
Answer extraction and final-answer judging use the same evaluation pipeline as the
original single-verifier experiments.

## Installation

Python 3.10 or newer is required. Install a CUDA-compatible PyTorch build for your system, then run:

```bash
pip install -r requirements.txt
```

MATH-500 evaluation uses the original `latex2sympy2==1.9.0` parser with
`antlr4-python3-runtime==4.11.1`, matching the original GradCuit environment.

The verifier uses a vLLM OpenAI-compatible server. The solver and verifier normally run in separate
processes and may use different GPUs.

## Quick start

Edit the static model and GPU settings in the two example scripts. Start the verifier first:

```bash
bash scripts/start_verifier.sh
```

Then run the GradCuit example:

```bash
bash scripts/run_example.sh
```

The example uses the public model identifier `meta-llama/Llama-3.2-3B-Instruct`, GSM8K, the relative
output directory `./output`, a learning rate of `0.001`, and `2048` maximum new tokens. It optimizes
the inserted prefix at decoder block input `14`, the middle of the model's 28 decoder blocks.

GPQA Diamond is gated. Accept its dataset terms and authenticate with Hugging Face before using
`gpqa_diamond`.

## Main arguments

`src/main.py` requires all experiment-defining arguments explicitly. Run:

```bash
python src/main.py --help
```

`--solver_prompt_idx` accepts `0` for boxed answers and `1` for JSON answers.
`--optimize_layer_idx 0` optimizes prefix embeddings; a positive value optimizes the prefix hidden
state at the input to that decoder block. `--optimizer` accepts `adam`, `sgd`, or `muon`.
`--max_num_steps` must be positive; the paper uses `10`. Set `--grad_clip 0` to disable clipping and
`--num_data -1` to process all remaining examples.

Use `--resume` with the same experiment arguments to continue the newest matching output directory.
The API key, when needed, is read from `VLLM_API_KEY`.

## Outputs

Each run writes:

- `run_args.json`: sanitized experiment settings.
- `test/<index>_data.json`: prompt, original output, optimization history, and final correctness.
- `logistics.pt`: resume state and aggregate counters.

Generated outputs are ignored by Git.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q test
python -m compileall src
```
