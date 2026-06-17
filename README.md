# Fourier Number Embedding for LLMs

Large Language Models treat numbers as arbitrary tokens, losing critical information about magnitude, ordinality, and modular arithmetic. This project investigates **Fourier-based embedding injection methods** that encode rich numerical semantics into pre-trained LLMs while preserving their autoregressive decoding capabilities.

Built on insights from mechanistic interpretability research showing that transformers develop periodic activation patterns during arithmetic, this framework injects sinusoidal Fourier features into GPT-2 via modular encoder networks and parallel adapter layers -- without requiring full model retraining.

---

## Key Results

| Configuration | Whole-Number Accuracy | Digit-Wise Accuracy | Test Loss | MSE |
|---|---|---|---|---|
| Affine + Linear (Fourier decoder, 10K train) | **92.64%** | **98.68%** | 0.3317 | 2.76 x 10^10 |
| None + Linear (Fourier decoder, 10K train) | 92.13% | 98.64% | 0.3324 | 3.74 x 10^9 |
| Affine + MLP (Fourier decoder, 10K train) | 89.70% | 98.49% | 0.3350 | 3.24 x 10^9 |
| None + MLP (Fourier decoder, 10K train) | 90.27% | 98.46% | 0.3347 | 3.36 x 10^9 |

**Critical finding**: Fourier-encoded embeddings paired with standard L2R tokenizer decoding plateau at ~44% digit accuracy regardless of encoder/adapter sophistication, while the native Fourier decoder achieves 92.6% -- revealing a fundamental encoding-decoding mismatch that architectural capacity alone cannot resolve.

---

## Architecture

```
Input: "549213 + 957626 ="
        |
        v
  [NUM] Token Replacement
        |
        v
+-------+--------+
|                 |
v                 v
Standard        Fourier
Token           Embedding
Embeddings      (cos/sin features)
|                 |
|                 v
|           Intermediate Network
|           (Linear / MLP)
|                 |
+-----> (+) <-----+
         |
         v
   GPT-2 Backbone
   (with Parallel Adapters
    in Attn + FFN sublayers)
         |
         v
   Decoder Head
   (Fourier / L2R / R2L)
         |
         v
   Output: "1506839"
```

### Components

**Fourier Number Encoder (FoNE)**: Decomposes numerical inputs into sinusoidal features using configurable period bases (default: base-10), producing cos/sin embeddings that capture digit-level modular structure. Maps to a 768-dimensional embedding space matching GPT-2's hidden size.

**Intermediate Encoder Network**: Projects Fourier embeddings before fusion with token embeddings.
- *Linear*: Single linear transformation (~590K params)
- *MLP*: 2-layer network with ReLU and residual connection (~7.3M params)
- *Identity*: Pass-through baseline

**Parallel Adapters**: Lightweight modules injected alongside both multi-head attention and feed-forward sublayers in each transformer block.
- *Affine*: Full linear transform with bias (~14.2M params total)
- *Low-Rank (LoRA-style)*: Factored down/up projections with configurable rank
- All adapters are zero-initialized for stable training start

**Decoding Strategies**:
- *Fourier*: Digit-parallel prediction via inverse Fourier mapping (digit-wise cross-entropy loss)
- *L2R*: Standard left-to-right autoregressive token generation
- *R2L*: Right-to-left generation (least significant digit first)

## Dataset

Synthetic 6-digit addition dataset with **1.18M total samples** following the paradigm from Saxton et al. (2019).

| Split | Samples | Purpose |
|---|---|---|
| Train | 1,004,000 | Model training (subsets of 10K-100K used in experiments) |
| Validation | 140,000 | Hyperparameter tuning |
| Test | 40,000 | Final evaluation |

**Leakage prevention**: Strict number-level isolation ensures no number appearing in the test set occurs in any train or validation question. Duplicate-aware filtering treats `num1+num2` and `num2+num1` as identical.

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Conda (for environment management)

### Installation

```bash
# Clone the repository
git clone https://github.com/Deep99739/fourier-encoder.git
cd fourier-encoder

# Create environment from spec
conda env create -f FoNE_old/environment.yml

# Or install key dependencies manually
pip install torch transformers datasets wandb
```

### Environment Variables

```bash
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_key"
```

---

## Usage

### Quick Start

```bash
cd FoNE_new

# Fourier decoder with linear encoder (best config)
python main.py \
    --method fne \
    --model gpt2 \
    --intermediate_network linear \
    --adapter_type affine \
    --decoder_type fne \
    --dataset 6_digits_add \
    --num_train_samples 10000 \
    --int_digit_len 10 \
    --frac_digit_len 0 \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --train_from_scratch
```

### Configuration Options

| Argument | Options | Description |
|---|---|---|
| `--method` | `fne`, `xval`, `vanilla`, `rene`, `regular` | Embedding method |
| `--intermediate_network` | `linear`, `mlp`, `identity` | Encoder network type |
| `--adapter_type` | `None`, `linear`, `affine`, `low_rank` | Parallel adapter variant |
| `--decoder_type` | `fne`, `greedy` | Decoding strategy |
| `--model` | `gpt2`, `llama`, `bert` | Backbone LLM |
| `--freeze_model` | flag | Freeze backbone weights |
| `--train_from_scratch` | flag | Random init vs. pretrained |
| `--model_size_level` | `1-10` | Custom model size (scratch only) |
| `--num_train_samples` | int | Training subset size |
| `--period_base_list` | float list | Fourier period bases |

### Reproducing Experiments

**Experiment 1** -- Fourier decoder, 10K training samples:
```bash
python main.py --method fne --decoder_type fne --num_train_samples 10000 \
    --intermediate_network linear --adapter_type affine --lr 1e-3 --epochs 80
```

**Experiment 2** -- L2R greedy decoder, 100K training samples:
```bash
python main.py --method fne --decoder_type greedy --num_train_samples 100000 \
    --intermediate_network linear --adapter_type affine --lr 1e-4 --epochs 80
```

**Experiment 3** -- R2L decoding (modify eval.py to uncomment R2L lines):
```bash
python main.py --method fne --decoder_type greedy --num_train_samples 100000 \
    --intermediate_network mlp --adapter_type affine --lr 1e-4 --epochs 80
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Whole-Number Accuracy** | Exact match at the integer level |
| **Digit-Wise Accuracy** | Fraction of correctly predicted individual digits |
| **MSE** | Mean squared error between predicted and true integers |
| **R-squared** | Proportion of variance explained |

All metrics are logged to Weights & Biases for real-time tracking across runs.

---

## Parameter Efficiency

| Configuration | Trainable Params | Notes |
|---|---|---|
| Frozen backbone + Linear encoder + No adapters | ~1M | Minimal config |
| Frozen backbone + MLP encoder + Affine adapters | ~21.5M | Recommended |
| Full fine-tuning (GPT-2 + MLP + adapters) | ~160M | Maximum capacity |

The parallel adapter design keeps the 125M GPT-2 backbone frozen while injecting numerical reasoning capability through only ~14.2M additional parameters -- a ~90% reduction in training compute versus full fine-tuning.

---

## Ablation Summary

Systematic experiments across **12+ configurations** varying four axes:

| Axis | Options Tested |
|---|---|
| Intermediate Encoder | Linear, MLP |
| Parallel Adapter | None, Affine |
| Decoder Strategy | Fourier, L2R, R2L |
| Dataset Scale | 10K, 100K |

**Key findings**:
1. Linear encoders match or outperform MLPs under Fourier decoding -- extra nonlinearity is unnecessary when the decoder operates in Fourier space
2. Affine adapters provide consistent but marginal gains (~0.5% accuracy)
3. Richer encoders/adapters accelerate early convergence by ~3pp but do not shift the final plateau under greedy decoding
4. R2L decoding halves digit accuracy (44% to 24%) yet improves regression metrics (R-squared from -7.47 to -3.14), suggesting the model captures magnitude but fails at positional alignment

---