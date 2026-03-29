# Quantization

**Quantization** reduces the bit-width of model weights and activations (e.g., from FP32 down to INT8 or INT4) to shrink model size and cut inference latency, accepting a small accuracy trade-off to make that happen. LLaMA 3 8B at full precision needs ~32 GB of VRAM; quantized to 4-bit, it fits in 6 GB and runs on a consumer GPU.

---

**When to use**
Reach for quantization when your bottleneck is memory or inference speed at serving time, not when you still have training or fine-tuning left to do on the model.

---

**Don't confuse with**
**Pruning**: pruning removes weights entirely; quantization keeps all weights but represents them with less precision.

---

**Gotcha**
**Outlier channels** in activations (common in large transformers) can cause INT8 quantization to silently degrade accuracy far more than benchmarks suggest: a model that scores fine on perplexity can still produce noticeably worse outputs on reasoning tasks. LLM.int8() exists specifically because naive INT8 breaks on these outliers.

---

## Comparison table

| Method | Bit-width | Size reduction | Accuracy hit | Requires retraining? |
|---|---|---|---|---|
| **FP32** (baseline) | 32-bit | none | none | N/A |
| **FP16 / BF16** | 16-bit | ~2x | negligible | no |
| **GPTQ / INT8** | 8-bit | ~4x | small | no (post-training) |
| **GGUF Q4_K_M** | 4-bit | ~8x | moderate | no (post-training) |
| **QAT** (quantization-aware training) | 4–8-bit | ~4–8x | minimal | yes |

**Post-training quantization (PTQ)** is the default fast path. **Quantization-aware training (QAT)** recovers accuracy at the cost of a retraining run.

---

## Code example

```python
# Naive load: ~28 GB VRAM for a 7B model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# 4-bit PTQ via bitsandbytes: drops to ~4 GB, no retraining
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in BF16, store in 4-bit
    bnb_4bit_use_double_quant=True,          # nested quantization: extra ~0.4 bpw saving
    bnb_4bit_quant_type="nf4",              # NormalFloat4: better for normally distributed weights
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
)
# Trap: calling .half() or .float() after this silently dequantizes the model
```

The `nf4` vs `fp4` choice matters: `nf4` is designed for weights that follow a normal distribution (which pretrained weights typically do) and consistently outperforms `fp4` on the same bit budget.

---

**Connects to:** **Inference**: quantization is purely a serving-time optimization; it reduces the memory bandwidth and compute cost of each forward pass, which is exactly where inference latency lives.