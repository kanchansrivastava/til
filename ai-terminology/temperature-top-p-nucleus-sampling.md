# Temperature + Top-p (Nucleus) Sampling

**Temperature** is a **sampling hyperparameter** that scales the probability distribution over the token vocabulary before a token is picked: high value = flatter distribution (more surprise), low value = sharper distribution (more predictable).
- Real-world instance: set `temperature=0.1` when generating SQL from a schema — you want the one right answer, not creative variations.

**Top-p (Nucleus) Sampling** is a decoding strategy that builds the smallest set of candidate tokens whose cumulative probability reaches threshold **p**, then samples only from that set.
- Real-world instance: set `top_p=0.9` for a story-continuation feature — the model skips the long tail of weird/irrelevant tokens but still has room to surprise.

| Property | Temperature | Top-p |
|---|---|---|
| Controls | Width of the whole distribution | Which tokens are even eligible |
| Range | 0 → ∞ (practical: 0–2) | 0.0 → 1.0 |
| Value meaning | Higher = more random | Lower = tighter candidate pool |

---

**When to use each**

Use **temperature** when you want a single dial to globally loosen or tighten the model's confidence across every token it generates. Use **top-p** when you want to hard-exclude low-probability tokens regardless of how confident the model is — it's a vocabulary filter, not a confidence rescaler.

---

**Don't confuse**

Top-p vs. **Top-k**: top-k cuts the candidate list to a fixed count (e.g., always 40 tokens); top-p cuts to a variable count based on cumulative probability mass. On a peaked distribution, top-p picks fewer candidates; on a flat one, it picks more. Top-k is oblivious to that shape.

---

**Gotcha**

Running temperature and top-p together is multiplicative in effect, not independent. A high temperature flattens the distribution *before* top-p applies, so `top_p=0.9` at `temperature=1.5` includes far more tokens than `top_p=0.9` at `temperature=0.5`. Most API docs treat them as independent knobs, but they interact. The common mistake: tuning top-p in isolation without realising the temperature setting is already doing most of the work.

---

## Comparison table

| Dimension | Temperature | Top-p (Nucleus) |
|---|---|---|
| **What it controls** | Sharpness of the full probability distribution | Size of the eligible token pool |
| **Mechanism** | Divides logits by T before softmax | Cumulative probability cutoff post-softmax |
| **Runtime vs. training** | Runtime (inference only) | Runtime (inference only) |
| **Who sets it** | Developer / prompt engineer | Developer / prompt engineer |
| **When to reach for it** | Adjusting overall creativity or determinism | Pruning implausible tokens while preserving diversity |
| **Low value effect** | Near-deterministic, picks the top token almost every time | Only top tokens considered, very focused |
| **High value effect** | Chaotic, low-probability tokens become likely | Nearly the full vocabulary is eligible |
| **Fails badly when** | Set high for factual tasks (hallucinations spike) | Set too low on niche domains (correct rare tokens excluded) |
| **Interacts with** | Top-p, top-k | Temperature (temperature reshapes distribution first) |
| **Type** | Hyperparameter | Decoding strategy / hyperparameter |

---

**Connects to:**
- **Inference**: both parameters are inference-time controls only — they change nothing about model weights, they only govern how the model samples during a forward pass.
- **Hyperparameter**: temperature is a textbook hyperparameter (you tune it, the model doesn't learn it); top-p sits in the same category even though it operates post-softmax rather than inside the loss function.
- **Token**: the entire mechanism of both parameters is about which **token** gets selected next — temperature rescales token probabilities, top-p restricts the token candidate set.
- **Chain-of-Thought (CoT) Prompting**: CoT works better at low temperature (0–0.3) because reasoning chains need consistency across steps; a high temperature mid-chain can introduce a wrong turn that compounds through the rest of the reasoning trace.