# Token + Embedding

A **token** is the atomic unit of text a model reads and predicts: not always a word, often a subword chunk or single character, depending on the tokenizer.
- Real example: `"unbelievable"` splits into `["un", "believ", "able"]` under GPT-4's BPE tokenizer — three tokens, one word.

An **embedding** is a fixed-length float vector that maps a token (or sequence) into a high-dimensional space where semantic proximity equals geometric proximity.
- Real example: The vectors for `"king"` and `"queen"` sit closer together than `"king"` and `"carburetor"` — subtraction `king - man + woman ≈ queen` actually works.

| | Token | Embedding |
|---|---|---|
| Form | Integer ID (e.g. `14523`) | Float vector (e.g. `[0.21, -0.87, ...]`) |
| Size | 1 integer | 768–4096+ floats |
| Human-readable | Yes (maps back to text) | No |

---

**When to use each**

- **Token**: you care about *how much* text costs or fits — count tokens when tracking context limits, billing, or chunking input.
- **Embedding**: you care about *what* text means — generate embeddings when comparing, searching, clustering, or classifying content semantically.

---

**Don't confuse**

Token IDs and embeddings look like they both "represent" text, but a token ID is just an index into a lookup table; the embedding is what that index resolves to inside the model. One is a key, the other is the value.

Also: an embedding for a full sentence (from a sentence-transformer) is *not* the same as averaging token embeddings from a GPT-style model — the pooling strategy matters and the two are not interchangeable.

---

**Gotcha**

Token count and embedding dimension are independent axes of cost. A 512-token input to GPT-4 uses 512 tokens of context, but each of those tokens has one embedding vector of ~12,288 floats internally. Confusing "context length" (token budget) with "embedding size" (model width) leads to wrong capacity estimates when you're planning memory or latency budgets.

---

## Comparison table

| Dimension | Token | Embedding |
|---|---|---|
| **What it is** | Discrete text unit with an integer ID | Continuous float vector encoding meaning |
| **When you use it** | Counting input length, splitting text, billing | Semantic search, similarity, classification |
| **What it controls** | Context window consumption | Geometric distance between concepts |
| **Who sets it** | Tokenizer vocab (fixed at training) | Model weights (learned at training) |
| **Runtime vs. training** | Used at both; tokenizer is frozen | Learned at training, read-only at inference |
| **Mutable by you** | No (you pick the tokenizer, not the vocab) | Fine-tuning can shift embeddings |
| **Output you see** | Integer IDs or decoded strings | Float arrays (via API or hidden layer) |
| **Typical size** | 1 int per unit | 256–4096+ floats per unit |
| **Key failure mode** | Misjudging token count vs. character count | Comparing embeddings across different models |

---

**Connects to:**

- **Inference**: at inference time, the model converts your input tokens into embeddings on the first pass through the embedding layer, so token count directly sets the memory footprint of that inference call.
- **Parameter**: the embedding matrix is itself a block of parameters (vocab size × embedding dim), often one of the largest single weight tensors in the model.