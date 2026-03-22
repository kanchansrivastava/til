# Zero-Shot Learning + Few-Shot Learning + Chain-of-Thought (CoT) Prompting

**Zero-Shot Learning**: The model handles a task using only your description, no examples provided.
- Real-world instance: You send `"Classify this support ticket as Bug, Feature Request, or Question: 'App crashes on login'"` with no prior examples in the prompt, and the model classifies it correctly.

**Few-Shot Learning**: You pack 2–5 worked examples directly into the prompt to anchor the model's output format and reasoning, no weight updates.
- Real-world instance: Before asking the model to extract invoice line items, you include three sample invoice snippets with their correctly parsed JSON outputs.

**Chain-of-Thought (CoT) Prompting**: You instruct the model to write out its reasoning steps before committing to an answer, which reduces errors on multi-step problems.
- Real-world instance: Adding `"Think step by step"` before asking `"A store sells 3 widgets at $4.50 each, applies a 10% discount, then adds 8% tax. What's the total?"` gets the correct answer; without it, the model often skips steps and miscalculates.

---

## When to use each

**Zero-Shot**: Use it when the task is well-defined in natural language and you don't have labeled examples handy, or when you're prototyping fast.

**Few-Shot**: Use it when zero-shot output is inconsistent, wrongly formatted, or missing domain-specific patterns you can demonstrate in 2–5 examples.

**CoT**: Use it when the task requires multiple reasoning steps (math, logic, multi-condition decisions) and accuracy matters more than response speed.

---

## Don't confuse

| Mix-up | One-line difference |
|---|---|
| Few-Shot vs. **Fine-Tuning** | Few-Shot puts examples in the prompt at runtime; Fine-Tuning bakes examples into the weights at training time |
| Zero-Shot vs. a badly written prompt | Zero-Shot failing usually means the task description is ambiguous, not that the technique is wrong |
| CoT vs. Few-Shot | Few-Shot controls *what* the model outputs; CoT controls *how* it reasons to get there. They can be combined: few-shot CoT includes examples that themselves show step-by-step reasoning |

---

## Gotcha

CoT increases token usage on both input and output, which raises latency and cost. On simple tasks it also *introduces* errors: the model can reason itself into a wrong answer it would have gotten right directly. Disable CoT for classification or retrieval tasks where reasoning steps add noise, not signal.

---

## Comparison table

| Dimension | Zero-Shot | Few-Shot | Chain-of-Thought (CoT) |
|---|---|---|---|
| **When to use** | Task is self-describing; no examples available | Output format is inconsistent or domain-specific | Multi-step reasoning; errors on complex logic |
| **What it controls** | Whether to provide examples at all | The model's output pattern and style | The model's internal reasoning process |
| **Who sets it** | Developer, via task description in prompt | Developer, via curated examples in prompt | Developer, via instruction (`"think step by step"`) or example reasoning chains |
| **Requires labeled data?** | No | Yes, a few (2–5 usually enough) | No (zero-shot CoT) or a few reasoning examples (few-shot CoT) |
| **Weight updates?** | No | No | No |
| **Runtime vs. training** | Runtime | Runtime | Runtime |
| **Token cost** | Lowest | Medium (scales with example count) | Highest (reasoning trace adds output tokens) |
| **Failure mode** | Wrong output shape, missed nuance | Model over-fits to example format, ignores edge cases | Verbose reasoning leading to a wrong conclusion; slow on simple tasks |
| **Composable?** | Baseline for the other two | Combines with CoT (few-shot CoT) | Combines with zero-shot or few-shot |

---

**Connects to:** **Token**: every example you add in Few-Shot and every reasoning step CoT generates costs tokens, directly affecting context window limits and inference cost per call.