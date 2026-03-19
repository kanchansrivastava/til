# Fine-Tuning

**Fine-tuning** takes a pre-trained model's existing weights and continues training on a smaller, task-specific dataset, so you get task-adapted behavior without training from scratch. A support team feeds GPT-3.5 several thousand resolved tickets and ends up with a model that routes and drafts responses in their product's tone without prompting tricks.

**When to use**
Reach for fine-tuning when prompt engineering hits a ceiling: the model consistently misses domain vocabulary, ignores output format constraints, or needs behavior that can't be injected in a context window reliably.

**Don't confuse with**
**RAG (Retrieval-Augmented Generation)**: RAG adds knowledge at inference time via retrieved docs; fine-tuning bakes behavior into the weights at training time. RAG is for facts that change; fine-tuning is for style, format, or domain reasoning.

**Gotcha**
Fine-tuning on a narrow dataset causes **catastrophic forgetting**: the model loses general capability in proportion to how aggressively you train. A low learning rate and a small number of epochs aren't optional caution, they're the difference between specialization and a lobotomy.

## Comparison table

| Approach | Modifies weights | Needs labeled data | Runtime cost | Best for |
|---|---|---|---|---|
| Prompt engineering | No | No | Per-request tokens | Quick behavior nudges |
| Fine-tuning | Yes | Yes (hundreds to thousands of examples) | One-time training + hosting | Consistent format, tone, domain style |
| **PEFT / LoRA** | Yes (adapters only) | Yes | Lower training cost than full fine-tune | Large models where full fine-tune is expensive |
| Pretraining from scratch | Yes (all) | Yes (massive) | Enormous | Net-new architecture or domain |

## Code example

```python
# The trap: using a high LR and too many epochs
# This silently overfits — loss looks great, general perf craters

# BAD
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        num_train_epochs=10,       # too many
        learning_rate=5e-4,        # too high for fine-tuning
        ...
    ),
)

# BETTER: conservative LR, early stopping, eval on held-out general benchmark
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        num_train_epochs=3,
        learning_rate=2e-5,        # standard fine-tune range: 1e-5 to 5e-5
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        ...
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)
```

**Connects to:** **Parameters** are what fine-tuning actually modifies: the pre-trained weight values shift toward the task distribution, which is why a small dataset can steer a large model without a full retrain.