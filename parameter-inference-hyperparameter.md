# Parameter + Inference + Hyperparameter

**Parameter**: an internal numerical weight the model updates during training to reduce prediction error.
**Inference**: running a trained model on new data to produce an output.
**Hyperparameter**: a config value you set *before* training that governs how training runs, not what the model learns.

---

**When to use each**

- **Parameter**: you're asking "what did the model learn?" or debugging why outputs are wrong after training.
- **Inference**: you're asking "how do I get a prediction from this model right now?"
- **Hyperparameter**: you're asking "what should I tune to make training faster, more stable, or more accurate?"

---

**Don't confuse**

Hyperparameters vs. parameters: hyperparameters control the training process (learning rate, batch size), parameters *are* what training produces. You set hyperparameters; the optimizer sets parameters. The name sounds similar, the role is opposite.

---

**Gotcha**

Some values start as hyperparameters and get absorbed into the model (e.g., learned optimizers, neural architecture search). At that point they become parameters. The boundary is not always fixed, it depends on whether the value is frozen before training starts or updated during it.

---

## Comparison table

| Dimension | Parameter | Hyperparameter | Inference |
|---|---|---|---|
| **When it applies** | During + after training | Before training starts | After training is done |
| **Who sets it** | Optimizer (automatically) | You (manually or via search) | Runtime / calling code |
| **What it controls** | Model's learned behavior | How the training process runs | Prediction on new data |
| **Runtime vs. training** | Training (updated), Runtime (fixed) | Training only | Runtime only |
| **Stored in** | Model weights file | Config / experiment log | Not stored, it's a process |
| **Tuning method** | Gradient descent | Grid search, random search, Bayesian opt | Batching, hardware, quantization |
| **Example** | Weight matrix `W`, bias `b` | Learning rate `0.001`, batch size `32` | `model.predict(x)` |
| **What breaks if wrong** | Model accuracy (underfits/overfits) | Training stability, convergence speed | Latency, throughput, memory |