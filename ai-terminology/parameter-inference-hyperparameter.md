# Parameter + Inference + Hyperparameter

**Parameter**: an internal numerical weight the model learns during training to minimize loss. A transformer's attention weight matrix, e.g., the 175B float values in GPT-3, is entirely parameters.

**Inference**: running a trained, frozen model on new input to get an output. Sending a user's prompt to a deployed model endpoint and receiving a completion is inference.

**Hyperparameter**: a configuration value *you* set before training starts that shapes how training proceeds. Setting `learning_rate=0.001` in your optimizer config is a hyperparameter choice, not something the model discovers.

| Concept | Real-world instance |
|---|---|
| Parameter | BERT's 110M weights stored in a `.bin` checkpoint |
| Inference | Calling `model.predict(image)` on a production server |
| Hyperparameter | `batch_size=32` in your training script |

---

**When to use each**

- **Parameter**: when you're asking "what did the model actually learn?" or debugging weight initialization and gradient flow.
- **Inference**: when you're asking "how do I serve this model?" or profiling latency and throughput in production.
- **Hyperparameter**: when you're asking "why is my training unstable or slow to converge?" or running a sweep to improve final model quality.

---

**Don't confuse**

- **Hyperparameter vs. Parameter**: hyperparameters control the training process; parameters are what training produces. Learning rate is never stored in the model file. Weights are never in your config YAML.
- **Inference vs. Training**: inference is read-only on weights; training mutates them. If your GPU memory spikes and weights are changing, you are not doing inference.

---

**Gotcha**

Some values start as hyperparameters and get baked into parameters. Dropout rate is a hyperparameter at training time, but at inference you disable it entirely. Forgetting to call `model.eval()` in PyTorch leaves dropout active during inference, producing non-deterministic outputs that look like model bugs but are actually a mode error.

---

## Comparison table

| Dimension | Parameter | Hyperparameter | Inference |
|---|---|---|---|
| **When set** | Learned during training | Before training starts | After training, at serve time |
| **Who sets it** | Optimizer (automatically) | Developer / tuning job | Runtime / application code |
| **What it controls** | Model's internal representation | Training dynamics and final model quality | Output generation on new data |
| **Stored where** | Model checkpoint (`.pt`, `.bin`, etc.) | Config file / CLI args / sweep config | Not stored; ephemeral per request |
| **Mutable at runtime** | No (frozen after training) | No (fixed before training run) | N/A, it's a process not a value |
| **Affects training** | Yes, updated each step | Yes, governs how updates happen | No |
| **Affects production** | Yes, determines prediction quality | Indirectly (via the model produced) | Yes, directly |
| **Example** | Attention weight `W_q` | `learning_rate`, `num_layers`, `dropout` | A single forward pass through the model |
| **Failure mode** | Overfitting, underfitting | Unstable loss, slow convergence | Latency spikes, OOM on large inputs |

---

**Connects to:** **Inference** reads the **parameters** a training run produced; the quality of those parameters is a direct function of the **hyperparameters** chosen during that run. Tune hyperparameters poorly, and every inference call serves a worse model regardless of infrastructure quality.