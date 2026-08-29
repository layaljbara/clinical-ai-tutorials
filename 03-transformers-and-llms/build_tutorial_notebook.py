#!/usr/bin/env python3
"""Build the learning notebook (run once)."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(s: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s: str) -> None:
    cells.append(nbf.v4.new_code_cell(s.strip()))


md(
    """
# Tutorial: Base Models, Reasoning, Mixture-of-Experts & Model Training

**Goal:** an *easy* path that still covers *every important detail* behind `gpt-oss-20b`
and how we train it in `clinical-gpt-oss`.

**How to use**
1. Run cells top to bottom.
2. Read the markdown, then run the tiny PyTorch demos.
3. Lines marked **In our project** map the toy idea onto Stage 1 DAPT / Stage 2 A&P.

**You will learn**
1. What a base language model does (next-token prediction)
2. What “reasoning” means for GPT-OSS (analysis vs final / Harmony)
3. How Mixture-of-Experts (MoE) works
4. Training details: loss, gradients, batches, epochs, LR schedules
5. Pretraining vs DAPT vs SFT vs LoRA — and our exact pipeline

**Requirements:** `torch`, `numpy`
"""
)

md("---\n## 0. Setup")

code(
    """
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
device = "cpu"
print("torch:", torch.__version__)
print("device:", device)
"""
)

md(
    """
---
# Part A — What is a base language model?

## A1. The only job: predict the next token

A causal LM (like GPT-OSS) answers one question repeatedly:

> Given tokens so far, what is the probability distribution over the **next** token?

Training maximizes the probability of the real next token (minimize cross-entropy / NLL).

**In a typical pipeline:** Stage 1 DAPT is this objective on clinical notes.
Stage 2 uses the same objective on chat text so the model produces the Assessment & Plan.
"""
)

code(
    """
# Tiny vocabulary LM: learn that after "tac" comes "rolimus"
vocab = {"<pad>": 0, "patient": 1, "on": 2, "tac": 3, "rolimus": 4, "sirolimus": 5, ".": 6}
id2tok = {i: t for t, i in vocab.items()}
V = len(vocab)

sequences = [
    [vocab["patient"], vocab["on"], vocab["tac"], vocab["rolimus"], vocab["."]],
    [vocab["patient"], vocab["on"], vocab["tac"], vocab["rolimus"], vocab["."]],
    [vocab["patient"], vocab["on"], vocab["sirolimus"], vocab["."]],
]

def make_xy(seqs):
    xs, ys = [], []
    for s in seqs:
        for i in range(len(s) - 1):
            xs.append(s[: i + 1])
            ys.append(s[i + 1])
    max_len = max(len(x) for x in xs)
    X = torch.zeros(len(xs), max_len, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, : len(x)] = torch.tensor(x)
    y = torch.tensor(ys)
    return X, y

X, y = make_xy(sequences)
print("Training pairs:", len(y))
print(
    "Example prefix",
    [id2tok[i] for i in X[2].tolist() if i != 0],
    "-> target",
    id2tok[y[2].item()],
)
"""
)

code(
    """
class TinyLM(nn.Module):
    # Embed tokens -> mean pool -> linear to vocab logits
    def __init__(self, vocab_size, d=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.out = nn.Linear(d, vocab_size)

    def forward(self, x):
        h = self.emb(x)
        mask = (x != 0).float().unsqueeze(-1)
        h = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.out(h)

model = TinyLM(V).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=0.05)

for step in range(200):
    logits = model(X)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"step {step:3d}  loss={loss.item():.3f}")

probe = torch.tensor([[vocab["patient"], vocab["on"], vocab["tac"]]])
with torch.no_grad():
    probs = F.softmax(model(probe), dim=-1)[0]
print("\\nP(next | 'patient on tac'):")
for tok, idx in vocab.items():
    if tok != "<pad>":
        print(f"  {tok:10s}  {probs[idx].item():.3f}")
"""
)

md(
    """
### What you just saw
- **Logits** = raw scores per vocabulary token
- **Softmax** = probabilities
- **Cross-entropy** = punishes low probability on the true next token
- After training, `tac` is followed by `rolimus` more often

Scale this to billions of parameters → GPT-OSS.
"""
)

md(
    """
## A2. Tokens, embeddings, context length

| Concept | Meaning |
|---|---|
| **Token** | Subword piece (not always a full word) |
| **Vocabulary** | Fixed set of tokens the model knows |
| **Embedding** | Token id → vector |
| **Context length** | Max tokens in one forward pass |
| **gpt-oss-20b** | Native context ~**131,072** tokens |
| **Our training** | `max_seq_length=4096` (GPU choice, not model max) |

**In a typical pipeline:** long notes may be truncated to fit 4096 during Stage 1/2 training.
"""
)

md(
    """
## A3. Transformers (minimal intuition)

A GPT-style block repeatedly does:
1. **Self-attention** — each token mixes information from previous tokens (causal mask)
2. **Feed-forward / MLP** — nonlinear transform per token
3. **Residuals + LayerNorm** — stabilize depth

GPT-OSS also uses Grouped Query Attention, sliding-window attention on some layers,
and RoPE/YaRN so long contexts work.

**Takeaway:** attention is how earlier clinical text conditions the next token / A&P.
"""
)

code(
    """
# Tiny causal self-attention — shape intuition
B, T, d = 1, 4, 8
Q = torch.randn(B, T, d)
K = torch.randn(B, T, d)
V = torch.randn(B, T, d)

scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d)
causal = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(causal == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)
out = weights @ V

print("attention weights (row=query pos, col=key pos):\\n", weights[0].detach().numpy().round(3))
print("each row sums to 1:", weights[0].sum(-1))
"""
)

md(
    """
---
# Part B — What is “reasoning”?

## B1. Human vs model meaning

**Human reasoning:** deliberate multi-step thinking.

**Model “reasoning” (GPT-OSS style):** the model is trained to emit an intermediate scratchpad
(**analysis** / chain-of-thought) before a **final** answer. Extra “thinking” tokens can help hard tasks,
but cost latency and can confuse you if you parse the wrong channel.

## B2. Harmony channels (critical for our project)

```text
<|channel|>analysis<|message|>Need graft function and IS sections...<|end|>
<|channel|>final<|message|>- Graft function: Excellent...
```

| Channel | Role |
|---|---|
| **analysis** | Internal working notes / “reasoning” |
| **final** | What the clinician should see |

**In a typical pipeline:** `training/harmony_output.py` extracts **final**.
Raw analysis like `Need assessment plan sections` is *not* the A&P.

Eval default: `reasoning_effort="low"` → less analysis budget, more concise finals.
"""
)

code(
    """
RAW = '''
<|channel|>analysis<|message|>Need assessment plan sections. Consider tacrolimus dose.<|end|>
<|channel|>final<|message|>- Graft function: Excellent, normal enzymes.
- Immunosuppression: therapeutic tacrolimus level.<|return|>
'''

final_re = re.compile(
    r"<\\|channel\\|>final<\\|message\\|>(.*?)(?:<\\|return\\|>|<\\|end\\|>|$)",
    re.DOTALL,
)
analysis_re = re.compile(
    r"<\\|channel\\|>analysis<\\|message\\|>(.*?)(?:<\\|end\\|>|<\\|channel\\|>)",
    re.DOTALL,
)

final = final_re.search(RAW).group(1).strip()
analysis = analysis_re.search(RAW).group(1).strip()
print("ANALYSIS (reasoning scratchpad):\\n", analysis)
print("\\nFINAL (keep for A&P eval):\\n", final)
"""
)

md(
    """
### Reasoning effort (low / medium / high)
A **budget dial**: higher → longer analysis → sometimes better hard answers → slower.
For clinical drafting we usually prefer **low**.
"""
)

md(
    """
---
# Part C — Mixture of Experts (MoE)

## C1. Dense MLP vs MoE

**Dense block:** every token uses the **same** big feed-forward network.

**MoE block:** many expert MLPs. A **router** scores experts per token and sends the token
to the **top-k** experts only (GPT-OSS: **top-4**).

| | Dense | MoE |
|---|---|---|
| Capacity | One FFN | Many experts |
| Compute per token | Full FFN | Only k experts |
| Why | Simple | More total params without always paying full FLOPs |

**gpt-oss-20b (public card):** ~20.9B **total** params, ~**3.6B active**/token, **32 experts**, top-4 routing.

### Mental picture
Token `"tacrolimus"` → router picks experts e.g. 3,7,11,19 → those four compute → weighted mix.
"""
)

code(
    """
class Expert(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def forward(self, x):
        return self.net(x)


class TinyMoE(nn.Module):
    # Educational MoE: E experts, top-k routing per token
    def __init__(self, d=16, hidden=32, n_experts=4, k=2):
        super().__init__()
        self.k = k
        self.router = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([Expert(d, hidden) for _ in range(n_experts)])

    def forward(self, x):
        scores = self.router(x)
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)
        out = torch.zeros_like(x)
        for b in range(x.size(0)):
            for j in range(self.k):
                e = topk_idx[b, j].item()
                w = topk_weights[b, j]
                out[b] = out[b] + w * self.experts[e](x[b])
        return out, topk_idx, topk_weights

moe = TinyMoE()
x = torch.randn(3, 16)
y, idx, w = moe(x)
print("top-2 expert indices per token:\\n", idx)
print("mixing weights:\\n", w.detach())
print("output shape:", y.shape)
"""
)

md(
    """
### MoE practical notes
1. **Load balancing** — avoid all tokens routing to one expert.
2. **Specialization** — experts *may* specialize; not guaranteed.
3. **Our LoRA** can target MoE expert projections (`7/15/23.mlp.experts.*` in `train_sft_trl.py`)
   plus `target_modules="all-linear"`.

You still load one HF model; routing is inside the architecture.
"""
)

md(
    """
---
# Part D — Model training (every core detail)

## D1. The universal training loop

```text
for each epoch:
  for each batch:
    forward -> logits
    loss = cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

| Term | Meaning |
|---|---|
| **Epoch** | One full pass over the training set |
| **Batch** | Examples per update |
| **Grad accumulation** | Sum grads over several micro-batches before stepping |
| **Learning rate** | Step size |
| **Warmup** | Ramp lr at start (`warmup_ratio=0.03` for us) |
| **Cosine schedule** | Decay lr over training |
| **Checkpoint** | Saved weights (our LoRA adapter folder) |
"""
)

code(
    """
class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x):
        return self.net(x)

Xdata = torch.randn(64, 4)
ydata = (Xdata.sum(1) > 0).long()
loader = DataLoader(TensorDataset(Xdata, ydata), batch_size=4, shuffle=True)

model = TinyClassifier()
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
grad_accum = 2
print(f"micro-batch=4, accum={grad_accum}, effective batch≈{4 * grad_accum}")

for epoch in range(5):
    opt.zero_grad()
    for i, (xb, yb) in enumerate(loader):
        loss = F.cross_entropy(model(xb), yb) / grad_accum
        loss.backward()
        if (i + 1) % grad_accum == 0:
            opt.step()
            opt.zero_grad()
    with torch.no_grad():
        acc = (model(Xdata).argmax(1) == ydata).float().mean().item()
    print(f"epoch {epoch+1}: acc={acc:.2f}")
"""
)

md(
    """
## D2. LM loss and perplexity

For true next token \(y\):

\\\\[\\\\mathcal{L} = -\\\\log p_\\\\theta(y \\\\mid context)\\\\]

**Perplexity** = \(e^{\\\\text{mean NLL}}\). Lower ⇒ better domain fit.
Our Stage 1 story: PPL ~**80 → ~5**.
"""
)

code(
    """
print("If mean NLL = log(80), PPL =", math.exp(math.log(80)))
print("If mean NLL = log(5),  PPL =", math.exp(math.log(5)))
"""
)

md(
    """
## D3. Backprop intuition
1. Loss measures “how wrong.”
2. `backward()` → gradient for each trainable parameter.
3. AdamW nudges parameters to reduce loss.

**Frozen base + LoRA:** gradients only update LoRA matrices (~15M params), not all 20B.

## D4. Stability knobs we actually use

| Knob | Our value | Why |
|---|---|---|
| bf16 | on | Memory/speed |
| gradient_checkpointing | on | Memory |
| use_cache=False | on | Needed with checkpointing |
| max_seq_length | 4096 | Cap activation memory |
| batch size | 1 | Fits dequantized 20B |
| grad_accum | 16 | Effective larger batch |
| attn | eager | Compatible; heavier than flash-attn |
"""
)

md(
    """
---
# Part E — Pretraining vs DAPT vs SFT vs LoRA

| Stage | Objective | Data | Skill |
|---|---|---|---|
| **Pretraining** (already done by OpenAI) | Next-token, general | Huge public mix | General LM |
| **DAPT Stage 1** | Next-token, domain | Notes **without** A&P | Transplant note language |
| **SFT Stage 2** | Chat target tokens | Note → gold A&P | Draft A&P |
| **RLHF** (not done yet) | Prefer better answers | Preferences | Alignment |

## LoRA formula

\\\\[W' = W + (\\\\alpha / r) B A\\\\]

- \(W\) frozen base, \(A,B\) trainable, \(r=8\), \(\\alpha=16\)
- Starts near no-op if \(B=0\); learns a small task/domain delta
- We save an **adapter folder**, not a full 20B copy
"""
)

code(
    """
torch.manual_seed(0)
d_in, d_out, r = 32, 32, 2
W = torch.randn(d_out, d_in)  # frozen
A = nn.Parameter(torch.randn(r, d_in) * 0.01)
B = nn.Parameter(torch.zeros(d_out, r))
alpha = 16.0

def lora_forward(x):
    return x @ W.T + (x @ A.T @ B.T) * (alpha / r)

x = torch.randn(4, d_in)
y0 = lora_forward(x)
opt = torch.optim.Adam([A, B], lr=0.05)
for _ in range(50):
    y = lora_forward(x)
    loss = (y ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
y1 = lora_forward(x)
print("output norm before", round(y0.norm().item(), 3), "after", round(y1.norm().item(), 3))
print("Base W was never in the optimizer — frozen.")
"""
)


md(
    """
## MXFP4 vs classic QLoRA (project-specific)

Classic QLoRA: BitsAndBytes 4-bit base + LoRA.

GPT-OSS: **native MXFP4**. We use `Mxfp4Config(dequantize=True)` → BF16, then LoRA.
Do **not** stack BitsAndBytes on MXFP4. Scripts say `qlora` but the recipe is
**MXFP4-dequantize + LoRA**.
"""
)

md(
    """
---
# Part F — Map onto `clinical-gpt-oss`

## Data walls
1. **Patient-level** 70/15/15 (seed 42) — no patient leakage across splits.
2. **DAPT vs SFT encounter disjoint** — Stage 1 never sees gold A&P used as Stage 2 targets for the same visit.

## Stage 1 (actual)
`--text-only`, 1 epoch, lr `1e-4`, batch 1, accum 16, seq 4096, LoRA r=8
→ `results/gpt-oss-20b-domain-lora/`

## Stage 2 (actual)
chat SFT, `--adapter-init` Stage 1, 2 epochs, lr `2e-4`, same batch/seq/LoRA
→ `results/gpt-oss-20b-ap-lora/`

## Eval
| Stage | Metric |
|---|---|
| 1 | Perplexity |
| 2 | Token F1, Entity F1, Section recall |

Domain-only ≈ base on A&P metrics; **domain+ap** is what unlocks drafting
(Entity F1 ~0.48, section recall ~0.62 on n=25 smoke test).
"""
)

code(
    """
def section_recall(gold: str, pred: str) -> float:
    gold_secs = set(re.findall(r"(?m)^-\\s*([^:]+):", gold))
    if not gold_secs:
        return 1.0
    pred_l = pred.lower()
    hits = sum(1 for s in gold_secs if s.strip().lower() in pred_l)
    return hits / len(gold_secs)

gold = \"\"\"- Graft function: Excellent
- Immunosuppression: tacrolimus therapeutic
- Follow-up: 6 months
\"\"\"
pred_bad = \"\"\"- Graft function: Excellent
- Follow-up: 6 months
\"\"\"
print("perfect:", section_recall(gold, gold))
print("missing IS:", round(section_recall(gold, pred_bad), 3))
"""
)

md(
    """
---
# Part G — Checklist before you change training

1. Changing the **objective**? (raw LM vs chat SFT)
2. **Leakage**? (patient split? DAPT/SFT overlap?)
3. Training **LoRA** or accidentally the full base?
4. Sequences fit **`max_seq_length`**?
5. At inference, parsing Harmony **`final`**?
6. Metrics match the claim? (PPL ≠ safety; F1 ≠ hepatologist OK)

---
# Part H — What to read next in the repo

1. This notebook
2. `STAGE12_DAPT_AND_SFT_DETAILED.md`
3. `TRAINING_CURRICULUM.md`
4. `training/train_sft_trl.py`
5. `training/harmony_output.py`

---
## You should now be able to explain

1. Base LMs predict next tokens via NLL/cross-entropy.
2. MoE routes each token to top-k experts.
3. “Reasoning” here is an analysis scratchpad before final (Harmony).
4. We adapt gpt-oss with LoRA: Stage 1 domain LM → Stage 2 A&P SFT.
5. Batch, accum, lr, epochs, seq length, schedules drive optimization.
6. Eval proxies are useful but not clinic sign-off.
"""
)

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

out = Path(__file__).resolve().parent / "01_base_model_moe_reasoning_and_training.ipynb"
nbf.write(nb, out)
print("Wrote", out, "n_cells=", len(cells))
