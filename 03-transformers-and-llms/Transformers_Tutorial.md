## What are things that are randomized automatically?

1. Model weights are randomly initialized

```python
layer = nn.Linear(8,8)
```
Python automatically creates random starting values for:

```python
layer.weight
layer.bias
```
You can see them:
```python
print(layer.weight)
print(layer.bias)
```

PyTorch initializes each weight approximately from: $\frac{-1}{\sqrt{d}$ to $\frac{+1}{\sqrt{d}$ 

Every number inside this interval has an equal chance of being selected. This is called a uniform distribution.
```python
import math

d = 4

lower_bound = -1 / math.sqrt(d)
upper_bound =  1 / math.sqrt(d)

print(lower_bound)
print(upper_bound)
```

In attention, these layers all start with random weights:

```python
self.query = nn.Linear(d, d)
self.key   = nn.Linear(d, d)
self.value = nn.Linear(d, d)
self.proj  = nn.Linear(d, d)
```

Therefore, the query, key, value, and output projection matrices are initially random.

The weights are not randomly changed during training. After initialization, the optimizer updates them using gradients.

2. Embedding Tables are randomly intialized

When you create token embeddings:
```python
embedding = nn.Embedding(vocab_size, d_model)
```
Pytorch creates a table containing one random vector for each token.

For example:

```python
embedding = nn.Embedding(1000, 128)

print(embedding.weight.shape)
# torch.Size([1000, 128])
```
The table has 1,000 token vectors, and each vector initially contains random numbers.

Learned positional embeddings are also randomly initialized:
```python
position_embedding = nn.Embedding(max_length, d_model)
```
However, fixed positional methods such as sinusoidal embeddings or RoPE are calculated mathematically rather than randomly initialized.

3. Dropout randomly removes activation

A transformer may contain:

```python
dropout=nn.dropout(0.1)
```
During training, dropout randomly chooses some activation values and sets them to zero.

For example:

```python
x=torch.ones(10)
dropout = nn.Dropout(0.5)
dropout.train()
print(dropout(x))
```

A possible result is:

```python
tensor([2., 0., 2., 2., 0., 0., 2., 0., 2., 0.])
```

Dropout may be applied:

after attention
to attention weights
inside the MLP
after embeddings
before residual addition

It is active in training mode:
```python
model.train()
```
It is disabled in evaluation mode:
```python
model.eval()
```

`dropout.train()` puts the dropout layer into training mode.
During training, `nn.Dropout(0.5)` randomly changes each input value to 0 with a 50% probability.
The values that are not removed are scaled by dividing them by 1 - p.
Here, p = 0.5, so each remaining value is divided by 0.5, changing it from 1 to 2.
This scaling keeps the average output approximately equal to the original input.
Because a new random dropout mask is generated every time you call dropout(x), different values are removed on each call, so the output changes each time.

Note: Setting a seed does not mean PyTorch always generates the same mask. It means PyTorch starts from the same point in a reproducible sequence of random numbers. Each time dropout(x) runs, it uses the next random numbers in that sequence, so it creates a different mask.

Weights: randomly generated once when the layer is created, then stored.
Dropout mask: randomly generated again every time data passes through the dropout layer.

4. Training data may be randomly shuffled

When using:

```python
DataLoader(
    dataset,
    batch_size=32,
    shuffle=true
)
```

Pytorch randomly changes the order of training examples.

This means each epoch can contain the same examples in different order.

That affects:

- which examples appear together in a batch
- the gradient calculated for each batch
- the path the optimizer takes during training

5. Random masking or corruption may be applied

Some training tasks randomly modify the input.

For masked-language models such as BERT, tokens may be randomly selected and replaced with a mask:

```text
The patient has liver disease
              ↓
The patient has [MASK] disease
```
For GPT-style causal language modelling, ordinary next-token training does not require randomly masking individual tokens. The causal mask is fixed based on token positions.

However, data pipelines may still randomly:

- crop long documents
- select a segment
- choose a starting location
- remove tokens
- corrupt spans
- pack examples together

These operations depend on how the dataset code was written.

6. Attention dropout may be random

After the transformer calculates attention weights:

```python
weights = F.softmax(scores, dim=-1)
```
it may apply dropout:
```python
weights = self.attention_dropout(weights)
```
Some attention connections are then randomly removed during training.

The causal attention mask itself is not random:
```python
causal_mask = torch.tril(torch.ones(T, T))
```
It always prevents a token from viewing future tokens.

**Casual mask: fixed not random** 

Suppose there are 4 tokens:

```text
[Token 1 Token 2 Token 3 Token 4] 
```
The casual mask looks like this:

```python
causalmask = torch.tril(torch.ones(4,4))
```
`torch.tril(...)` keeps only the lower triangular part of that matrix and changes everything above the diagonal to 0
Output:
```pyhton
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```
Each row represents the token currently calculating attention.

For example, token 3 may look at:

```text
[Token 1, Token 2, Token 3]
```
but not token 4, because token 4 is in the future.

This mask is always the same for a sequence of the same length. Its purpose is to prevent information from leaking backward from future tokens.

**Attention dropout: random during training**

After the causal mask is applied, attention scores are converted into probabilities:

```python
weights = F.softmax(scores, dim=-1)
```
For token 3, the attention weights might be:

```python
[0.20, 0.50, 0.30, 0.00]
```
This means token 3 gives:

- 20% attention to token 1
- 50% attention to token 2
- 30% attention to itself
- 0% attention to token 4 because token 4 is in the future

Then attention dropout might randomly remove one allowed connection:
```python
weights = self.attention_dropout(weights)
```
For example:

```text
Before dropout: [0.20, 0.50, 0.30, 0.00]
After dropout:  [0.40, 0.00, 0.60, 0.00]
```

Here, the connection to token 2 was randomly removed. The remaining weights were scaled upward to preserve the expected total magnitude.

On another call, dropout might remove a different connection:
```text
After dropout: [0.00, 0.625, 0.375, 0.00]
```

The key distinction is:
```text
Causal mask:
“Which tokens are legally allowed to be seen?”

Attention dropout:
“Among the allowed connections, which ones are temporarily removed during this training step?”
```
During evaluation mode:
```python
model.eval()
```
attention dropout is turned off, but the causal mask is still applied.

In attention dropout, the model must also learn to use information from other tokens. This helps it learn more robust patterns instead of memorizing specific connections.

7. Generation may use random token sampling

When a language model generates text, it does not directly know the next word. It produces probabilities (a score) for every possible next token, converts those scores into probabilities, and then chooses one token. Suppose the prompt is:

```text
The patient has:
```

The model might assign these probabilities:

```text
cirrhosis     0.40
pain          0.25
fever         0.15
improved      0.10
a             0.06
other tokens  0.04
```

The probabilities add up to `1.0`

1. Greedy decoding: always choose the largest probability
```python
next_token = torch.argmax(probabilities)
```

`argmax` returns the position of the largest value


```python
import torch

tokens = ["cirrhosis", "pain", "fever", "improved", "a"]

probabilities = torch.tensor([
    0.40,
    0.25,
    0.15,
    0.10,
    0.10
])

index = torch.argmax(probabilities)

print(index)
print(tokens[index])
```
Output:

```python
tensor(0)
cirrhosis
```
Because cirrhosis has the highest probability, greedy decoding always selects it.
Running the code repeatedly gives the same result:

```python
cirrhosis
cirrhosis
cirrhosis
cirrhosis
```
This is deterministic.
However, choosing the highest-probability token at every step does not necessarily produce the best complete sentence. It only makes the locally most likely choice at each step.

2. Random Sampling: choose according to the probabilities

With sampling:

```python

next_token = torch.multinomial(
    probabilities,
    num_samples=1
)
```

A token with probability 0.40 should be selected approximately 40% of the time. A token with probability 0.25 should be selected approximately 25% of the time.

```python
import torch

torch.manual_seed(42)

tokens = ["cirrhosis", "pain", "fever", "improved", "a"]

probabilities = torch.tensor([
    0.40,
    0.25,
    0.15,
    0.10,
    0.10
])

for _ in range(10):
    index = torch.multinomial(
        probabilities,
        num_samples=1
    )

    print(tokens[index.item()])
```

Possible Output:

```python
cirrhosis
cirrhosis
pain
cirrhosis
fever
pain
improved
cirrhosis
a
pain
```
The model is not choosing every token equally. It is using a weighted random choice.
cirrhosis is more likely than a, but a is still possible. cirrhosis is more likely than a, but a is still possible


3.  The model initially produces logits, not probabilities
Before probabilities, the model produces raw scores called logits.
For example:

```python

logits = torch.tensor([
    3.0,   # cirrhosis
    2.2,   # pain
    1.4,   # fever
    0.7,   # improved
    0.2    # a
])
```
Logits:
- do not need to be between 0 and 1
- may be positive or negative
- do not add up to 1
- Softmax converts logits into probabilities:
  
```python
probabilities = torch.softmax(logits, dim=-1)
```

Complete example:

```python

import torch

tokens = ["cirrhosis", "pain", "fever", "improved", "a"]

logits = torch.tensor([
    3.0,
    2.2,
    1.4,
    0.7,
    0.2
])

probabilities = torch.softmax(logits, dim=-1)

for token, probability in zip(tokens, probabilities):
    print(token, probability.item())
```
The softmax formula is:
$$ P_i = \frac{e^{z_i}}{\sum_j e_j}$$	

where:

- $z_i$ is the token's logit 
- $P_i$is its probability
- the denominator sums over all tokens
- 
The token with the largest logit also gets the largest probability

4. Temperature changes how concentrated the probabilities are

Temperature modifies the logits before softmax:

$$ P_i = \frac{e^{z_i/T}}{\sum_j e_j/T}$$	

In code:

```python
probabilities = torch.softmax(
    logits / temperature,
    dim = -1
)
```

Low temperature

Suppose:

```python 
temperature = 0.5
```
Dividing by 0.5 makes the logits larger:

```python

Original logits:
3.0, 2.2, 1.4, 0.7, 0.2

After division by 0.5:
6.0, 4.4, 2.8, 1.4, 0.4
```
The largest logit becomes much more dominant.

The resulting probabilities might look approximately like:

```python

cirrhosis    0.79
pain         0.16
fever        0.03
improved     0.01
a            0.01

```

The outcome becomes more predictable and conservative.
