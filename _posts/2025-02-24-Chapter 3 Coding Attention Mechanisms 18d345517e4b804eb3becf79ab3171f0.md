---
layout: page
title: "Chapter 3: Coding Attention mechanism"
date: 2025-02-23 20:40:00 +0530
categories: Notes Deep-learning
---



Code: https://github.com/gsailesh/self-learn/blob/llm-book/Build-LLMs-from-scratch/03a.coding_attention.ipynb

### Illustration: Self Attention without trainable weights

An illustration of a simple self attention (sans weights):

![image.png](gsailesh.github.io/_posts/2025-02-24-Chapter 3 Coding Attention Mechanisms/image.png)

- The input sequence $X_{1}$ is *Your journey starts…*
- Each token in this sequence may be represented as $x^{(1)}$ = Your **and $x^{(2)}$ = journey and so on
- The illustration depicts the context vector element $z^{(2)}$ that is computed from all of the input tokens weighted by specific attention weights - one for each token
- Finally a context vector $Z_{1}$ is made up of the elements $z^{(1)}, z^{(2)},$ etc.
- The purpose of a context vector is to create enriched representations of each element in an input sequence by incorporating information from all other elements in the sequence. This is essential in LLMs.

Below illustration shows how the intermediate attention scores (weights) are computed for the query token $x^{(2)}$:

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%201.png)

Attention weights are calculated by normalizing (softmax) the scores:

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%202.png)

Finally, the context vector for the second token is obtained:

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%203.png)

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%204.png)

### Illustration: Self Attention with trainable weights

Below illustration chooses the same input token $x^{(2)}$. The query vector for the token is $q^{(2)}$ and is derived from the randomly initialized weight matrix $W_{q}$ by multiplying with the input token $x^{(2)}$ Similarly, the keys and values are derived from $W_{k}$ and $W_{v}$ respectively through multiplications with the input sequence.

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%205.png)

The *attention scores* are computed for each input token based on the query vector ($q^{(2)}$, in the below illustration) 

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%206.png)

The *attention weights* are computed as follows:

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%207.png)

Query, key and values are terms borrowed from the domain of *information retrieval* and *databases*.

*Query* represents the input query which the model tries to identify the relations with the rest of the input sequences.

*Key* is the index that maps with the different tokens in the input sequences, and it comes with the corresponding values stored in *Value*.

## An overview of Self Attention

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%208.png)

## Causal Attention and Multi-head Attention

The causal aspect involves modifying the attention mechanism to prevent the model from accessing future information in the sequence, which is crucial for tasks like language modelling, where each word prediction should only depend on previous words.

The multi-head component involves splitting the attention mechanism into multiple “heads.” Each head learns different aspects of the data, allowing the model to simultaneously attend to information from different representation subspaces at different positions. This improves the model’s performance in complex tasks.

An illustration of masked attention (causal attention) is given below:

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%209.png)

The normalized attention weights are masked by multiplying them with a *lower* triangular matrix of **1**s (to obtain masked scores) and then re-normalized (to masked weights).

```python
mask_simple = torch.tril(torch.ones(context_length, context_length))
masked_simple = attn_weights*mask_simple

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
```

However, there’s a hint of information leakage here since the attention weights are normalized prior to the application of masks. This is however untrue as there’s no evidence of masked weights affecting the softmax output. Intuitively, this is because the masking step prohibits these attention weights to be available to softmax. Since they’re are essentially masked, they are not available for the softmax, i.e., the softmax is applied only on the unmasked subset.

However, there’s another efficient method to implement causal attention. This is by simply setting the masked elements to $-\infty$.

```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf) # Apply to attention scores.

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
```

Additionally, *dropout* techniques may be introduced to the GPT (or transformer module) to help them learn different aspects of the data. The dropout may be applied in two ways:

1. To the masked attention weights (common)
2. After applying masked attention weights to the values matrix

An illustration of dropout technique is given:

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%2010.png)

## Multi-head Attention

An illustration is MHA is provided:

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%2011.png)

Another illustration:

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%2012.png)

## Summary

- Attention mechanisms transform input elements into enhanced context vector representations that incorporate information about all inputs.
- A self-attention mechanism computes the context vector representation as a weighted sum over the inputs.
- Scaled dot product attention is used to softmax from saturating and gradients from becoming zero during training.
- A causal attention mask can prevent the LLM from accessing future tokens. A dropout mask to reduce overfitting in LLMs.
- The attention modules in transformer-based LLMs involve multiple instances of causal attention, which is called multi-head attention.
- A multi-head attention can be created by stacking multiple instances of causal attention modules, or more efficiently using batched matrix multiplications.

## Additional Figures

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%2013.png)