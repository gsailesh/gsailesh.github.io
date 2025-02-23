# Chapter 3: Coding Attention Mechanisms

Code: https://github.com/gsailesh/self-learn/blob/llm-book/Build-LLMs-from-scratch/03a.coding_attention.ipynb

### Illustration: Self Attention without trainable weights

An illustration of a simple self attention (sans weights):

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image.png)

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

## Additional Figures

![image.png](Chapter%203%20Coding%20Attention%20Mechanisms%2018d345517e4b804eb3becf79ab3171f0/image%209.png)