---
layout: page
title: "Word Embeddings - An Intuition"
date: 2020-07-12 11:40:00 +0530
categories: Notes NLP
---

These jottings are largely from Jason Brownlee's life-saver [blog](https://machinelearningmastery.com/what-are-word-embeddings/) !

### What's a word embedding ?

Word embedding is a way of representing words, where semantically similar words will have closer / similar representations. And talking about representations, they're real-valued vectors from a predefined vector space (possibly high dimensional).

### How are they learnt ?

1. Using **Embedding layers** (from scratch)

These are learnt separately, or jointly alongside the task at hand (be it classification, language modeling, etc) and are in a way, by-products. Steps are as follows:

- Input words are one-hot encoded with a fixed dimension
- The words are fed to a randomly-initialized Embedding layer, possibly of a different (lower) dimension
- The Embedding layer undergoes supervised learning with back-propagation to fine tune the embedded vectors to represent each word
- Training from scratch requires huge volumes of data and can be painfully slow. Therefore, an alternative would be to go for pre-trained word embedding, where the embedding would have already been trained on a large corpus of data and each word would have a pre-defined representation (as a word vector)

2. Using **Word2Vec / GloVe**

Word2Vec is a approach where CBOW / Skip-gram models are used to predict the word and its context. A Continuous Bag of Words (CBOW) model uses the surrounding words (or the context) to predict the current word. While the Skip-gram uses the current word to predict it's context. Visualized below:

![Word%20Embeddings%20An%20intuition%20f408808339e54877b35c7bb485b758a7/Untitled.png](Word%20Embeddings%20An%20intuition%20f408808339e54877b35c7bb485b758a7/Untitled.png)

**Source**: Word2Vec Training Models (Taken from “Efficient Estimation of Word Representations in Vector Space”, 2013)

***Word2Vec*** learns about the word using the local context. 

***GloVe***, ******on the other hand marries local context information with the global corpus statistics (using approaches like LSA) to obtain better word embeddings.
