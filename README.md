# Word2Vec in Pure NumPy

## Overview

This project implements the core training loop of **word2vec** in pure NumPy, without using any deep learning frameworks such as PyTorch or TensorFlow.

The implementation uses:

* **Skip-gram model**
* **Negative sampling**
* **Stochastic Gradient Descent (SGD)**

---

## Model

For a center word embedding $ v_c $, a positive context embedding $ u_o $, and negative context embeddings $ u_k $, the loss for one training example is:

$$
L = -\log \sigma(u_o^T v_c) - \sum_k \log \sigma(-u_k^T v_c)
$$

where:

* $\sigma(x) = \frac{1}{1 + e^{-x}}$  is the sigmoid function
* the first term encourages true context words to have high similarity
* the second term pushes negative samples away

---

## Gradients

Let $ x = u_o^T v_c $ and $ z_k = u_k^T v_c $.

Then:

* Positive term:
  $$
  \frac{d}{dx}[-\log \sigma(x)] = \sigma(x) - 1
  $$
* Negative term:
  $$
  \frac{d}{dz_k}[-\log \sigma(-z_k)] = \sigma(z_k)
  $$

From this:

* $ \frac{\partial L}{\partial u_o} = (\sigma(x) - 1) v_c $
* $ \frac{\partial L}{\partial u_k} = \sigma(z_k) v_c $
* $ \frac{\partial L}{\partial v_c} = (\sigma(x) - 1) u_o + \sum_k \sigma(z_k) u_k $

These gradients are implemented directly in `word2vec.py`.

---

## Files

* `word2vec.py` — model, loss, gradients, and updates
* `train.py` — data loading and training loop
* `data/text8_processed.txt` — processed dataset
* `preprocess.py` — one-time script to preprocess raw text8 data

---

## Dataset

This project uses a **subset of the Text8 dataset**, a cleaned Wikipedia corpus commonly used for word2vec experiments.

To keep runtime manageable in a pure NumPy implementation:

* only a subset of tokens is used
* a minimum frequency threshold (`min_count`) filters rare words

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python train.py --data_path data/text8_processed.txt --epochs 2 --min_count 5 --window_size 1
```

---

## Implementation Details

* Two embedding matrices are learned:

  * `W_in`: center word embeddings
  * `W_out`: context word embeddings

* Training uses:

  * one training pair at a time (no batching)
  * negative sampling with unigram distribution raised to the power 0.75
  * plain SGD updates

* Skip-gram pairs are generated using a fixed window size

---

## Sample Output

During training, the loss decreases:

```
Epoch 1/2 - avg loss: ~3.0
Epoch 2/2 - avg loss: lower
```

This indicates the model is learning to distinguish positive and negative contexts.

---



