import numpy as np
from collections import Counter
from typing import List, Tuple, Dict


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid
    x = np.clip(x, -15, 15)
    return 1.0 / (1.0 + np.exp(-x))


class Vocabulary:
    def __init__(self, min_count: int = 1):
        self.min_count = min_count
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: List[str] = []
        self.counts: np.ndarray = np.array([], dtype=np.int64)

    def build(self, tokenized_sentences: List[List[str]]) -> None:
        counter = Counter()
        for sent in tokenized_sentences:
            counter.update(sent)

        kept = [(w, c) for w, c in counter.items() if c >= self.min_count]
        kept.sort(key=lambda x: (-x[1], x[0]))

        self.id_to_word = [w for w, _ in kept]
        self.word_to_id = {w: i for i, w in enumerate(self.id_to_word)}
        self.counts = np.array([c for _, c in kept], dtype=np.int64)

    def encode_sentence(self, sentence: List[str]) -> List[int]:
        return [self.word_to_id[w] for w in sentence if w in self.word_to_id]

    def __len__(self) -> int:
        return len(self.id_to_word)


def tokenize_text(text: str) -> List[List[str]]:
    # Simple tokenization: lowercase and split by whitespace
    lines = text.strip().lower().splitlines()
    sentences = []
    for line in lines:
        tokens = line.strip().split()
        if tokens:
            sentences.append(tokens)
    return sentences


def generate_skipgram_pairs(
    encoded_sentences: List[List[int]],
    window_size: int
) -> List[Tuple[int, int]]:
    pairs = []
    for sentence in encoded_sentences:
        n = len(sentence)
        for i, center in enumerate(sentence):
            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)
            for j in range(left, right):
                if i == j:
                    continue
                context = sentence[j]
                pairs.append((center, context))
    return pairs


class NegativeSampler:
    def __init__(self, counts: np.ndarray):
        # Standard word2vec uses unigram^(3/4)
        probs = counts.astype(np.float64) ** 0.75
        probs /= probs.sum()
        self.probs = probs
        self.vocab_size = len(probs)

    def sample(self, num_samples: int, forbidden: int = None) -> np.ndarray:
        # Avoid sampling the positive context word as a negative
        samples = []
        while len(samples) < num_samples:
            candidate = np.random.choice(self.vocab_size, p=self.probs)
            if forbidden is not None and candidate == forbidden:
                continue
            samples.append(candidate)
        return np.array(samples, dtype=np.int64)


class SkipGramNegativeSampling:
    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)

        # Input embeddings: center words
        self.W_in = rng.normal(0.0, 0.1, size=(vocab_size, embedding_dim))

        # Output embeddings: context words
        self.W_out = np.zeros((vocab_size, embedding_dim), dtype=np.float64)

    def training_step(
        self,
        center_id: int,
        pos_context_id: int,
        neg_context_ids: np.ndarray,
        lr: float
    ) -> float:
        """
        One SGD step for one positive pair (center, positive context)
        plus K negative samples.

        Loss:
            -log sigma(u_o^T v_c) - sum_k log sigma(-u_k^T v_c)
        where:
            v_c = input embedding of center word
            u_o = output embedding of positive context
            u_k = output embeddings of negative contexts
        """
        v_c = self.W_in[center_id]                 # shape: (D,)
        u_o = self.W_out[pos_context_id]          # shape: (D,)
        u_neg = self.W_out[neg_context_ids]       # shape: (K, D)

        # Forward pass
        pos_score = np.dot(u_o, v_c)              # scalar
        neg_scores = u_neg @ v_c                  # shape: (K,)

        pos_sig = sigmoid(pos_score)              # sigma(u_o^T v_c)
        neg_sig = sigmoid(neg_scores)             # sigma(u_k^T v_c)

        # Loss
        eps = 1e-10
        loss_pos = -np.log(pos_sig + eps)
        loss_neg = -np.sum(np.log(1.0 - neg_sig + eps))
        loss = float(loss_pos + loss_neg)

        # Gradients
        #
        # For positive term: -log sigma(x)
        # d/dx = sigma(x) - 1
        #
        # For negative term: -log sigma(-x) = -log(1 - sigma(x))
        # d/dx = sigma(x)
        #
        # grad wrt pos score = pos_sig - 1
        # grad wrt neg scores = neg_sig
        grad_pos_score = pos_sig - 1.0            # scalar
        grad_neg_scores = neg_sig                 # shape: (K,)

        # Gradients wrt embeddings
        grad_u_o = grad_pos_score * v_c                           # (D,)
        grad_u_neg = grad_neg_scores[:, None] * v_c[None, :]      # (K, D)
        grad_v_c = grad_pos_score * u_o + np.sum(
            grad_neg_scores[:, None] * u_neg, axis=0
        )                                                         # (D,)

        # Parameter update (SGD)
        self.W_in[center_id] -= lr * grad_v_c
        self.W_out[pos_context_id] -= lr * grad_u_o
        self.W_out[neg_context_ids] -= lr * grad_u_neg

        return loss

    def get_word_vectors(self) -> np.ndarray:
        # Use input embeddings only
        return self.W_in

    def most_similar(
        self,
        word: str,
        vocab: Vocabulary,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        if word not in vocab.word_to_id:
            return []

        vectors = self.get_word_vectors()
        query_id = vocab.word_to_id[word]
        query = vectors[query_id]

        norms = np.linalg.norm(vectors, axis=1) + 1e-10
        query_norm = np.linalg.norm(query) + 1e-10
        sims = (vectors @ query) / (norms * query_norm)

        sims[query_id] = -np.inf
        best_ids = np.argsort(-sims)[:top_k]
        return [(vocab.id_to_word[i], float(sims[i])) for i in best_ids]