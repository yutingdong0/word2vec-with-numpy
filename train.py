import argparse
import numpy as np

from word2vec import (
    Vocabulary,
    tokenize_text,
    generate_skipgram_pairs,
    NegativeSampler,
    SkipGramNegativeSampling,
)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--embedding_dim", type=int, default=50)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    text = load_text(args.data_path)
    tokenized_sentences = tokenize_text(text)

    vocab = Vocabulary(min_count=args.min_count)
    vocab.build(tokenized_sentences)

    encoded_sentences = [vocab.encode_sentence(s) for s in tokenized_sentences]
    encoded_sentences = [s for s in encoded_sentences if len(s) > 1]

    pairs = generate_skipgram_pairs(encoded_sentences, args.window_size)

    model = SkipGramNegativeSampling(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        seed=args.seed
    )

    sampler = NegativeSampler(vocab.counts)

    print(f"Vocab size: {len(vocab)}")
    print(f"Training pairs: {len(pairs)}")

    for epoch in range(args.epochs):
        np.random.shuffle(pairs)
        total_loss = 0.0

        for step, (center_id, pos_context_id) in enumerate(pairs, start=1):
            neg_ids = sampler.sample(args.negatives, forbidden=pos_context_id)
            loss = model.training_step(
                center_id=center_id,
                pos_context_id=pos_context_id,
                neg_context_ids=neg_ids,
                lr=args.lr
            )
            total_loss += loss

            if step % 10000 == 0:
                avg_so_far = total_loss / step
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"step {step}/{len(pairs)} | "
                    f"avg loss so far: {avg_so_far:.4f}"
                )

        avg_loss = total_loss / max(len(pairs), 1)
        print(f"Epoch {epoch + 1}/{args.epochs} - avg loss: {avg_loss:.4f}")

        probe_words = ["city", "world", "government", "people", "history"]

    for w in probe_words:
        if w in vocab.word_to_id:
            print(f"Nearest to '{w}': {model.most_similar(w, vocab, top_k=5)}")

    np.save("embeddings.npy", model.get_word_vectors())

if __name__ == "__main__":
    main()