with open("text8", "r", encoding="utf-8") as f:
    text = f.read()

words = text.split()
words = words[:200000]  # keep only first 200k tokens to speed up processing

sentences = [" ".join(words[i:i+30]) for i in range(0, len(words), 30)]

with open("data/text8_processed.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sentences))