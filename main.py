from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = ["The dog walked over there.",
  "The cat ran from it.",
  "I want to bake cookies tonight.",
  "My pet walked outside last night."
  ]

embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)

print(similarities)