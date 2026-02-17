from sentence_transformers import SentenceTransformer
import faiss
from sklearn.datasets import fetch_20newsgroups
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# sentences = ["The dog walked over there.",
#   "The cat ran from it.",
#   "I want to bake cookies tonight.",
#   "My pet walked outside last night."
#   ]

categories = ['sci.crypt', 'sci.electronics', 'comp.sys.ibm.pc.hardware', 'misc.forsale']

newsgroups = fetch_20newsgroups(subset='train', categories=categories)
documents = newsgroups.data

embeddings = model.encode(documents)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))


query = "encryption"
query_vector = model.encode([query])
distances, indices = index.search(np.array(query_vector), 5)

# Display results
print(f"Query: {query}\n")

for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
    print(f"Result {rank + 1} (distance: {dist}):")
    print(f"Category: {newsgroups.target_names[newsgroups.target[idx]]}")
    print(f"Preview: {documents[idx][:200]}...\n")