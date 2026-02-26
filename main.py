import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    # 1. Your "document database" (imagine these are chunks from a PDF)
    documents = [
        "The transformer architecture uses self-attention mechanisms.",
        "Python is a popular programming language for data science.",
        "Vector databases store high-dimensional embeddings for similarity search.",
        "Gradient descent minimizes the loss function during training.",
        "RAG combines retrieval systems with language model generation.",
        "Docker containers ensure reproducible deployment environments.",
        "The Eiffel Tower is located in Paris, France.",
        "LLMs are trained on massive text corpora using next-token prediction.",
    ]

    # 2. Load an embedding model (this is what Pinecone/Weaviate use internally)
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim embeddings

    # 3. Embed all documents — this is your "vector database"
    doc_embeddings = model.encode(documents)  # shape: [8, 384]
    print(f"Document matrix shape: {doc_embeddings.shape}")

    # 4. Embed a query
    query = "How do language models generate text?"
    query_embedding = model.encode([query])  # shape: [1, 384]

    # 5. Compute cosine similarity (this is what vector DBs do)
    def cosine_similarity(query_emb, doc_embs):
        # Normalize to unit length
        query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        doc_norm   = doc_embs  / np.linalg.norm(doc_embs,  axis=1, keepdims=True)
        # Dot product of normalized vectors = cosine similarity
        return (query_norm @ doc_norm.T).squeeze()

    scores = cosine_similarity(query_embedding, doc_embeddings)

    # 6. Rank and retrieve
    ranked_indices = np.argsort(scores)[::-1]
    print(f"\nQuery: '{query}'")
    print("\nTop 3 retrieved documents:")
    for i, idx in enumerate(ranked_indices[:3]):
        print(f"  {i+1}. [{scores[idx]:.3f}] {documents[idx]}")


if __name__ == "__main__":
    main()
