from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_store = FAISS.load_local("faiss_index2", embeddings,allow_dangerous_deserialization=True)


'''
# Get the raw FAISS index
faiss_index = faiss_store.index
# Number of vectors
print("Number of vectors in FAISS index:", faiss_index.ntotal)

# Dimensionality (embedding size)
print("Dimension of vectors:", faiss_index.d)

'''



import time

query = "medicine for cold ?"
start_time = time.time()
docs = faiss_store.similarity_search(query, k=2)
end_time = time.time()

print(f"Retrieval time: {end_time - start_time:.4f} seconds")
print("Retrieved docs:", docs)
