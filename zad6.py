import re
import numpy as np
import gensim.downloader as api
from gensim.models import Word2Vec
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity

data = fetch_20newsgroups(remove=("headers", "footers", "quotes"))
documents = data.data


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    return text.split()

def document_vector(tokens, model):
    tokens = [t for t in tokens if t in model]
    if not tokens:
        return np.zeros(model.vector_size)
    return model.get_mean_vector(tokens)


print("\n=== Zadanie 6a: Google News Word2Vec ===")

w2v_google = api.load("word2vec-google-news-300")

doc_vectors = []
for doc in documents:
    tokens = preprocess(doc)
    doc_vectors.append(document_vector(tokens, w2v_google))

doc_vectors = np.array(doc_vectors)


def search(query, model, doc_vectors, documents, top_k=5):
    query_tokens = preprocess(query)
    query_vec = document_vector(query_tokens, model).reshape(1, -1)

    sims = cosine_similarity(query_vec, doc_vectors)[0]
    top_idx = sims.argsort()[::-1][:top_k]

    return [(sims[i], documents[i]) for i in top_idx]

query = "space shuttle nasa mission"
results = search(query, w2v_google, doc_vectors, documents)

for score, doc in results:
    print(f"\nSIM={score:.4f}\n{doc[:400]}")


print("\n=== Zadanie 6b: Word2Vec trenowany na 20newsgroups ===")

sentences = [preprocess(doc) for doc in documents]

w2v_20ng = Word2Vec(
    sentences,
    vector_size=300,
    window=5,
    min_count=5,
    workers=4
)

doc_vectors_20ng = []
for doc in documents:
    tokens = preprocess(doc)
    doc_vectors_20ng.append(document_vector(tokens, w2v_20ng.wv))

doc_vectors_20ng = np.array(doc_vectors_20ng)

results = search(query, w2v_20ng.wv, doc_vectors_20ng, documents)

for score, doc in results:
    print(f"\nSIM={score:.4f}\n{doc[:400]}")
