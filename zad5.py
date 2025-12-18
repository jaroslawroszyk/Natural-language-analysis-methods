import re
from gensim.models import KeyedVectors, Word2Vec
from sklearn.datasets import fetch_20newsgroups

# ================================
# A) WORD2VEC GOOGLE NEWS
# ================================

import gensim.downloader as api

print("\n=== A) Word2Vec Google News ===")
w2v_google = api.load("word2vec-google-news-300")


words = ["hockey", "windows", "religion", "gun", "space"]

for w in words:
    print(f"\n{w}:")
    print(w2v_google.most_similar(w, topn=5))


# ================================
# B) WORD2VEC – 20 NEWSGROUPS
# ================================

print("\n=== B) Word2Vec trenowany na 20newsgroups ===")

data = fetch_20newsgroups(remove=("headers", "footers", "quotes"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z ]", " ", text)
    return text.split()

sentences = [preprocess(doc) for doc in data.data]

w2v_20ng = Word2Vec(
    sentences,
    vector_size=300,
    window=5,
    min_count=5,
    workers=4
)

for w in words:
    if w in w2v_20ng.wv:
        print(f"\n{w}:")
        print(w2v_20ng.wv.most_similar(w, topn=5))


# ================================
# C) WPŁYW PARAMETRÓW (WINDOW)
# ================================

print("\n=== C) Wpływ wielkości okna ===")

w2v_window2 = Word2Vec(sentences, vector_size=300, window=2, min_count=5)
w2v_window10 = Word2Vec(sentences, vector_size=300, window=10, min_count=5)

word = "gun"
print(f"\nMałe okno (window=2) dla '{word}':")
print(w2v_window2.wv.most_similar(word, topn=5))

print(f"\nDuże okno (window=10) dla '{word}':")
print(w2v_window10.wv.most_similar(word, topn=5))


# ================================
# D) FASTTEXT i GLOVE
# ================================

print("\n=== D) FastText i GloVe ===")


fasttext = api.load("fasttext-wiki-news-subwords-300")
glove = api.load("glove-wiki-gigaword-300")


for w in words:
    print(f"\n--- {w.upper()} ---")

    print("FastText:")
    if w in fasttext:
        print(fasttext.most_similar(w, topn=5))
    else:
        print("brak w słowniku")

    print("GloVe:")
    if w in glove:
        print(glove.most_similar(w, topn=5))
    else:
        print("brak w słowniku")
