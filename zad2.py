# **Zadanie 2.** Napisz wyszukiwarkę dokumentów ze zbioru 20newsgroups lub innego dużego korpusu. Do wyszukiwarki można podać dokument (łańcuch znaków), dla którego zwróci ona zadaną liczbę dokumentów z korpusu o największym podobieństwie kosinusowym. Zwracana lista ma być posortowana w porządku nierosnącego prawdopodobieństwa.

# Następnie:

# a) sprawdź działanie wyszukiwarki na przykładzie, oceń jej skuteczność i opisz to w sprawozdaniu

# b) **(*)** wypróbuj iloczyn skalarny zamiast podobieństwa kosinusowego, porównaj do wyników z punktu a) i opisz w sprawozdaniu różnicę i jej przyczyny,

# c) **(*)** wypróbuj różne sposoby wyliczania BoW (wektor lub liczba wystąpień), TF, IDF i porównaj wyniki do poprzednich podpunktów. Opisz różnice i ich przyczyny w sprawozdaniu.


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# przetwarzam tekst do macierzy TF-IDF - najbardziej istotne cechy
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(newsgroups.data)  # macierz dokumentów (rows) i ich cech (cols)


def document_searcher(document, X, vectorizer, count_return = 7):
    """
    Funkcja zwracająca oczekiwaną liczbę dokumentów z 20newsgroups w kolejności od największego do najmniejszego podobieństwa kosinusowego.
    
    :param document: dokument, na podstawie którego wyszukiwane są dokumeny ze zbioru
    :param X: macież wszystkich dokumentow 20newsgroup
    :param vectorizer: tekst z datasetu przekształcony na wektor TF-IDF
    :param count_return: liczba oczekiwanych zwroconych odkumentow
    :return: lista dokumentow ze zbioru 20newsgroup
    """
    query_vec = vectorizer.transform([document])
    cosine_similarities = cosine_similarity(query_vec, X).flatten()
    related_documents_indices = cosine_similarities.argsort()[-count_return:][::-1]

    return related_documents_indices, cosine_similarities[related_documents_indices]


dokument = "cars are vehicles with four wheels and bell"
related_documents_indices, similarities = document_searcher(dokument, X, vectorizer, count_return=5)

print("Top 5 najbardziej podobnych dokumentów:")

for idx, sim in zip(related_documents_indices, similarities):
    print(f"Dokument {idx} (Podobieństwo: {sim:.2f}):")
    print(newsgroups.data[idx][:500])  # pokazuję pierwsze 500 znaków
    print("-" * 80)