"""
Zadanie 2: Wyszukiwarka dokumentów

Napisz wyszukiwarkę dokumentów ze zbioru 20newsgroups lub innego dużego korpusu.
Do wyszukiwarki można podać dokument (łańcuch znaków), dla którego zwróci ona
zadaną liczbę dokumentów z korpusu o największym podobieństwie kosinusowym.

Następnie:
a) sprawdź działanie wyszukiwarki na przykładzie, oceń jej skuteczność
b) (*) wypróbuj iloczyn skalarny zamiast podobieństwa kosinusowego
c) (*) wypróbuj różne sposoby wyliczania BoW (wektor lub liczba wystąpień), TF, IDF
"""

# Todo: komy z klas/kodu wrzucic do collaba i opisac w jaki sposob to dziala!

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
from typing import Tuple, List, Union
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Klasa reprezentująca wynik wyszukiwania."""
    index: int
    similarity: float
    document: str
    category: str


class DocumentSearcher:
    """
    Klasa do wyszukiwania podobnych dokumentów w korpusie.
    """
    
    def __init__(
        self,
        corpus_matrix: csr_matrix,
        vectorizer,
        documents: List[str],
        categories: List[str] = None
    ):
        """
        Inicjalizacja wyszukiwarki.
        
        Parameters:
        -----------
        corpus_matrix : csr_matrix
            Macierz dokumentów (dokumenty x cechy)
        vectorizer : object
            Wektoryzator z metodą transform()
        documents : List[str]
            Lista dokumentów tekstowych
        categories : List[str], optional
            Lista kategorii dokumentów
        """
        self.corpus_matrix = corpus_matrix
        self.vectorizer = vectorizer
        self.documents = documents
        self.categories = categories or [None] * len(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 7,
        similarity_metric: str = 'cosine'
    ) -> List[SearchResult]:
        """
        Wyszukuje najbardziej podobne dokumenty do zapytania.
        
        Parameters:
        -----------
        query : str
            Dokument zapytania
        top_k : int
            Liczba dokumentów do zwrócenia
        similarity_metric : str
            Metryka podobieństwa: 'cosine' lub 'dot'
            
        Returns:
        --------
        List[SearchResult]
            Lista wyników posortowana od największego podobieństwa
        """
        # Przekształć zapytanie na wektor
        query_vector = self.vectorizer.transform([query])
        
        # Oblicz podobieństwo
        if similarity_metric == 'cosine':
            similarities = cosine_similarity(query_vector, self.corpus_matrix).flatten()
        elif similarity_metric == 'dot':
            similarities = (query_vector @ self.corpus_matrix.T).toarray().flatten()
        else:
            raise ValueError(f"Nieznana metryka: {similarity_metric}. Użyj 'cosine' lub 'dot'")
        
        # Znajdź top_k najbardziej podobnych dokumentów
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Utwórz listę wyników
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                index=idx,
                similarity=float(similarities[idx]),
                document=self.documents[idx],
                category=self.categories[idx] if idx < len(self.categories) else None
            ))
        
        return results


def prepare_vectorizers(
    documents: List[str],
    max_features: int = 10000
) -> dict:
    """
    Przygotowuje różne typy wektoryzatorów i macierze dokumentów.
    
    Parameters:
    -----------
    documents : List[str]
        Lista dokumentów do wektoryzacji
    max_features : int
        Maksymalna liczba cech
        
    Returns:
    --------
    dict
        Słownik z wektoryzatorami i macierzami:
        - 'tfidf': (vectorizer, matrix) - TF-IDF bezpośrednio
        - 'count': (vectorizer, matrix) - Count (TF)
        - 'tfidf_from_count': (vectorizer, matrix) - TF-IDF z CountVectorizer
    """
    results = {}
    
    # TF-IDF bezpośrednio
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X_tfidf = tfidf_vectorizer.fit_transform(documents)
    results['tfidf'] = (tfidf_vectorizer, X_tfidf)
    
    # Count (TF)
    count_vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
    X_count = count_vectorizer.fit_transform(documents)
    results['count'] = (count_vectorizer, X_count)
    
    # TF-IDF z CountVectorizer
    tfidf_transformer = TfidfTransformer()
    X_tfidf_from_count = tfidf_transformer.fit_transform(X_count)
    results['tfidf_from_count'] = (count_vectorizer, X_tfidf_from_count)
    
    return results


def display_results(
    results: List[SearchResult],
    title: str,
    preview_length: int = 500
):
    """
    Wyświetla wyniki wyszukiwania w czytelnej formie.
    
    Parameters:
    -----------
    results : List[SearchResult]
        Lista wyników wyszukiwania
    title : str
        Tytuł sekcji wyników
    preview_length : int
        Długość podglądu dokumentu w znakach
    """
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Dokument #{result.index} (Podobieństwo: {result.similarity:.4f})")
        if result.category:
            print(f"Kategoria: {result.category}")
        print(f"Podgląd ({preview_length} znaków):")
        print("-" * 80)
        preview = result.document[:preview_length]
        if len(result.document) > preview_length:
            preview += "..."
        print(preview)
    print("-" * 80)


def compare_results(
    results1: List[SearchResult],
    results2: List[SearchResult],
    label1: str,
    label2: str
):
    """
    Porównuje wyniki dwóch wyszukiwań.
    
    Parameters:
    -----------
    results1 : List[SearchResult]
        Pierwsza lista wyników
    results2 : List[SearchResult]
        Druga lista wyników
    label1 : str
        Etykieta pierwszej metody
    label2 : str
        Etykieta drugiej metody
    """
    print(f"\n{'=' * 80}")
    print(f"PORÓWNANIE: {label1} vs {label2}")
    print(f"{'=' * 80}")
    
    indices1 = {r.index for r in results1}
    indices2 = {r.index for r in results2}
    
    common = indices1 & indices2
    only1 = indices1 - indices2
    only2 = indices2 - indices1
    
    print(f"\nWspólne dokumenty: {len(common)}")
    if common:
        print(f"  Indeksy: {sorted(common)}")
    
    print(f"\nTylko w {label1}: {len(only1)}")
    if only1:
        print(f"  Indeksy: {sorted(only1)}")
    
    print(f"\nTylko w {label2}: {len(only2)}")
    if only2:
        print(f"  Indeksy: {sorted(only2)}")


def run_all_experiments(query: str, top_k: int = 5):
    """
    Uruchamia wszystkie eksperymenty z zadania 2.
    
    Parameters:
    -----------
    query : str
        Dokument zapytania
    top_k : int
        Liczba dokumentów do zwrócenia
    """
    print("=" * 80)
    print("ZADANIE 2: WYSZUKIWARKA DOKUMENTÓW")
    print("=" * 80)
    print(f"\nZapytanie: '{query}'")
    print(f"Liczba wyników: {top_k}")
    
    # Wczytaj dane
    print("\nWczytywanie danych 20newsgroups...")
    newsgroups = fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes')
    )
    
    # Przygotuj wektoryzatory
    print("Przygotowywanie wektoryzatorów...")
    vectorizers = prepare_vectorizers(newsgroups.data)
    
    # Pobierz nazwy kategorii
    category_names = [newsgroups.target_names[cat] for cat in newsgroups.target]
    
    # ========================================================================
    # PODPUNKT A: Podstawowa wyszukiwarka z podobieństwem kosinusowym
    # ========================================================================
    print("\n" + "=" * 80)
    print("PODPUNKT A: Podobieństwo kosinusowe (TF-IDF)")
    print("=" * 80)
    
    vectorizer_tfidf, X_tfidf = vectorizers['tfidf']
    searcher_tfidf = DocumentSearcher(
        X_tfidf, vectorizer_tfidf, newsgroups.data, category_names
    )
    
    results_cosine = searcher_tfidf.search(query, top_k, similarity_metric='cosine')
    display_results(results_cosine, "Top dokumenty - Podobieństwo kosinusowe (TF-IDF)")
    
    # ========================================================================
    # PODPUNKT B: Iloczyn skalarny
    # ========================================================================
    print("\n" + "=" * 80)
    print("PODPUNKT B: Iloczyn skalarny (TF-IDF)")
    print("=" * 80)
    
    results_dot = searcher_tfidf.search(query, top_k, similarity_metric='dot')
    display_results(results_dot, "Top dokumenty - Iloczyn skalarny (TF-IDF)")
    
    # Porównaj wyniki
    compare_results(
        results_cosine, results_dot,
        "Podobieństwo kosinusowe", "Iloczyn skalarny"
    )
    
    # ========================================================================
    # PODPUNKT C: Różne metody wektoryzacji
    # ========================================================================
    print("\n" + "=" * 80)
    print("PODPUNKT C: Różne metody wektoryzacji")
    print("=" * 80)
    
    # C.1: Count (TF) z podobieństwem kosinusowym
    vectorizer_count, X_count = vectorizers['count']
    searcher_count = DocumentSearcher(
        X_count, vectorizer_count, newsgroups.data, category_names
    )
    results_tf = searcher_count.search(query, top_k, similarity_metric='cosine')
    display_results(results_tf, "Top dokumenty - TF (CountVectorizer)")
    
    # C.2: TF-IDF z CountVectorizer
    _, X_tfidf_from_count = vectorizers['tfidf_from_count']
    searcher_tfidf_count = DocumentSearcher(
        X_tfidf_from_count, vectorizer_count, newsgroups.data, category_names
    )
    results_tfidf_count = searcher_tfidf_count.search(query, top_k, similarity_metric='cosine')
    display_results(results_tfidf_count, "Top dokumenty - TF-IDF (z CountVectorizer)")
    
    # Porównaj wszystkie metody
    print("\n" + "=" * 80)
    print("PODSUMOWANIE PORÓWNAŃ")
    print("=" * 80)
    
    compare_results(
        results_cosine, results_tf,
        "TF-IDF (bezpośrednio)", "TF (CountVectorizer)"
    )
    
    compare_results(
        results_cosine, results_tfidf_count,
        "TF-IDF (bezpośrednio)", "TF-IDF (z CountVectorizer)"
    )
    
    compare_results(
        results_tf, results_tfidf_count,
        "TF (CountVectorizer)", "TF-IDF (z CountVectorizer)"
    )


if __name__ == "__main__":
    # Uwaga: 20newsgroups to korpus z lat 90. o różnych tematach (religia, polityka, technologia, itp.)
    # Zapytania powinny być związane z tymi tematami dla lepszych wyników
    
    # query = "cars are vehicles with four wheels and bell"
    
    # query = "What is the capital of Poland?"
    
    query = "What do Christians believe about God?"
    
    # query = "baseball players and teams statistics"
    
    run_all_experiments(query, top_k=5)


