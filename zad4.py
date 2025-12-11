"""
Zadanie 4: Analiza podobieństwa grup w 20newsgroups

Dla każdej grupy ze zbioru 20newsgroups wypisz inną grupę najbardziej i najmniej
podobną znaczeniowo. Jako miarę podobieństwa dwóch grup wykorzystaj średnie podobieństwo
kosinusowe wyliczone dla wszystkich par dokumentów z porównywanych grup.

(*) Wykonaj podpunkty z zadania 2 w tym zadaniu.
"""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from collections import defaultdict
from typing import Dict, List, Tuple
import itertools


class GroupSimilarityAnalyzer:
    """
    Klasa do analizy podobieństwa między grupami w zbiorze 20newsgroups.
    """
    
    def __init__(
        self,
        vectorizer_type: str = 'tfidf',
        similarity_metric: str = 'cosine',
        max_features: int = 10000
    ):
        """
        Inicjalizacja analizatora.
        
        Parameters:
        -----------
        vectorizer_type : str
            Typ wektoryzacji: 'tfidf', 'count', 'tfidf_from_count'
        similarity_metric : str
            Metryka podobieństwa: 'cosine' lub 'dot'
        max_features : int
            Maksymalna liczba cech
        """
        self.vectorizer_type = vectorizer_type
        self.similarity_metric = similarity_metric
        self.max_features = max_features
        
        self.vectorizer = None
        self.documents_matrix = None
        self.group_documents = {}
        self.group_indices = {}
        self.group_names = []
    
    def load_data(self, subset='train'):
        """
        Wczytuje dane 20newsgroups.
        
        Parameters:
        -----------
        subset : str
            'train', 'test' lub 'all'
        """
        print(f"Wczytywanie danych 20newsgroups (subset={subset})...")
        newsgroups = fetch_20newsgroups(
            subset=subset,
            remove=('headers', 'footers', 'quotes')
        )
        
        self.documents = newsgroups.data
        self.targets = newsgroups.target
        self.group_names = newsgroups.target_names
        
        # Grupuj dokumenty według kategorii
        self.group_documents = defaultdict(list)
        self.group_indices = defaultdict(list)
        
        for idx, (doc, target) in enumerate(zip(self.documents, self.targets)):
            group_name = self.group_names[target]
            self.group_documents[group_name].append(doc)
            self.group_indices[group_name].append(idx)
        
        print(f"Wczytano {len(self.documents)} dokumentów w {len(self.group_names)} grupach")
        print(f"Grupy: {', '.join(self.group_names)}")
        
        return self
    
    def vectorize(self):
        """
        Wektoryzuje dokumenty.
        """
        print(f"\nWektoryzacja dokumentów (typ: {self.vectorizer_type})...")
        
        if self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=self.max_features
            )
            self.documents_matrix = self.vectorizer.fit_transform(self.documents)
        
        elif self.vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                stop_words='english',
                max_features=self.max_features
            )
            self.documents_matrix = self.vectorizer.fit_transform(self.documents)
        
        elif self.vectorizer_type == 'tfidf_from_count':
            count_vectorizer = CountVectorizer(
                stop_words='english',
                max_features=self.max_features
            )
            X_count = count_vectorizer.fit_transform(self.documents)
            tfidf_transformer = TfidfTransformer()
            self.documents_matrix = tfidf_transformer.fit_transform(X_count)
            self.vectorizer = count_vectorizer
        
        print(f"Macierz dokumentów: {self.documents_matrix.shape}")
        return self
    
    def compute_group_similarity(
        self,
        group1: str,
        group2: str,
        sample_size: int = None
    ) -> float:
        """
        Oblicza średnie podobieństwo kosinusowe między dwiema grupami.
        
        Parameters:
        -----------
        group1 : str
            Nazwa pierwszej grupy
        group2 : str
            Nazwa drugiej grupy
        sample_size : int, optional
            Jeśli podane, losuje próbkę dokumentów (dla szybkości)
        
        Returns:
        --------
        float
            Średnie podobieństwo kosinusowe
        """
        indices1 = self.group_indices[group1]
        indices2 = self.group_indices[group2]
        
        # Losuj próbkę jeśli podano sample_size
        if sample_size:
            np.random.seed(42)  # Dla powtarzalności wyników
            if len(indices1) > sample_size:
                indices1 = np.random.choice(indices1, sample_size, replace=False)
            if len(indices2) > sample_size:
                indices2 = np.random.choice(indices2, sample_size, replace=False)
        
        # Pobierz wektory dokumentów
        vectors1 = self.documents_matrix[indices1]
        vectors2 = self.documents_matrix[indices2]
        
        # Oblicz podobieństwo dla wszystkich par
        if self.similarity_metric == 'cosine':
            similarity_matrix = cosine_similarity(vectors1, vectors2)
        elif self.similarity_metric == 'dot':
            similarity_matrix = (vectors1 @ vectors2.T).toarray()
        else:
            raise ValueError(f"Nieznana metryka: {self.similarity_metric}")
        
        # Średnie podobieństwo
        mean_similarity = float(np.mean(similarity_matrix))
        
        return mean_similarity
    
    def find_most_similar_groups(self, sample_size: int = None) -> Dict[str, Tuple[str, str, float, float]]:
        """
        Znajduje najbardziej i najmniej podobne grupy dla każdej grupy.
        
        Parameters:
        -----------
        sample_size : int, optional
            Liczba dokumentów do próbkowania (dla szybkości)
        
        Returns:
        --------
        dict
            Słownik: {grupa: (najbardziej_podobna, najmniej_podobna, sim_max, sim_min)}
        """
        print(f"\nObliczanie podobieństwa między grupami...")
        if sample_size:
            print(f"(używam próbki {sample_size} dokumentów na grupę dla szybkości)")
        
        results = {}
        n_groups = len(self.group_names)
        
        # Oblicz podobieństwa dla wszystkich par grup
        similarity_matrix = np.zeros((n_groups, n_groups))
        
        for i, group1 in enumerate(self.group_names):
            for j, group2 in enumerate(self.group_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0  # Grupa z samą sobą
                else:
                    sim = self.compute_group_similarity(group1, group2, sample_size)
                    similarity_matrix[i, j] = sim
                    print(f"  {group1} <-> {group2}: {sim:.4f}")
        
        # Dla każdej grupy znajdź najbardziej i najmniej podobną
        for i, group in enumerate(self.group_names):
            # Wyklucz samą siebie (wartość 1.0)
            similarities = similarity_matrix[i, :].copy()
            similarities[i] = -1  # Ustaw na -1 żeby wykluczyć z max
            
            # Najbardziej podobna (najwyższa wartość oprócz siebie)
            most_similar_idx = np.argmax(similarities)
            most_similar = self.group_names[most_similar_idx]
            sim_max = float(similarity_matrix[i, most_similar_idx])
            
            # Najmniej podobna (najniższa wartość)
            similarities[i] = 2  # Ustaw na 2 żeby wykluczyć z min
            least_similar_idx = np.argmin(similarities)
            least_similar = self.group_names[least_similar_idx]
            sim_min = float(similarity_matrix[i, least_similar_idx])
            
            results[group] = (most_similar, least_similar, sim_max, sim_min)
        
        return results, similarity_matrix
    
    def display_results(
        self,
        results: Dict[str, Tuple[str, str, float, float]],
        similarity_matrix: np.ndarray
    ):
        """
        Wyświetla wyniki analizy.
        """
        print("\n" + "=" * 80)
        print("WYNIKI ANALIZY PODOBIEŃSTWA GRUP")
        print("=" * 80)
        
        print(f"\nMetryka podobieństwa: {self.similarity_metric}")
        print(f"Typ wektoryzacji: {self.vectorizer_type}")
        
        print("\n" + "-" * 80)
        print(f"{'Grupa':<30} {'Najbardziej podobna':<30} {'Najmniej podobna':<30}")
        print("-" * 80)
        
        for group, (most_sim, least_sim, sim_max, sim_min) in results.items():
            print(f"{group:<30} {most_sim:<30} ({sim_max:.4f})  {least_sim:<30} ({sim_min:.4f})")
        
        # Macierz podobieństwa
        print("\n" + "=" * 80)
        print("MACIERZ PODOBIEŃSTWA (średnie podobieństwo kosinusowe)")
        print("=" * 80)
        print(f"\n{'Grupa':<20}", end="")
        for name in self.group_names:
            print(f"{name[:15]:<16}", end="")
        print()
        print("-" * (20 + 16 * len(self.group_names)))
        
        for i, group in enumerate(self.group_names):
            print(f"{group[:19]:<20}", end="")
            for j in range(len(self.group_names)):
                print(f"{similarity_matrix[i, j]:>15.4f}", end="")
            print()


def run_task4_experiments(
    use_task2_variants: bool = False,
    sample_size: int = 100
):
    """
    Uruchamia wszystkie eksperymenty z zadania 4.
    
    Parameters:
    -----------
    use_task2_variants : bool
        Czy wykonać podpunkty z zadania 2 (różne metryki i wektoryzacje)
    sample_size : int
        Liczba dokumentów do próbkowania (None = wszystkie)
    """
    print("=" * 80)
    print("ZADANIE 4: ANALIZA PODOBIEŃSTWA GRUP W 20NEWSGROUPS")
    print("=" * 80)
    
    # Podstawowa analiza z TF-IDF i podobieństwem kosinusowym
    print("\n" + "=" * 80)
    print("PODSTAWOWA ANALIZA: TF-IDF + Podobieństwo kosinusowe")
    print("=" * 80)
    
    analyzer = GroupSimilarityAnalyzer(
        vectorizer_type='tfidf',
        similarity_metric='cosine'
    )
    analyzer.load_data(subset='train')
    analyzer.vectorize()
    results, sim_matrix = analyzer.find_most_similar_groups(sample_size=sample_size)
    analyzer.display_results(results, sim_matrix)
    
    # Podpunkty z zadania 2 (opcjonalnie)
    if use_task2_variants:
        # B) Iloczyn skalarny
        print("\n" + "=" * 80)
        print("PODPUNKT B: TF-IDF + Iloczyn skalarny")
        print("=" * 80)
        
        analyzer_dot = GroupSimilarityAnalyzer(
            vectorizer_type='tfidf',
            similarity_metric='dot'
        )
        analyzer_dot.load_data(subset='train')
        analyzer_dot.vectorize()
        results_dot, sim_matrix_dot = analyzer_dot.find_most_similar_groups(sample_size=sample_size)
        analyzer_dot.display_results(results_dot, sim_matrix_dot)
        
        # C) Różne metody wektoryzacji
        for vec_type, vec_name in [
            ('count', 'TF (CountVectorizer)'),
            ('tfidf_from_count', 'TF-IDF (z CountVectorizer)')
        ]:
            print("\n" + "=" * 80)
            print(f"PODPUNKT C: {vec_name} + Podobieństwo kosinusowe")
            print("=" * 80)
            
            analyzer_var = GroupSimilarityAnalyzer(
                vectorizer_type=vec_type,
                similarity_metric='cosine'
            )
            analyzer_var.load_data(subset='train')
            analyzer_var.vectorize()
            results_var, sim_matrix_var = analyzer_var.find_most_similar_groups(sample_size=sample_size)
            analyzer_var.display_results(results_var, sim_matrix_var)
    
    print("\n" + "=" * 80)
    print("ZAKOŃCZONO")
    print("=" * 80)


if __name__ == "__main__":
    # Uruchom podstawową analizę
    # sample_size=100 oznacza że użyjemy 100 losowych dokumentów z każdej grupy
    # (dla szybkości - ustaw None aby użyć wszystkich)
    
    run_task4_experiments(
        use_task2_variants=True,  # Ustaw True aby wykonać podpunkty z zadania 2
        sample_size=100  # Użyj None dla pełnej analizy (może być wolne)
    )
