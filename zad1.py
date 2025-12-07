# **Zadanie 1. (*)** Napisz klasę lub zbiór funkcji o opisanej poniżej funkcjonalności. Kod ma umożliwiać tworzenie wektorów Bow i TF-IDF (tak, jak robią to klasy CountVectorizer i TfidfVectorizer, których nie można używać w tym zadaniu). Kod powinien zawierać:

# • odpowiednik metody fit dla zadanego korpusu (który podajemy w postaci listy łańcuchów znakowych). Podczas wykonywania metody fit powinny być utworzone dane konieczne do kolejnego podpunktu, np. wspólny dla całego korpusu słownik tokenów.

# • odpowiednik metody transform, która dla zadanego (w postaci listy łańcuchów znakowych) zbioru dokumentów utworzy wektory BoW lub TF-IDF. Wektory powinny być zwracane w postaci tablicy Numpy lub ramki danych Pandas, gdzie wiersze odpowiadają dokumentom, a kolumny tokenom ze słownika. Należy też rozwiązać problem występowania w danych wejściowych tokenów nieobecnych w korpusie przekazanym do metody fit, ale obecnych w danych przekazywanych do metody transform.

# Dodatkowo można zaimplementować:

#     • wybór tokenizera,

#     • wykorzystanie stoplisty,

#     • różne sposoby wyliczania BoW (wektor lub liczba wystąpień), TF, IDF,

#     • stemming,
    
#     • itp.

# ***W sprawozdaniu przedstaw przykłady wykorzystania stworzonego kodu***

import re
import numpy as np
from collections import Counter, defaultdict
from typing import List, Union, Optional, Callable
import math

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class CustomVectorizer:
    """
    Klasa do tworzenia wektorów BoW i TF-IDF z dokumentów tekstowych.
    """
    
    def __init__(
        self,
        vector_type: str = 'bow',  # 'bow' lub 'tfidf'
        tokenizer: Optional[Callable] = None,
        lowercase: bool = True,
        stop_words: Optional[List[str]] = None,
        use_stemming: bool = False,
        min_df: int = 1,  # minimalna liczba dokumentów zawierających token
        max_df: float = 1.0,  # maksymalna częstotliwość dokumentów (może być float < 1.0)
        bow_type: str = 'count',  # 'count' (liczba wystąpień) lub 'binary' (wektor binarny)
        tf_type: str = 'raw',  # 'raw', 'log', 'augmented'
        idf_type: str = 'standard',  # 'standard', 'smooth', 'probabilistic'
        return_type: str = 'numpy'  # 'numpy' lub 'pandas'
    ):
        """
        Parametry:
        ----------
        vector_type : str
            Typ wektoryzacji: 'bow' lub 'tfidf'
        tokenizer : callable, optional
            Funkcja tokenizująca. Jeśli None, używa domyślnej tokenizacji.
        lowercase : bool
            Czy konwertować tekst na małe litery
        stop_words : list, optional
            Lista słów do usunięcia (stoplist)
        use_stemming : bool
            Czy używać stemmingu (wymaga biblioteki nltk)
        min_df : int
            Minimalna liczba dokumentów zawierających token
        max_df : float
            Maksymalna częstotliwość dokumentów (0.0-1.0)
        bow_type : str
            Typ BoW: 'count' (liczba) lub 'binary' (binarny)
        tf_type : str
            Typ TF: 'raw', 'log', 'augmented'
        idf_type : str
            Typ IDF: 'standard', 'smooth', 'probabilistic'
        return_type : str
            Typ zwracanych danych: 'numpy' lub 'pandas'
        """
        self.vector_type = vector_type
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        self.stop_words = set(stop_words) if stop_words else set()
        self.use_stemming = use_stemming
        self.min_df = min_df
        self.max_df = max_df
        self.bow_type = bow_type
        self.tf_type = tf_type
        self.idf_type = idf_type
        self.return_type = return_type
        
        # Atrybuty ustawiane podczas fit
        self.vocabulary_ = {}
        self.inverse_vocabulary_ = {}
        self.idf_ = None
        self.n_docs_ = 0
        
        # Stemmer (jeśli używany)
        self.stemmer = None
        if self.use_stemming:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
            except ImportError:
                print("Ostrzeżenie: nltk nie jest zainstalowane. Stemming nie będzie działał.")
                self.use_stemming = False
    
    def _default_tokenizer(self, text: str) -> List[str]:
        """Domyślna tokenizacja - dzieli tekst na słowa."""
        # Usuwa znaki interpunkcyjne i dzieli na słowa
        tokens = re.findall(r'\b\w+\b', text.lower() if self.lowercase else text)
        return tokens
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenizuje tekst używając wybranego tokenizera."""
        if self.tokenizer:
            tokens = self.tokenizer(text)
        else:
            tokens = self._default_tokenizer(text)
        
        # Zastosuj lowercase jeśli potrzebne
        if self.lowercase and self.tokenizer:
            tokens = [t.lower() for t in tokens]
        
        # Usuń stop words
        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Zastosuj stemming
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]
        
        return tokens
    
    def _calculate_tf(self, term_count: int, doc_length: int) -> float:
        """Oblicza TF (Term Frequency) dla termu w dokumencie."""
        if doc_length == 0:
            return 0.0
        
        if self.tf_type == 'raw':
            return term_count / doc_length
        elif self.tf_type == 'log':
            return 1 + math.log(term_count) if term_count > 0 else 0
        elif self.tf_type == 'augmented':
            # Augmented TF: 0.5 + 0.5 * (term_count / max_term_count)
            # Dla uproszczenia używamy średniej częstotliwości
            return 0.5 + 0.5 * (term_count / doc_length) if doc_length > 0 else 0
        else:
            return term_count / doc_length
    
    def _calculate_idf(self, df: int, n_docs: int) -> float:
        """Oblicza IDF (Inverse Document Frequency) dla termu."""
        if df == 0:
            return 0.0
        
        if self.idf_type == 'standard':
            return math.log(n_docs / df)
        elif self.idf_type == 'smooth':
            return math.log((n_docs + 1) / (df + 1)) + 1
        elif self.idf_type == 'probabilistic':
            return math.log((n_docs - df) / df) if df < n_docs else 0
        else:
            return math.log(n_docs / df)
    
    def fit(self, corpus: List[str]) -> 'CustomVectorizer':
        """
        Uczy się słownika na podstawie korpusu.
        
        Parametry:
        ----------
        corpus : list of str
            Lista dokumentów tekstowych
            
        Zwraca:
        -------
        self : CustomVectorizer
        """
        self.n_docs_ = len(corpus)
        
        # Zbierz wszystkie tokeny z korpusu
        token_doc_counts = defaultdict(int)  # liczba dokumentów zawierających token
        
        for doc in corpus:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                token_doc_counts[token] += 1
        
        # Filtruj tokeny według min_df i max_df
        filtered_tokens = []
        max_doc_count = int(self.max_df * self.n_docs_) if self.max_df < 1.0 else self.max_df
        
        for token, doc_count in token_doc_counts.items():
            if doc_count >= self.min_df and doc_count <= max_doc_count:
                filtered_tokens.append(token)
        
        # Utwórz słownik (vocabulary)
        filtered_tokens.sort()  # Sortuj dla spójności
        self.vocabulary_ = {token: idx for idx, token in enumerate(filtered_tokens)}
        self.inverse_vocabulary_ = {idx: token for token, idx in self.vocabulary_.items()}
        
        # Oblicz IDF dla każdego termu (jeśli używamy TF-IDF)
        if self.vector_type == 'tfidf':
            self.idf_ = np.zeros(len(self.vocabulary_))
            for token, idx in self.vocabulary_.items():
                df = token_doc_counts[token]
                self.idf_[idx] = self._calculate_idf(df, self.n_docs_)
        
        return self
    
    def transform(self, documents: List[str]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Przekształca dokumenty na wektory BoW lub TF-IDF.
        
        Parametry:
        ----------
        documents : list of str
            Lista dokumentów do przekształcenia
            
        Zwraca:
        -------
        array or DataFrame
            Macierz wektorów (dokumenty x tokeny)
        """
        if not self.vocabulary_:
            raise ValueError("Musisz najpierw wywołać fit() przed transform().")
        
        n_docs = len(documents)
        n_features = len(self.vocabulary_)
        
        # Inicjalizuj macierz wyników
        matrix = np.zeros((n_docs, n_features))
        
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            
            # Policz wystąpienia tokenów w dokumencie
            token_counts = Counter(tokens)
            doc_length = len(tokens)
            
            # Dla każdego tokena w słowniku
            for token, count in token_counts.items():
                if token in self.vocabulary_:
                    token_idx = self.vocabulary_[token]
                    
                    if self.vector_type == 'bow':
                        if self.bow_type == 'count':
                            matrix[doc_idx, token_idx] = count
                        elif self.bow_type == 'binary':
                            matrix[doc_idx, token_idx] = 1
                    elif self.vector_type == 'tfidf':
                        # Oblicz TF
                        tf = self._calculate_tf(count, doc_length)
                        # Oblicz TF-IDF
                        matrix[doc_idx, token_idx] = tf * self.idf_[token_idx]
            
            # Normalizuj wektor dokumentu (L2 normalization dla TF-IDF)
            if self.vector_type == 'tfidf':
                norm = np.linalg.norm(matrix[doc_idx, :])
                if norm > 0:
                    matrix[doc_idx, :] /= norm
        
        # Zwróć w odpowiednim formacie
        if self.return_type == 'pandas':
            if not HAS_PANDAS:
                print("Ostrzeżenie: pandas nie jest zainstalowane. Zwracam numpy array.")
                return matrix
            feature_names = [self.inverse_vocabulary_[i] for i in range(n_features)]
            return pd.DataFrame(matrix, columns=feature_names)
        else:
            return matrix
    
    def fit_transform(self, corpus: List[str]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Uczy się słownika i przekształca korpus w jednym kroku.
        
        Parametry:
        ----------
        corpus : list of str
            Lista dokumentów tekstowych
            
        Zwraca:
        -------
        array or DataFrame
            Macierz wektorów (dokumenty x tokeny)
        """
        return self.fit(corpus).transform(corpus)
    
    def get_feature_names(self) -> List[str]:
        """Zwraca nazwy cech (tokeny) w kolejności słownika."""
        return [self.inverse_vocabulary_[i] for i in range(len(self.vocabulary_))]


# ============================================================================
# PRZYKŁADY WYKORZYSTANIA
# ============================================================================

if __name__ == "__main__":
    # Przykładowy korpus
    corpus = [
        "To jest pierwszy dokument o analizie języka naturalnego.",
        "Ten dokument jest drugim dokumentem.",
        "I to jest trzeci dokument.",
        "Czy to jest pierwszy dokument?",
    ]
    
    print("=" * 80)
    print("PRZYKŁAD 1: Podstawowy BoW (Bag of Words)")
    print("=" * 80)
    vectorizer_bow = CustomVectorizer(vector_type='bow', return_type='pandas')
    X_bow = vectorizer_bow.fit_transform(corpus)
    print(X_bow)
    print(f"\nSłownik: {vectorizer_bow.get_feature_names()}")
    
    print("\n" + "=" * 80)
    print("PRZYKŁAD 2: TF-IDF")
    print("=" * 80)
    vectorizer_tfidf = CustomVectorizer(vector_type='tfidf', return_type='pandas')
    X_tfidf = vectorizer_tfidf.fit_transform(corpus)
    print(X_tfidf)
    
    print("\n" + "=" * 80)
    print("PRZYKŁAD 3: BoW z stoplistą")
    print("=" * 80)
    stop_words_pl = ['to', 'jest', 'o', 'i', 'czy']
    vectorizer_stop = CustomVectorizer(
        vector_type='bow',
        stop_words=stop_words_pl,
        return_type='pandas'
    )
    X_stop = vectorizer_stop.fit_transform(corpus)
    print(X_stop)
    print(f"\nSłownik (bez stop words): {vectorizer_stop.get_feature_names()}")
    
    print("\n" + "=" * 80)
    print("PRZYKŁAD 4: TF-IDF z różnymi parametrami TF i IDF")
    print("=" * 80)
    vectorizer_custom = CustomVectorizer(
        vector_type='tfidf',
        tf_type='log',
        idf_type='smooth',
        return_type='pandas'
    )
    X_custom = vectorizer_custom.fit_transform(corpus)
    print(X_custom)
    
    print("\n" + "=" * 80)
    print("PRZYKŁAD 5: BoW binarny (binary)")
    print("=" * 80)
    vectorizer_binary = CustomVectorizer(
        vector_type='bow',
        bow_type='binary',
        return_type='pandas'
    )
    X_binary = vectorizer_binary.fit_transform(corpus)
    print(X_binary)
    
    print("\n" + "=" * 80)
    print("PRZYKŁAD 6: Transform na nowych dokumentach (z tokenami spoza słownika)")
    print("=" * 80)
    # Najpierw fit na korpusie
    vectorizer = CustomVectorizer(vector_type='bow', return_type='pandas')
    vectorizer.fit(corpus)
    
    # Nowe dokumenty z tokenami, które mogą nie być w słowniku
    new_docs = [
        "Nowy dokument z nieznanymi słowami.",
        "Kolejny dokument testowy.",
    ]
    X_new = vectorizer.transform(new_docs)
    print("Wektory dla nowych dokumentów:")
    print(X_new)
    print("\nTokeny spoza słownika są ignorowane (wartości 0).")
    
    print("\n" + "=" * 80)
    print("PRZYKŁAD 7: TF-IDF z min_df i max_df")
    print("=" * 80)
    vectorizer_filtered = CustomVectorizer(
        vector_type='tfidf',
        min_df=2,  # Token musi wystąpić w co najmniej 2 dokumentach
        max_df=0.8,  # Token może wystąpić w maksymalnie 80% dokumentów
        return_type='pandas'
    )
    X_filtered = vectorizer_filtered.fit_transform(corpus)
    print(X_filtered)
    print(f"\nSłownik (po filtracji): {vectorizer_filtered.get_feature_names()}")
    
    print("\n" + "=" * 80)
    print("PRZYKŁAD 8: Zwracanie jako NumPy array")
    print("=" * 80)
    vectorizer_numpy = CustomVectorizer(vector_type='bow', return_type='numpy')
    X_numpy = vectorizer_numpy.fit_transform(corpus)
    print(f"Typ: {type(X_numpy)}")
    print(f"Kształt: {X_numpy.shape}")
    print(X_numpy)
