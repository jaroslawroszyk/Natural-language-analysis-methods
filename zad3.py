"""
Zadanie 3: Klasyfikacja sentymentu recenzji IMDB

Na wykładzie prezentowano wykorzystanie klasyfikatora Bayesa na zbiorze recenzji z IMDB.
Wybierz inny klasyfikator i oceń jego dokładność.

(*) Wykorzystaj także słownik VADER do oceny wydźwięku recenzji.
W sprawozdaniu opisz skrótowo sposób wykorzystania słownika i porównaj osiągnięty efekt
(dokładność) z dokładnością klasyfikatorów.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    print("Ostrzeżenie: vaderSentiment nie jest zainstalowane.")
    print("Zainstaluj: pip install vaderSentiment")
    print("VADER będzie pominięty w analizie.")


def load_imdb_data(data_path=None):
    """
    Wczytuje dane IMDB z folderu.
    
    Parameters:
    -----------
    data_path : str, optional
        Ścieżka do folderu z danymi IMDB.
        Struktura: data_path/aclImdb/train/pos/ i data_path/aclImdb/train/neg/
        
    Returns:
    --------
    reviews : list
        Lista recenzji
    labels : array
        Etykiety (1 = pozytywna, 0 = negatywna)
    """
    import os
    
    if data_path is None:
        # Próba automatycznego znalezienia
        possible_paths = [
            './aclImdb',
            './data/aclImdb',
            '../aclImdb',
            '~/Downloads/aclImdb'
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                data_path = expanded_path
                break
    
    if data_path and os.path.exists(data_path):
        try:
            from sklearn.datasets import load_files
            
            # Wczytaj dane treningowe
            train_path = os.path.join(data_path, 'train')
            if os.path.exists(train_path):
                data = load_files(train_path, encoding='utf-8')
                # Konwertuj etykiety: 'pos' -> 1, 'neg' -> 0
                labels = np.array([1 if 'pos' in folder else 0 
                                  for folder in data.filenames])
                return data.data, labels
        except Exception as e:
            print(f"Błąd wczytywania danych z {data_path}: {e}")
    
    print("\n" + "=" * 80)
    print("INSTRUKCJA: Jak wczytać prawdziwy zbiór IMDB")
    print("=" * 80)
    print("1. Pobierz zbiór danych IMDB:")
    print("   https://ai.stanford.edu/~amaas/data/sentiment/")
    print("2. Rozpakuj archiwum (np. aclImdb_v1.tar.gz)")
    print("3. Użyj funkcji load_imdb_data('ścieżka/do/aclImdb')")
    print("   lub umieść folder 'aclImdb' w katalogu roboczym")
    print("=" * 80 + "\n")
    
    return None, None


def create_sample_data():
    """
    Tworzy przykładowe dane do testowania (jeśli nie ma dostępu do pełnego zbioru IMDB).
    W rzeczywistym użyciu zastąp to wczytaniem prawdziwych danych.
    """
    # Przykładowe recenzje pozytywne
    positive_reviews = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Amazing film with great acting and an excellent storyline.",
        "One of the best movies I've ever seen. Highly recommended!",
        "Brilliant cinematography and outstanding performances by all actors.",
        "A masterpiece! The plot is engaging and the characters are well-developed.",
        "I was completely captivated from start to finish. Wonderful movie!",
        "Excellent direction and superb acting. A must-watch!",
        "This film exceeded all my expectations. Truly remarkable!",
        "Outstanding movie with incredible special effects and great story.",
        "I thoroughly enjoyed this film. It's a perfect blend of action and drama.",
    ]
    
    # Przykładowe recenzje negatywne
    negative_reviews = [
        "This movie is terrible. I wasted my time watching it.",
        "Poor acting and a boring plot. Not worth watching.",
        "One of the worst films I've ever seen. Completely disappointing.",
        "The storyline is confusing and the characters are poorly developed.",
        "A complete waste of time. I don't recommend this movie at all.",
        "Terrible cinematography and awful performances by the actors.",
        "This film was a huge disappointment. Very poorly made.",
        "Boring and predictable. I couldn't wait for it to end.",
        "The worst movie I've seen this year. Save your money.",
        "Completely uninteresting and poorly executed. Not recommended.",
    ]
    
    reviews = positive_reviews + negative_reviews
    labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)
    
    return reviews, np.array(labels)


def preprocess_text(text):
    """
    Podstawowe przetwarzanie tekstu.
    """
    # Usuń HTML tags jeśli są
    text = re.sub(r'<[^>]+>', '', text)
    # Usuń znaki specjalne (opcjonalnie)
    # text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()


class SentimentClassifier:
    """
    Klasa do klasyfikacji sentymentu używająca różnych klasyfikatorów.
    """
    
    def __init__(self, classifier_type='logistic_regression'):
        """
        Inicjalizacja klasyfikatora.
        
        Parameters:
        -----------
        classifier_type : str
            Typ klasyfikatora: 'logistic_regression', 'svm', 'naive_bayes'
        """
        self.classifier_type = classifier_type
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),  # unigramy i bigramy
            min_df=2,
            max_df=0.95
        )
        
        if classifier_type == 'logistic_regression':
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0
            )
        elif classifier_type == 'svm':
            self.classifier = SVC(
                kernel='linear',
                random_state=42,
                C=1.0
            )
        elif classifier_type == 'naive_bayes':
            self.classifier = MultinomialNB(alpha=1.0)
        else:
            raise ValueError(f"Nieznany typ klasyfikatora: {classifier_type}")
        
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
    
    def train(self, X_train, y_train):
        """Trenuje klasyfikator."""
        self.pipeline.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Przewiduje etykiety dla danych testowych."""
        return self.pipeline.predict(X_test)
    
    def predict_proba(self, X_test):
        """Zwraca prawdopodobieństwa przewidywań."""
        return self.pipeline.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        """Ocenia dokładność klasyfikatora."""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'=' * 80}")
        print(f"Klasyfikator: {self.classifier_type.upper()}")
        print(f"{'=' * 80}")
        print(f"Dokładność (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nRaport klasyfikacji:")
        print(classification_report(y_test, y_pred, target_names=['Negatywny', 'Pozytywny']))
        print(f"\nMacierz pomyłek:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy, y_pred


class VADERSentimentAnalyzer:
    """
    Klasa do analizy sentymentu używająca VADER.
    """
    
    def __init__(self):
        """Inicjalizacja analizatora VADER."""
        if not HAS_VADER:
            raise ImportError("vaderSentiment nie jest zainstalowane. Zainstaluj: pip install vaderSentiment")
        
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text):
        """
        Analizuje sentyment pojedynczego tekstu.
        
        Returns:
        --------
        dict : Słownik z wynikami (compound, pos, neu, neg)
        """
        return self.analyzer.polarity_scores(text)
    
    def classify(self, text, threshold=0.0):
        """
        Klasyfikuje tekst jako pozytywny (1) lub negatywny (0).
        
        Parameters:
        -----------
        text : str
            Tekst do klasyfikacji
        threshold : float
            Próg dla klasyfikacji (domyślnie 0.0)
            compound >= threshold -> pozytywny (1)
            compound < threshold -> negatywny (0)
        
        Returns:
        --------
        int : 1 dla pozytywnego, 0 dla negatywnego
        """
        scores = self.analyze(text)
        return 1 if scores['compound'] >= threshold else 0
    
    def evaluate(self, X_test, y_test, threshold=0.0):
        """
        Ocenia dokładność VADER na zbiorze testowym.
        
        Parameters:
        -----------
        X_test : list
            Lista tekstów do klasyfikacji
        y_test : array
            Prawdziwe etykiety
        threshold : float
            Próg klasyfikacji
        
        Returns:
        --------
        float : Dokładność
        """
        predictions = [self.classify(text, threshold) for text in X_test]
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\n{'=' * 80}")
        print(f"VADER Sentiment Analyzer")
        print(f"{'=' * 80}")
        print(f"Próg klasyfikacji: {threshold}")
        print(f"Dokładność (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nRaport klasyfikacji:")
        print(classification_report(y_test, predictions, target_names=['Negatywny', 'Pozytywny']))
        print(f"\nMacierz pomyłek:")
        print(confusion_matrix(y_test, predictions))
        
        return accuracy, predictions


def compare_all_methods(X_train, X_test, y_train, y_test):
    """
    Porównuje wszystkie metody klasyfikacji.
    
    Parameters:
    -----------
    X_train, X_test : list
        Dane treningowe i testowe
    y_train, y_test : array
        Etykiety treningowe i testowe
    """
    results = {}
    
    # 1. Logistic Regression
    print("\n" + "=" * 80)
    print("TRENOWANIE I OCENA KLASYFIKATORÓW")
    print("=" * 80)
    
    lr_classifier = SentimentClassifier('logistic_regression')
    lr_classifier.train(X_train, y_train)
    lr_accuracy, lr_pred = lr_classifier.evaluate(X_test, y_test)
    results['Logistic Regression'] = lr_accuracy
    
    # 2. SVM
    svm_classifier = SentimentClassifier('svm')
    svm_classifier.train(X_train, y_train)
    svm_accuracy, svm_pred = svm_classifier.evaluate(X_test, y_test)
    results['SVM'] = svm_accuracy
    
    # 3. Naive Bayes (dla porównania)
    nb_classifier = SentimentClassifier('naive_bayes')
    nb_classifier.train(X_train, y_train)
    nb_accuracy, nb_pred = nb_classifier.evaluate(X_test, y_test)
    results['Naive Bayes'] = nb_accuracy
    
    # 4. VADER
    if HAS_VADER:
        print("\n" + "=" * 80)
        print("ANALIZA VADER")
        print("=" * 80)
        
        vader_analyzer = VADERSentimentAnalyzer()
        
        # Testuj różne progi
        best_threshold = 0.0
        best_accuracy = 0.0
        
        print("\nTestowanie różnych progów klasyfikacji VADER:")
        for threshold in [-0.5, -0.2, 0.0, 0.2, 0.5]:
            accuracy, _ = vader_analyzer.evaluate(X_test, y_test, threshold=threshold)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
            print(f"  Próg {threshold:+.1f}: {accuracy:.4f}")
        
        print(f"\nNajlepszy próg: {best_threshold:.1f} (dokładność: {best_accuracy:.4f})")
        results['VADER'] = best_accuracy
    
    # Podsumowanie
    print("\n" + "=" * 80)
    print("PODSUMOWANIE - PORÓWNANIE DOKŁADNOŚCI")
    print("=" * 80)
    print(f"{'Metoda':<25} {'Dokładność':<15} {'%':<10}")
    print("-" * 50)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for method, accuracy in sorted_results:
        print(f"{method:<25} {accuracy:.4f}        {accuracy*100:>6.2f}%")
    
    print("\n" + "=" * 80)
    print("WNIOSKI")
    print("=" * 80)
    best_method = sorted_results[0][0]
    print(f"Najlepsza metoda: {best_method} ({sorted_results[0][1]:.4f})")
    
    if HAS_VADER and 'VADER' in results:
        vader_rank = [i for i, (m, _) in enumerate(sorted_results) if m == 'VADER'][0] + 1
        print(f"VADER zajmuje {vader_rank}. miejsce na {len(sorted_results)} metod.")
        print("\nUwagi o VADER:")
        print("- VADER jest słownikiem opartym na regułach, nie wymaga treningu")
        print("- Działa dobrze na krótkich tekstach (np. recenzje, tweety)")
        print("- Może być mniej dokładny na długich dokumentach")
        print("- Nie wymaga danych treningowych, działa out-of-the-box")


def main():
    """
    Główna funkcja uruchamiająca wszystkie eksperymenty.
    """
    print("=" * 80)
    print("ZADANIE 3: KLASYFIKACJA SENTYMENTU RECENZJI IMDB")
    print("=" * 80)
    
    # Wczytaj dane
    print("\nWczytywanie danych...")
    
    # Próba wczytania prawdziwego zbioru IMDB
    X, y = load_imdb_data()
    
    # Jeśli nie udało się wczytać, użyj przykładowych danych
    if X is None or y is None:
        print("\nUżywam przykładowych danych do demonstracji.")
        print("Aby użyć prawdziwego zbioru IMDB, pobierz go i użyj load_imdb_data('ścieżka').\n")
        X, y = create_sample_data()
    
    # Przetwarzanie wstępne
    X = [preprocess_text(text) for text in X]
    
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Liczba przykładów treningowych: {len(X_train)}")
    print(f"Liczba przykładów testowych: {len(X_test)}")
    print(f"Rozkład klas (trening): {np.bincount(y_train)}")
    print(f"Rozkład klas (test): {np.bincount(y_test)}")
    
    # Porównaj wszystkie metody
    compare_all_methods(X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 80)
    print("ZAKOŃCZONO")
    print("=" * 80)


if __name__ == "__main__":
    main()
