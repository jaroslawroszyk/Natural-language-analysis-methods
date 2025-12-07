# Natural-language-analysis-methods

## Zadanie 1. (*)

Napisz klasę lub zbiór funkcji o opisanej poniżej funkcjonalności. Kod ma umożliwiać tworzenie wektorów BoW i TF-IDF (tak, jak robią to klasy `CountVectorizer` i `TfidfVectorizer`, których nie można używać w tym zadaniu). Kod powinien zawierać:

• **Odpowiednik metody `fit`** dla zadanego korpusu (który podajemy w postaci listy łańcuchów znakowych). Podczas wykonywania metody `fit` powinny być utworzone dane konieczne do kolejnego podpunktu, np. wspólny dla całego korpusu słownik tokenów.

• **Odpowiednik metody `transform`**, która dla zadanego (w postaci listy łańcuchów znakowych) zbioru dokumentów utworzy wektory BoW lub TF-IDF. Wektory powinny być zwracane w postaci tablicy Numpy lub ramki danych Pandas, gdzie wiersze odpowiadają dokumentom, a kolumny tokenom ze słownika. Należy też rozwiązać problem występowania w danych wejściowych tokenów nieobecnych w korpusie przekazanym do metody `fit`, ale obecnych w danych przekazywanych do metody `transform`.

**Dodatkowo można zaimplementować:**

- wybór tokenizera,
- wykorzystanie stoplisty,
- różne sposoby wyliczania BoW (wektor lub liczba wystąpień), TF, IDF,
- stemming,
- itp.

> ***W sprawozdaniu przedstaw przykłady wykorzystania stworzonego kodu***

---

## Zadanie 2.

Napisz wyszukiwarkę dokumentów ze zbioru 20newsgroups lub innego dużego korpusu. Do wyszukiwarki można podać dokument (łańcuch znaków), dla którego zwróci ona zadaną liczbę dokumentów z korpusu o największym podobieństwie kosinusowym. Zwracana lista ma być posortowana w porządku nierosnącego prawdopodobieństwa.

**Następnie:**

a) sprawdź działanie wyszukiwarki na przykładzie, oceń jej skuteczność i opisz to w sprawozdaniu

b) **(*)** wypróbuj iloczyn skalarny zamiast podobieństwa kosinusowego, porównaj do wyników z punktu a) i opisz w sprawozdaniu różnicę i jej przyczyny,

c) **(*)** wypróbuj różne sposoby wyliczania BoW (wektor lub liczba wystąpień), TF, IDF i porównaj wyniki do poprzednich podpunktów. Opisz różnice i ich przyczyny w sprawozdaniu.

---

## Zadanie 3.

Na wykładzie prezentowano wykorzystanie klasyfikatora Bayesa na zbiorze recenzji z IMDB. Wybierz inny klasyfikator i oceń jego dokładność.

**(*)** Wykorzystaj także słownik VADER do oceny wydźwięku recenzji. W sprawozdaniu opisz skrótowo sposób wykorzystania słownika i porównaj osiągnięty efekt (dokładność) z dokładnością klasyfikatorów.

---

## Zadanie 4.

Dla każdej grupy ze zbioru 20newsgroups wypisz inną grupę najbardziej i najmniej podobną znaczeniowo. Jako miarę podobieństwa dwóch grup wykorzystaj średnie podobieństwo kosinusowe wyliczone dla wszystkich par dokumentów z porównywanych grup.

**(*)** Wykonaj podpunkty z zadania 2 w tym zadaniu.

---

## Zadanie 5.

Pobierz osadzenia word2vec wytrenowane dla wiadomości googla (`word2vec-google-news-300`), a następnie:

a) wybierz 5 dowolnych charakterystycznych wyrazów z różnych tematycznie grup 20newsgroups i wyświetl dla każdego z nich najbliższe wyrazy z pobranej przestrzeni osadzeń word2vec (`word2vec-google-news-300`),

b) wytrenuj osadzenia word2vec dla korpusu 20newsgroups i wyświetl najbardziej podobne do wybranych w poprzednim podpunkcie wyrazów – czy pojawiają się różnice w stosunku do poprzedniego podpunktu, a jeżeli tak, to z czego wynikają?

c) czy wybór innych niż domyślne wartości parametrów trenowania (np. inna wielkość okna) przy trenowaniu osadzeń doprowadzi do innych list zwracanych najbliższych słów?

d) dla 5 wyrazów z podpunktu a wyświetl także najbliższe wyrazy w przestrzeniach osadzeń `fasttext-wiki-news-subwords-300` oraz `glove-wiki-gigaword-300`. Które z nich są wg Ciebie najlepsze i dlaczego tak uważasz?

---

## Zadanie 6. (*)

Wczytaj dowolną przestrzeń osadzeń z Zadania 5, a następnie:

a) wykorzystaj wczytane osadzenia do wykonania punktu a) z Zadania 2 (pomocna może być metoda `get_mean_vector` obiektu typu `KeyedVectors` zwracająca wektor osadzeń uśredniony dla podanych tokenów),

b) wytrenuj osadzenia word2vec tak, jak w punkcie b) Zadania 5, a następnie wykorzystaj wczytane osadzenia do wykonania punktu a) z Zadania 2.