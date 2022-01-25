# ScaleGrad reimplementation and experiments

Wymagania techniczne: pytorch, transformers, datasets

Przeprowadzone eksperymenty: 
   <br> - open_ended_generation:
       <br>     - zbiór danych: wikitext
       <br>     - model: distilgpt2
       <br>     - eksperymenty polegały na finetuningu modelu na zbiorze danych przy wykorzystaniu funkcji staty zaproponowanej przez autorów artykułu - ScalGrad oraz, dla porównania przy wykorzystaniu MLE
       <br> - raportowane miary to perplexity
   <br> - directed_generation: text_summarization 
       <br>     - zbiór danych: cnn_dailymail
       <br>     - model: mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization
       <br>     - eksperymenty polegały na finetuningu modelu na zbiorze danych przy wykorzystaniu funkcji staty zaproponowanej przez autorów artykułu - ScalGrad oraz, dla porównania przy wykorzystaniu MLE
       <br>     - raportowane miary to R-1, R-2, R-L

### Struktura repozytorium: 
1. original_experiment - kod jedynego udostępnionego przez autorów eksperymentu (minimalnie zmieniona struktura)

2. reimplementation_experiments:
- open_ended_generation:
    - Open_ended_genetarion_mle.ipynb: trening modelu z wykorzystaniem MLE (sg_loss, ale gamma=1.0), zebranie wartości miar oraz przezentacja wyników
    - Open_ended_genetarion_scalegrad.ipynb: trening modelu z wykorzystaniem sg_loss (gamma=0.2), zebranie wartości miar oraz przezentacja wyników
- directed_generation: 
    - Directed_generation_text_summarization_mle.ipynb: trening modelu z wykorzystaniem MLE (sg_loss, ale gamma=1.0), zebranie wartości miar oraz przezentacja wyników
    - Directed_generation_text_summarization_scalegrad.ipynb: trening modelu z wykorzystaniem sg_loss (gamma=0.8), zebranie wartości miar oraz przezentacja wyników


<br> Kluczowym elementem w każdym z notebooków jest sekcja Reimplementacja funkcji straty ScaleGrad
 
 ### Wnioski:
 Ostatecznie nie jest obserwowana różnica między modelami finetuningowanymi z wykorzystaniem sg_loss i MLE. Jednak w ramach projektu nie odtworzyłyśmy dokładnie eksperymentów przeprowadzonych przez autorów ze względu na brak niektórych istotnych informacji jak liczba epok, konkretna wersja modelu czy zbioru danych. Największą trudnością i zarazem ograniczeniem okazał się dostęp do wystarczających zasobów obliczeniowych, co zapewne ostatecznie wpłynęło na otrzymane wyniki. 
 Co istotne autorzy dostarczyli kod jedynie dla eksperymentu dotyczącego open ended generation, dla pozostałych bazowałyśmy na informacjach zamieszczonych w artykule i suplemencie -  jednak były one dość ubogie w szczegóły dotyczące wyboru modeli czy parametrów treningu. Dodatkowo, porównanie uzyskanych przez nas wyników i wyników otrzymanych w artykule nie jest wskazane, ze względu na wspomniane braki informacji. Porównanie zbieranych miar i tekstów otrzymanych z wykorzystaniem dotrenowanych modeli w naszym przypadku wskazuje na brak wpływu wartości parametru gamma na wyniki, jednak treningi trwające jedną epokę zapewne przyczyniły się do takiego rezultatu.
