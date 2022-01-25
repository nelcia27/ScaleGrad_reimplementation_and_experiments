# ScaleGrad reimplementation and experiments

### Wymagania techniczne: 
   - pytorch
   - transformers
   - datasets <br>

### Przeprowadzone eksperymenty: 
   - open_ended_generation: <br>     
      - zbiór danych: wikitext <br>     
      - model: distilgpt2 <br>     
      - eksperymenty: finetuning modelu przy wykorzystaniu funkcji staty zaproponowanej przez autorów artykułu - ScalGrad (dla porównania - miara MLE) <br> 
      - raportowane miary: perplexity <br>
 
   - directed_generation - text_summarization: <br>     
      - zbiór danych: cnn_dailymail <br>     
      - model: mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization <br>
      - eksperymenty: finetuning modelu przy wykorzystaniu funkcji staty zaproponowanej przez autorów artykułu - ScalGrad (dla porównania - miara MLE) <br>     
      - raportowane miary: R-1, R-2, R-L <br>

### Struktura repozytorium: 
1. original_experiment - kod jedynego udostępnionego przez autorów eksperymentu (minimalnie zmieniona struktura)

2. reimplementation_experiments:
   - open_ended_generation: <br>
         - Open_ended_genetarion_mle.ipynb: trening modelu z wykorzystaniem MLE (sg_loss, ale gamma=1.0), zebranie wartości miar oraz przezentacja wyników <br>
         - Open_ended_genetarion_scalegrad.ipynb: trening modelu z wykorzystaniem sg_loss (gamma=0.2), zebranie wartości miar oraz przezentacja wyników <br> 
   - directed_generation: <br>
         - Directed_generation_text_summarization_mle.ipynb: trening modelu z wykorzystaniem MLE (sg_loss, ale gamma=1.0), zebranie wartości miar oraz przezentacja wyników <br>
         - Directed_generation_text_summarization_scalegrad.ipynb: trening modelu z wykorzystaniem sg_loss (gamma=0.8), zebranie wartości miar oraz przezentacja wyników <br>


##### Kluczowym elementem w każdym z notebooków jest sekcja Reimplementacja funkcji straty ScaleGrad
 
 ### Wnioski:
 Po przeprowadzonych ekeperymentach różnica między modelami finetuningowanymi z wykorzystaniem sg_loss i MLE nie jest widoczna. W ramach projektu nie odtworzyłyśmy jednak dokładnie eksperymentów przeprowadzonych przez autorów ze względu na brak niektórych istotnych informacji jak:
 - liczba epok, 
 - konkretna wersja modelu 
 - konkretna wersja zbioru danych

Największą trudnością, i zarazem ograniczeniem, podczas przeprowadzania badań, okazał się dostęp do wystarczających zasobów obliczeniowych. Wpłynęło to kluczowo na otrzymane wyniki. 

Dodatkowo autorzy dostarczyli kod jedynie dla eksperymentu dotyczącego *open ended generation*, więc pozostałe warianty zostały przeprowadzone na podstawie ogólnych informacji zamieszczonych w artykule oraz suplemencie. Były one dość ubogie w szczegóły dotyczące wyboru modeli, czy parametrów treningu. 

Ze względu na wspomniane braki informacji, porównywanie uzyskanych przez nas wyników z wynikami otrzymanymi w artykule nie jest wskazane. Nasze badania wykazały brak wpływu wartości parametru *gamma* na wyniki. Zebrane miary i teksty otrzymane z wykorzystaniem dotrenowanych modeli są podobne niezależenie od badanej metody. Może to wynikać z bardzo wąskiego zakresu eksperymentu - treningi trwające jedną epokę zapewne przyczyniły się do takiego rezultatu.
