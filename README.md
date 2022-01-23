# ScaleGrad reimplementation and experiments

1. Wymagania techniczne: pytorch, transformers, datasets
2. Wykorzystane zbiory: 'wikitext', 'wikitext-2-raw-v1' (TODO: sprawdzić inne zbiory w directed_generation)
3. Wykorzystane modele: GPT2 ('distilgpt2') (TODO: wczytać inne modele w directed_generation)
4. Przeprowadzone eksperymenty: (TODO: uruchomienie) 
    - open_ended_generation 
    - directed_generation: text_summarization 

### Struktura repozytorium: 
1. original_experiment - kod jedynego udostępnionego przez autorów eksperymentu (minimalnie zmieniona struktura)

2. reimplementation_experiments:
- open_ended_generation:
    - run_gpt2.py: trening modelu i konfiguracja eksperymentu
    - scaleGrad.py: reimplementacja funkcji straty "sg_loss" (główne zadanie w projekcie)
    - Open_ended_genetarion_experiments.ipynb: trening modeli z wykorzystaniem sg_loss (gamma=0.2) oraz mle (sg_loss, ale gamma=1.0)
- directed_generation: (TODO) 
    - text_summarization 
