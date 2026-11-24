# IUM pipeline ML z CI

![CI](https://github.com/irintinuviel/ium-lab05/actions/workflows/ci.yml/badge.svg)

Pipeline klasyfikacji win

## Etapy

1. `src/data_prep.py` - przygotowanie danych
2. `src/train.py` - trening modelu
3. `src/evaluate.py` - ewaluacja i zapis metryk

## Uruchomienie

```bash
docker compose up --build      # panel z wynikami na http://localhost:8080
pytest -v                      
```