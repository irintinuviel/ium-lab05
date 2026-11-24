!\[CI](https://github.com/irintinuviel/ium-lab05/actions/workflows/ci.yml/badge.svg)

Pipeline klasyfikacji wine 

## Etapy

1. przygotowanie danych
2. trening modelu
3. ewaluacja i zapis metryk

## Uruchomienie

```bash
docker compose up --build      # panel z wynikami na http://localhost:8080
pytest -v                      
```

