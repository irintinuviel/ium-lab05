# obraz bazowy: lekki Python 3.11
FROM python:3.11-slim

WORKDIR /app

# logi Pythona od razu trafiają do docker logs (bez buforowania)
ENV PYTHONUNBUFFERED=1

# 1) najpierw same zależności - ta warstwa jest cache'owana,
#    dopóki requirements.txt się nie zmieni
COPY requirements.txt .
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -r requirements.txt

# 2) potem kod (src/, tests/, .env.example...) - zmiana kodu nie reinstaluje bibliotek
COPY . .

# domyślne polecenie: sam trening (docker-compose nadpisuje je pełnym pipeline'em)
CMD ["python", "src/train.py"]
