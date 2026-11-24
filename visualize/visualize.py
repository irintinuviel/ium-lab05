"""Usługa visualize: czeka na usługę train po nazwie, potem serwuje wyniki."""
import json
import os
import time
import urllib.request
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

TRAIN_URL = os.getenv("TRAIN_URL", "http://train:5000/status")
RESULTS_DIR = os.getenv("RESULTS_DIR", "/results")
PORT = int(os.getenv("PORT", 8080))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 60))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", 2))


def wait_for_train() -> dict:
    """Odpytuje usługę train po nazwie, aż zwróci status 'ready'."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(TRAIN_URL, timeout=3) as resp:
                status = json.loads(resp.read().decode("utf-8"))
            if status.get("status") == "ready":
                print(f"[visualize] Połączono z usługą train ({TRAIN_URL}) "
                      f"w próbie {attempt}: {status}", flush=True)
                return status
        except Exception as exc:  # usługa jeszcze nie odpowiada
            print(f"[visualize] Próba {attempt}/{MAX_RETRIES}: train niedostępny "
                  f"({type(exc).__name__}), czekam {RETRY_DELAY} s...", flush=True)
        time.sleep(RETRY_DELAY)
    raise SystemExit(f"[visualize] Usługa train nie odpowiedziała: {TRAIN_URL}")


def make_handler(status: dict):
    """Strona główna z metrykami + pliki z katalogu wyników."""

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                return super().do_GET()        # np. /wyniki.txt, /metrics.json
            html = (
                "<!DOCTYPE html><html lang='pl'><head><meta charset='utf-8'>"
                "<title>Wyniki treningu</title></head><body>"
                "<h1>Wyniki treningu (usługa visualize)</h1>"
                f"<p>Model: <b>{status.get('model_type')}</b></p>"
                f"<p>Accuracy: <b>{status.get('accuracy')}</b> "
                f"na {status.get('n_test')} próbkach testowych</p>"
                f"<p>Dane pobrane z usługi train: <code>{TRAIN_URL}</code></p>"
                "<p><a href='/wyniki.txt'>wyniki.txt</a> | "
                "<a href='/metrics.json'>metrics.json</a></p>"
                "</body></html>"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

    return partial(Handler, directory=RESULTS_DIR)


def main() -> None:
    print(f"[visualize] Start, sprawdzam usługę train pod {TRAIN_URL}", flush=True)
    status = wait_for_train()
    print(f"[visualize] Panel dostępny na porcie {PORT}", flush=True)
    HTTPServer(("0.0.0.0", PORT), make_handler(status)).serve_forever()


if __name__ == "__main__":
    main()
