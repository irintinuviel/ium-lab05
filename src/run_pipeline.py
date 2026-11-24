"""Uruchamia cały pipeline: data_prep -> train -> evaluate.

Jeśli ustawiona jest zmienna STATUS_PORT, po zakończeniu kontener działa dalej
i udostępnia endpoint /status z metrykami (dla usługi visualize).
"""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import data_prep
import evaluate
import train

STATUS_PORT = os.getenv("STATUS_PORT")


def serve_status(port: int, metrics: dict) -> None:
    """Prosty serwer HTTP: GET /status zwraca stan pipeline'u i metryki."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/status":
                self.send_error(404)
                return
            body = json.dumps({"status": "ready", **metrics}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            print(f"[status] {self.client_address[0]} -> {self.command} {self.path}", flush=True)

    print(f"[status] Endpoint /status dostępny na porcie {port}", flush=True)
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


def main() -> None:
    print("=== Etap 1/3: przygotowanie danych ===", flush=True)
    data_prep.main()
    print("=== Etap 2/3: trening ===", flush=True)
    train.main()
    print("=== Etap 3/3: ewaluacja ===", flush=True)
    metrics = evaluate.main()
    print("=== Pipeline zakończony ===", flush=True)

    if STATUS_PORT:
        serve_status(int(STATUS_PORT), metrics)


if __name__ == "__main__":
    main()
