"""
run_dashboard.py — One-Click Launcher for OmniGuard-RAG Studio Dashboard

Starts the lightweight, multi-threaded REST API dashboard server and opens
the interactive web studio in the default browser.
"""
import argparse
import sys
import os
import time
import webbrowser
import threading

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.dashboard_server import run_server


def print_banner(host: str, port: int):
    url = f"http://{host}:{port}"
    print("=" * 72)
    print(" 🛡️   OmniGuard-RAG Interactive Studio — Web Dashboard & Local LLM")
    print("=" * 72)
    print(f" • Dashboard UI:      {url}")
    print(f" • REST API Status:   {url}/api/status")
    print(f" • Topics & Corpus:   {url}/api/topics (16 Topics, 480 Clean Documents)")
    print(" • Defense Engine:    OmniGuard-RAG 4-Ring Architecture + 5 Baselines")
    print(" • Local LLMs:        Ollama (11434), LM Studio (1234), Built-in Fallback")
    print(" • Attack Playground: Clean, Standard, PIDP, Collusion, Stealth, Custom")
    print("=" * 72)
    print(" Press Ctrl+C in terminal to stop the server.\n")


def open_browser_delayed(url: str, delay: float = 1.0):
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="Launch OmniGuard-RAG Interactive Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    args = parser.parse_args()

    server = run_server(host=args.host, port=args.port)
    print_banner(args.host, args.port)

    if not args.no_browser:
        open_browser_delayed(f"http://{args.host}:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[!] Shutting down OmniGuard-RAG Dashboard Server...")
        server.server_close()
        print("[✓] Server stopped cleanly.")


if __name__ == "__main__":
    main()
