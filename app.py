"""
app.py
------
Main entry point for the AI Real Estate Analytics Platform.

Usage
-----
  python app.py --pipeline        # Run full training pipeline
  python app.py --api             # Start FastAPI server
  python app.py --dashboard       # Serve HTML dashboard in browser
  python app.py --all             # Pipeline + API + Dashboard
  python app.py                   # Print help
"""

import argparse
import sys
import subprocess
import webbrowser
import threading
import time
import http.server
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────
def run_pipeline():
    """Execute the full ML training pipeline."""
    print("\n🚀  Starting Training Pipeline …\n")
    from train_pipeline import main
    main()


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI backend server."""
    print(f"\n🌐  Starting API server at http://{host}:{port}")
    print(f"    Swagger docs: http://localhost:{port}/docs\n")
    try:
        import uvicorn
        uvicorn.run("src.api.api:app", host=host, port=port, reload=False)
    except ImportError:
        print("[Error] uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)


def run_dashboard(port: int = 8080):
    """Serve the HTML dashboard via a simple HTTP server."""
    dashboard_dir = ROOT / "src" / "dashboard"

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(dashboard_dir), **kwargs)
        def log_message(self, format, *args):
            pass  # suppress request logs

    url = f"http://localhost:{port}"
    print(f"\n🏠  Dashboard: {url}")

    def _open():
        time.sleep(0.8)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"    Serving at port {port}  (Ctrl+C to stop)\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Dashboard server stopped.")


# ─────────────────────────────────────────────
def print_help():
    help_text = """
╔══════════════════════════════════════════════════════════╗
║     AI Real Estate Analytics & Property Valuation        ║
║     MCA Final Year Project                               ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Commands:                                               ║
║                                                          ║
║  python app.py --pipeline     Train all ML models        ║
║  python app.py --api          Start REST API (port 8000) ║
║  python app.py --dashboard    Open web dashboard         ║
║  python app.py --all          Run everything             ║
║                                                          ║
║  API Endpoints:                                          ║
║    POST /predict       Price prediction                   ║
║    POST /recommend     Similar properties                 ║
║    POST /investment    ROI & rental analysis              ║
║    POST /anomaly       Anomaly detection                  ║
║    GET  /health        Health check                      ║
║    GET  /docs          Swagger UI                        ║
╚══════════════════════════════════════════════════════════╝
    """
    print(help_text)


# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AI Real Estate Analytics Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pipeline",  action="store_true", help="Run training pipeline")
    parser.add_argument("--api",       action="store_true", help="Start FastAPI server")
    parser.add_argument("--dashboard", action="store_true", help="Open HTML dashboard")
    parser.add_argument("--all",       action="store_true", help="Run pipeline + API + dashboard")
    parser.add_argument("--api-port",  type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--dash-port", type=int, default=8080, help="Dashboard port (default: 8080)")

    args = parser.parse_args()

    if args.all or (not args.pipeline and not args.api and not args.dashboard):
        if args.all:
            run_pipeline()
            # Start API in background thread
            api_thread = threading.Thread(
                target=run_api, kwargs={"port": args.api_port}, daemon=True
            )
            api_thread.start()
            time.sleep(2)
            run_dashboard(port=args.dash_port)
        else:
            print_help()
        return

    if args.pipeline:
        run_pipeline()

    if args.api:
        run_api(port=args.api_port)

    if args.dashboard:
        run_dashboard(port=args.dash_port)


if __name__ == "__main__":
    main()
