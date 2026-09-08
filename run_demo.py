"""
One-command demo launcher.

Starts the control-room server, waits for it to answer, seeds the database with
shift history if it is empty, opens the dashboard in a browser and then runs the
cabin edge node in this terminal. Ctrl+C stops everything cleanly.

    python run_demo.py                    # webcam if present, else simulated
    python run_demo.py --synthetic        # force the scripted operator
    python run_demo.py --no-history       # skip seeding the report data
    python run_demo.py --server-only      # just the control room
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edge.config import load_config          # noqa: E402

PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))


def wait_for_server(url: str, timeout: float = 25.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url + "/api/health", timeout=1.5).ok:
                return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def pipe_output(stream, prefix: str) -> None:
    """Prefix the server's stdout so the two processes stay distinguishable."""
    for line in iter(stream.readline, ""):
        if line.strip():
            print("%s %s" % (prefix, line.rstrip()), flush=True)


def database_is_empty(cfg) -> bool:
    from server.database import Database
    try:
        return Database(str(cfg.server.database)).shift_report(hours=24)["samples"] == 0
    except Exception:
        return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the full fatigue-monitor demo")
    parser.add_argument("--synthetic", action="store_true",
                        help="force the scripted operator instead of the webcam")
    parser.add_argument("--headless", action="store_true",
                        help="run the edge node without its GUI window")
    parser.add_argument("--no-history", action="store_true",
                        help="do not seed simulated shift history")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--server-only", action="store_true")
    parser.add_argument("--scenario",
                        choices=["alert", "progressive_fatigue", "microsleep_event"],
                        default="progressive_fatigue")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="stop the edge node after N seconds")
    args = parser.parse_args(argv)

    cfg = load_config()
    url = "http://%s:%d" % (cfg.server.host, int(cfg.server.port))

    print("=" * 70)
    print("  Crane Operator Fatigue Monitor - full system demo")
    print("=" * 70)

    # ---- 1. control-room server -------------------------------------
    print("\n[1/4] starting control-room server ...")
    server = subprocess.Popen(
        [PY, "-m", "server.app"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    threading.Thread(target=pipe_output, args=(server.stdout, "  [server]"),
                     daemon=True).start()

    if not wait_for_server(url):
        print("  server did not start - check the messages above")
        server.terminate()
        return 1
    print("  server ready at %s" % url)

    try:
        # ---- 2. seed report history ---------------------------------
        if not args.no_history and database_is_empty(cfg):
            print("\n[2/4] seeding 8 hours of shift history for the report page ...")
            subprocess.run([PY, "-m", "tools.simulate_shift", "--hours", "8",
                            "--cranes", "3"], cwd=ROOT, check=False)
        else:
            print("\n[2/4] database already has history - skipping seed")

        # ---- 3. dashboard -------------------------------------------
        print("\n[3/4] dashboard : %s/" % url)
        print("      report    : %s/report" % url)
        if not args.no_browser:
            webbrowser.open(url + "/")

        if args.server_only:
            print("\nserver running. Press Ctrl+C to stop.")
            server.wait()
            return 0

        # ---- 4. cabin edge node -------------------------------------
        print("\n[4/4] starting cabin edge node ...")
        print("      close its window or press 'q' in it to end the demo\n")
        time.sleep(1.0)

        cmd = [PY, "-m", "edge.node", "--scenario", args.scenario]
        if args.synthetic:
            cmd += ["--backend", "synthetic"]
        if args.headless:
            cmd.append("--headless")
        if args.duration:
            cmd += ["--duration", str(args.duration)]
        subprocess.run(cmd, cwd=ROOT, check=False)

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        print("\nstopping control-room server ...")
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
        print("done. The database is kept at %s" % cfg.server.database)
        print("Re-open the report any time with:  python -m server.app")

    return 0


if __name__ == "__main__":
    sys.exit(main())
