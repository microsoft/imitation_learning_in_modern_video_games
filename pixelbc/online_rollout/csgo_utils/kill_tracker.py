"""Live GSI kill tracking shared by the CS:GO rollouts.

Runs the GSI HTTP server (already bound to 127.0.0.1:3000 in meta_utils) in a
daemon thread so CS:GO's game-state POSTs keep ``server.data_all`` current, and
reads cumulative kills each frame WITHOUT blocking the render loop. Needs the GSI
cfg in the LEGACY path
(``...\\Counter-Strike Global Offensive\\csgo\\cfg\\gamestate_integration_*.cfg``);
with no cfg / no connection the count simply stays 0 (harmless).
"""
import json
import threading
import time
from pathlib import Path

from pixelbc.online_rollout.csgo_utils.meta_utils import server

_server_thread_started = False


def _ensure_gsi_server():
    """Serve GSI POSTs in a daemon thread (idempotent)."""
    global _server_thread_started
    if not _server_thread_started:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        _server_thread_started = True


def _read_total_kills():
    data = getattr(server, "data_all", None) or {}
    try:
        return int(data["player"]["match_stats"]["kills"])
    except (KeyError, TypeError, ValueError):
        return None


class KillTracker:
    """Session-relative cumulative kills + kills/minute, driven by live GSI."""

    def __init__(self):
        _ensure_gsi_server()
        self._start_kills = None
        self.session_kills = 0
        self._t0 = time.time()
        self.series = []

    def update(self):
        """Poll GSI once (non-blocking) and return session cumulative kills."""
        total = _read_total_kills()
        if total is not None:
            if self._start_kills is None:
                self._start_kills = total
            self.session_kills = max(0, total - self._start_kills)
        self.series.append(self.session_kills)
        return self.session_kills

    def report(self, out_path=None):
        """Print kills/minute and (optionally) save a per-step cumulative-kills
        JSON compatible with csgo_report_results.py (`rollout_data["kills"]`)."""
        duration = max(1e-6, time.time() - self._t0)
        kpm = self.session_kills / duration * 60.0
        print(f"\n[kills] {self.session_kills} kills in {duration:.0f}s = {kpm:.2f} kills/min", flush=True)
        if out_path is not None:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(
                {"kills": self.series, "duration_s": duration, "kills_per_minute": kpm}
            ))
            print(f"[kills] saved {p}", flush=True)
        return kpm
