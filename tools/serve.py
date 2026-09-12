#!/usr/bin/env python3
"""Serve web/ on localhost with caching off.

`python3 -m http.server` works, but it sends no cache headers, and browsers
then heuristically cache ES modules — after pulling an update you can quietly
be running last week's asr.js. This is the same server with
Cache-Control: no-cache, bound to 127.0.0.1 only.

    python3 tools/serve.py [port] [--no-isolate]   # default 8791 -> http://localhost:8791

By default the page is served cross-origin isolated (Cross-Origin-Opener-Policy
+ Cross-Origin-Embedder-Policy), which unlocks SharedArrayBuffer and therefore
multi-threaded wasm: Whisper on wasm went from 0.84x to 0.24x real time with
four threads. Every cross-origin resource then needs CORS or CORP headers; the
CDNs used here send them, and the minutes API is a CORS fetch. --no-isolate
turns it off if something you add does not.
"""
import functools
import http.server
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "web"


ISOLATE = "--no-isolate" not in sys.argv


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        if ISOLATE:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, fmt, *args):  # quiet unless it is an error
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)


args = [a for a in sys.argv[1:] if not a.startswith("--")]
port = int(args[0]) if args else 8791
print(f"serving {ROOT} at http://localhost:{port}{'  [cross-origin isolated]' if ISOLATE else ''}  (self-test: /selftest.html)", flush=True)
http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=str(ROOT))).serve_forever()
