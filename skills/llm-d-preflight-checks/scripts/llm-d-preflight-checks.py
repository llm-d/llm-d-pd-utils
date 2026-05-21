#!/usr/bin/env python3
"""
LLM-D preflight checks for pod inspection before vLLM starts.

Behavior is controlled by the LLMD_PREFLIGHT_CHECKS environment variable:
    - unset, "disable", "none" — print diagnostics and exit immediately
    - "pause" — start HTTP server that blocks until /exit is called
    - "topology" — print diagnostics and exit (reserved for future topology checks)
    - "nixl" — print diagnostics and exit (reserved for future NixL checks)

When in "pause" mode, the HTTP server provides:
    GET /health  — 200 OK (for K8s probes)
    GET /info    — system diagnostics (env, GPU topology, CPU, PCI, etc.)
    GET /exit    — gracefully shut down the server and continue startup
    GET *        — 200 OK (catch-all for any other probe paths)

Usage:
    python3 llm-d-preflight-check.py [--info] [PORT]

    --info  Print system diagnostics to stdout and exit (no server started).
    PORT    defaults to 8000. If the port is in use, the next available port
            (8001, 8002, ...) is tried automatically.
"""

import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import threading


def run_cmd(cmd):
    """Run a shell command and return its output, or an error message."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr.strip():
            output += "\n[stderr] " + result.stderr.strip()
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "(command timed out after 30s)"
    except Exception as e:
        return f"(error: {e})"


def gather_info():
    """Collect system diagnostics."""
    sections = {}

    # Environment variables (sorted)
    sections["environment"] = "\n".join(
        f"{k}={v}" for k, v in sorted(os.environ.items())
    )

    has_nvidia_smi = shutil.which("nvidia-smi") is not None
    has_rocm_smi = shutil.which("rocm-smi") is not None

    if has_nvidia_smi:
        sections["nvidia-smi"] = run_cmd("nvidia-smi")
        sections["nvidia-smi topo -m"] = run_cmd("nvidia-smi topo -m")
        sections["nvidia-smi nvlink --status"] = run_cmd("nvidia-smi nvlink --status")
    elif has_rocm_smi:
        sections["rocm-smi"] = run_cmd("rocm-smi")
        sections["rocm-smi --showtopo"] = run_cmd("rocm-smi --showtopo")
        sections["rocm-smi --showbus"] = run_cmd("rocm-smi --showbus")
    else:
        sections["gpu"] = "(neither nvidia-smi nor rocm-smi found)"

    if shutil.which("lscpu"):
        sections["lscpu"] = run_cmd("lscpu")

    if shutil.which("lspci"):
        sections["lspci -tv"] = run_cmd("lspci -tv")

    return sections


def build_info_text(sections):
    """Format diagnostics as plain text with section headers."""
    parts = []
    for title, content in sections.items():
        parts.append(f"===== {title} =====")
        parts.append(content)
        parts.append("")
    return "\n".join(parts)


def find_available_port(start_port, max_attempts=100):
    """Find an available port starting from start_port."""
    for offset in range(max_attempts):
        port = start_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                if offset == 0:
                    print(f"Port {port} is in use, trying next...")
                continue
    raise RuntimeError(
        f"No available port found in range {start_port}-{start_port + max_attempts - 1}"
    )


def make_handler(shutdown_event):
    """Create an HTTP request handler with /info and /exit endpoints."""

    # Cache info output so repeated /info calls are fast
    info_cache = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/exit":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"shutting down"}\n')
                shutdown_event.set()

            elif self.path == "/info":
                if "text" not in info_cache:
                    sections = gather_info()
                    info_cache["text"] = build_info_text(sections)
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(info_cache["text"].encode("utf-8", errors="replace"))

            else:
                # Catch-all: /health, /, or any other path — 200 OK for probes
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}\n')

        def log_message(self, format, *args):
            # Suppress default access logs
            pass

    return Handler


def run_server(start_port):
    """Start the HTTP server and block until /exit is called."""
    port = find_available_port(start_port)

    shutdown_event = threading.Event()
    handler_class = make_handler(shutdown_event)

    server = http.server.HTTPServer(("", port), handler_class)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"=== PAUSED: vLLM startup is on hold ===")
    print(f"Preflight server listening on :{port}")
    print(f"  GET /health — probe endpoint")
    print(f"  GET /info   — system diagnostics (env, GPU, CPU, PCI)")
    print(f"  GET /exit   — shut down server and continue startup")
    print(f"To resume vLLM startup, call /exit:")
    print(f"  curl http://localhost:{port}/exit")
    print(f"Or from within the cluster:")
    print(f"  kubectl exec <pod> -c vllm -- curl -s http://localhost:{port}/exit")

    shutdown_event.wait()

    print("Shutting down preflight server...")
    server.server_close()
    print("Continuing...")


def main():
    print("=== llm-d-preflight-checks.py starting ===")

    # Parse arguments: [--info] [PORT]
    args = sys.argv[1:]
    info_mode = False
    if "--info" in args:
        info_mode = True
        args.remove("--info")

    if args:
        start_port = int(args[0])
    else:
        start_port = int(os.environ.get("VLLM_INFERENCE_PORT", "8000"))

    # --info: print diagnostics to stdout and exit
    if info_mode:
        sections = gather_info()
        print(build_info_text(sections))
        return

    raw_mode = os.environ.get("LLMD_PREFLIGHT_CHECKS", "")
    mode = raw_mode.lower().strip()
    if raw_mode:
        print(f"LLMD_PREFLIGHT_CHECKS={raw_mode!r} (mode={mode!r})")
    else:
        print("LLMD_PREFLIGHT_CHECKS is not set, defaulting to diagnostics-only mode")

    if mode == "pause":
        sections = gather_info()
        print(build_info_text(sections))
        print("=== Preflight checks PAUSED: waiting for /exit before allowing regular pod startup ===")
        run_server(start_port)
    elif mode in ("topology", "nixl"):
        print(f"Mode {mode!r}: printing diagnostics (extended checks not yet implemented)")
        sections = gather_info()
        print(build_info_text(sections))
    else:
        # unset, "disable", "none", or any unrecognized value
        sections = gather_info()
        print(build_info_text(sections))


if __name__ == "__main__":
    main()
