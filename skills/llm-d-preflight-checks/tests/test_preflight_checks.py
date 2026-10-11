# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP regressions for the preflight pause server (standard library only)."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.client import RemoteDisconnected
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import ProxyHandler, build_opener


# Run the real entry point in a child process, with only diagnostics collection
# replaced so its duration and environment output are deterministic.
BOOTSTRAP = """
import importlib.util
from pathlib import Path
import sys
import time

script, port, state, mode = sys.argv[1:]
state = Path(state)
spec = importlib.util.spec_from_file_location("preflight_checks", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
calls = 0
def gather_info():
    global calls
    calls += 1
    call = calls
    with (state / "calls").open("a") as stream:
        stream.write(str(call) + "\\n")
    if mode == "fail" and call == 2:
        raise RuntimeError("temporary diagnostic collection failure")
    if mode == "slow" and call > 1:
        (state / "collecting").touch()
        deadline = time.monotonic() + 5
        while not (state / "release").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
    return {"diagnostics": "test diagnostics " + str(call)}
module.gather_info = gather_info
sys.argv = [script, port]
module.main()
"""


class PreflightServerTests(unittest.TestCase):
    def wait_for_collection(self, state):
        deadline = time.monotonic() + 5
        while not (state / "collecting").exists():
            if time.monotonic() >= deadline:
                self.fail("Diagnostics collection did not start")
            time.sleep(0.01)

    @contextmanager
    def server(self, mode="slow"):
        script = Path(__file__).parents[1] / "scripts" / "llm-d-preflight-checks.py"
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        opener = build_opener(ProxyHandler({}))

        def get(path, timeout=5):
            with opener.open(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
                return response.status, response.read().decode()

        with tempfile.TemporaryDirectory(prefix="preflight-test-") as directory:
            state = Path(directory)
            process = subprocess.Popen(
                [sys.executable, "-u", "-c", BOOTSTRAP, str(script), str(port), directory, mode],
                env={"PATH": os.defpath, "LLMD_PREFLIGHT_CHECKS": "pause"},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        if get("/health", timeout=0.2)[0] == 200:
                            break
                    except OSError:
                        time.sleep(0.01)
                else:
                    self.fail("Preflight server did not start")

                yield state, process, get
            finally:
                (state / "release").touch()
                if process.poll() is None:
                    try:
                        get("/exit")
                        process.communicate(timeout=5)
                    except (OSError, subprocess.TimeoutExpired):
                        process.kill()
                        process.communicate(timeout=5)
                else:
                    process.communicate(timeout=5)

    def test_health_remains_available_while_info_is_collected(self):
        with self.server() as (state, process, get):
            with ThreadPoolExecutor(max_workers=1) as executor:
                info = executor.submit(get, "/info")
                self.wait_for_collection(state)
                try:
                    status, body = get("/health", timeout=0.5)
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body), {"status": "ok"})
                finally:
                    (state / "release").touch()
                response = info.result(timeout=5)
                self.assertEqual(response[0], 200)
                self.assertIn("test diagnostics", response[1])
                self.assertEqual(get("/info"), response)

    def test_concurrent_info_requests_share_one_collection(self):
        with self.server() as (state, process, get):
            with ThreadPoolExecutor(max_workers=8) as executor:
                first = executor.submit(get, "/info")
                self.wait_for_collection(state)
                pending = [executor.submit(get, "/info") for _ in range(7)]
                try:
                    self.assertEqual(get("/health", timeout=0.5)[0], 200)
                    self.assertEqual(get("/health", timeout=0.5)[0], 200)
                finally:
                    (state / "release").touch()
                response = first.result(timeout=5)
                for request in pending:
                    self.assertEqual(request.result(timeout=5), response)
                self.assertEqual(get("/info"), response)
                self.assertEqual((state / "calls").read_text().splitlines(), ["1", "2"])

    def test_info_can_retry_after_collection_failure(self):
        with self.server(mode="fail") as (state, process, get):
            with self.assertRaises(RemoteDisconnected):
                get("/info")
            self.assertEqual(get("/health", timeout=0.5)[0], 200)
            response = get("/info")
            self.assertEqual(response[0], 200)
            self.assertIn("test diagnostics 3", response[1])
            self.assertEqual(get("/info"), response)
            self.assertEqual((state / "calls").read_text().splitlines(), ["1", "2", "3"])

    def test_exit_waits_for_in_flight_info_response(self):
        with self.server() as (state, process, get):
            with ThreadPoolExecutor(max_workers=1) as executor:
                info = executor.submit(get, "/info")
                self.wait_for_collection(state)
                try:
                    status, body = get("/exit", timeout=0.5)
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body), {"status": "shutting down"})
                    with self.assertRaises(subprocess.TimeoutExpired):
                        process.wait(timeout=1)
                finally:
                    (state / "release").touch()
                self.assertEqual(info.result(timeout=5)[0], 200)
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertIn("Continuing...", stdout)


if __name__ == "__main__":
    unittest.main()
