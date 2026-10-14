import contextlib
import importlib
import io
from pathlib import Path
import shlex
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/llm-d-networking-tests/scripts"))
ccl = importlib.import_module("run-tests-ccl")
discovery = importlib.import_module("run-tests-discovery")


def completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def csv_row(size=8, world=4, **overrides):
    cols = [str(world), "allreduce", "sum", "float32", "4", str(size // 4),
            str(size), "1", "20", "1", "2", "1.5", "0", "0", "0.1", "0.15"]
    for key, value in overrides.items():
        cols[int(key)] = value
    return ",".join(cols)


class TopologyTests(unittest.TestCase):
    def test_bordered_topology_and_p2p(self):
        topo = """
+---------+---------+---------+---------------+
|         | GPU 0   | GPU 1   | CPU Affinity  |
+---------+---------+---------+---------------+
| GPU 0   | S       | SYS     | 0-15          |
| GPU 1   | SYS     | S       | 16-31         |
+---------+---------+---------+---------------+
"""
        for text in (topo, topo.replace("|", "\u2502")):
            with self.subTest(text=text):
                connections, labels = discovery._parse_xpu_topo(text)
                self.assertEqual(labels, ["GPU0", "GPU1"])
                self.assertEqual(connections["GPU0", "GPU1"], "SYS")
                self.assertTrue(discovery._check_hop_distance(connections, labels))
        p2p = topo.replace("SYS", "NS").replace("| S ", "| X ")
        self.assertEqual(discovery._parse_xpu_link_status(p2p), (0, 2))

    def test_tiles_are_aggregated_without_intra_device_fabric(self):
        connections, labels = discovery._parse_xpu_topo("""
        GPU 0/0 GPU 0/1 GPU 1/0 GPU 1/1
GPU 0/0 S MDF PIX SYS
GPU 0/1 MDF S PIX PIX
GPU 1/0 PIX PIX S MDF
GPU 1/1 SYS PIX MDF S
""")
        self.assertEqual(labels, ["GPU0", "GPU1"])
        self.assertEqual(connections, {
            ("GPU0", "GPU0"): "S", ("GPU1", "GPU1"): "S",
            ("GPU0", "GPU1"): "SYS", ("GPU1", "GPU0"): "SYS",
        })
        self.assertFalse(discovery._check_nvlink_topology(connections, labels, "xpu")[0])

    def test_tile_p2p_uses_weakest_capability(self):
        text = """
        GPU 0/0 GPU 0/1 GPU 1/0 GPU 1/1
GPU 0/0 X OK OK NS
GPU 0/1 OK X OK OK
GPU 1/0 OK OK X OK
GPU 1/1 OK OK OK X
"""
        self.assertEqual(discovery._parse_xpu_link_status(text), (1, 1))

    def test_single_gpu_and_alternate_labels(self):
        for label in ("GPU 0", "Device 0", "xe0", "0"):
            with self.subTest(label=label):
                self.assertEqual(
                    discovery._parse_xpu_topo(f"  {label}\n{label} S\n"),
                    ({("GPU0", "GPU0"): "S"}, ["GPU0"]),
                )
                self.assertEqual(
                    discovery._parse_xpu_link_status(f"  {label}\n{label} X\n"), (0, 0)
                )

    def test_incomplete_or_invalid_matrix_is_not_success(self):
        for text in ("", "Error: unavailable", "GPU0 GPU1\nGPU0 S SYS",
                     "GPU0 GPU1\nGPU0 S GARBAGE\nGPU1 SYS S"):
            with self.subTest(text=text):
                self.assertEqual(discovery._parse_xpu_topo(text), ({}, []))
                with self.assertRaises(ValueError):
                    discovery._parse_xpu_link_status(text)

    def test_existing_nvidia_and_amd_fabric_checks(self):
        for vendor, fabric in (("nvidia", "NV18"), ("amd", "XGMI")):
            with self.subTest(vendor=vendor):
                connections = {("GPU0", "GPU1"): fabric, ("GPU1", "GPU0"): fabric}
                labels = ["GPU0", "GPU1"]
                self.assertTrue(discovery._check_nvlink_topology(connections, labels, vendor)[0])
                self.assertEqual(discovery._check_hop_distance(connections, labels), [])
                self.assertEqual(discovery._check_nvswitch_domains(connections, labels, vendor),
                                 (True, []))

    def test_discovery_reports_parse_failure_as_unknown_with_warnings(self):
        with patch.object(discovery, "_detect_gpu_vendor", return_value="xpu"), patch.object(
                discovery, "_get_gpu_topo", return_value=("unexpected output", True)), patch.object(
                discovery, "_get_gpu_link_status", return_value=("unexpected output", True)), patch.object(
                discovery, "exec_in_pod", return_value=completed(returncode=1)):
            topo, warnings = discovery._discover_pod_topology("pod", "pod", io.StringIO())
        self.assertIsNone(topo["has_nvlink"])
        self.assertNotIn("gpu_count", topo)
        self.assertTrue(any("could not parse XPU topology" in warning for warning in warnings))
        self.assertTrue(any("could not parse XPU P2P" in warning for warning in warnings))


class CollectiveTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.output = self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def test_csv_accepts_exact_world_and_sizes(self):
        output = "#ranks,collective\n" + csv_row() + "\n" + csv_row(16)
        with patch.object(ccl, "exec_in_pod", return_value=completed(output)) as execute:
            rows = ccl._parse_xccl_csv("p1", "/tmp/current.csv", 4, [8, 16])
        self.assertEqual([row["size"] for row in rows], [8, 16])
        self.assertEqual(execute.call_args.args[1], ["cat", "/tmp/current.csv"])

    def test_csv_rejects_invalid_or_incomplete_results(self):
        invalid = [
            "", csv_row(world=1), csv_row(16), "broken",
            csv_row() + "\n" + csv_row(),
            csv_row(**{"1": "allgather"}), csv_row(**{"3": "float64"}),
            csv_row(**{"7": "2"}), csv_row(**{"8": "0"}),
            csv_row(**{"8": "1"}), csv_row(**{"8": "21"}), csv_row(**{"11": "nan"}),
            csv_row(**{"14": "-1"}), csv_row(**{"15": "inf"}),
        ]
        for text in invalid:
            with self.subTest(text=text), patch.object(
                    ccl, "exec_in_pod", return_value=completed(text)):
                self.assertIsNone(ccl._parse_xccl_csv("p1", "/tmp/current.csv", 4, [8]))

    def test_csv_requires_configured_repetition_count(self):
        with patch.object(ccl, "AR_ITERS", "7"), patch.object(
                ccl, "exec_in_pod", return_value=completed(csv_row(**{"8": "7"}))):
            self.assertIsNotNone(ccl._parse_xccl_csv("p1", "/tmp/current.csv", 4, [8]))

    def test_csv_reports_mean_time(self):
        with patch.object(ccl, "exec_in_pod", return_value=completed(csv_row())):
            rows = ccl._parse_xccl_csv("p1", "/tmp/current.csv", 4, [8])
        self.assertEqual(rows[0]["time_us"], 1.5)

    def test_intel_mpi_is_found_after_sourcing_oneapi(self):
        response = completed("Intel(R) MPI Library\nMPI_PATH=/opt/intel/oneapi/mpi/2021.15/bin/mpirun\n")
        with patch.object(ccl, "exec_in_pod", return_value=response) as execute:
            self.assertEqual(ccl.find_intel_mpirun("p1"),
                             "/opt/intel/oneapi/mpi/2021.15/bin/mpirun")
        script = execute.call_args.args[1][-1]
        self.assertIn("source /opt/intel/oneapi/setvars.sh", script)
        self.assertIn('"$I_MPI_ROOT/bin/mpirun" --version', script)

    def test_openmpi_or_failed_mpi_probe_is_rejected(self):
        for response in (
            completed("Open MPI\nMPI_PATH=/usr/bin/mpirun\n"),
            completed("Open MPI\nMPI_PATH=/opt/intel/mpi/bin/mpirun\n"),
            completed(returncode=1, stderr="missing oneAPI"),
        ):
            with self.subTest(response=response), patch.object(
                    ccl, "exec_in_pod", return_value=response):
                self.assertIsNone(ccl.find_intel_mpirun("p1"))

    def test_build_respects_install_deps_and_failure_status(self):
        with patch.object(ccl._get_common(), "INSTALL_DEPS", False), patch.object(
                ccl, "_xccl_bench_available", return_value=False), patch.object(
                ccl, "exec_in_pod") as execute:
            self.assertFalse(ccl.ensure_xccl_bench("p1"))
            execute.assert_not_called()
        with patch.object(ccl._get_common(), "INSTALL_DEPS", True), patch.object(
                ccl, "_xccl_bench_available", return_value=False), patch.object(
                ccl, "exec_in_pod", return_value=completed("XCCL_BUILD_OK", 1)):
            self.assertFalse(ccl.ensure_xccl_bench("p1"))

    def test_xpu_build_uses_existing_vendor_api(self):
        with patch.object(ccl, "ensure_xccl_bench", return_value=True) as build:
            self.assertTrue(ccl.ensure_collective_tests("p1", "xpu"))
        build.assert_called_once()

    def prepare_run(self, response=None):
        def mocked(name, **kwargs):
            return self.stack.enter_context(patch.object(ccl, name, **kwargs))
        self.hosts = mocked("_ensure_hosts_entries")
        self.env = mocked("_env_values", return_value={})
        self.parse = mocked("_parse_xccl_csv", return_value=[{"world": 4, "size": 8}])
        self.execute = mocked("exec_in_pod", side_effect=[
            response if response is not None else completed(), completed()
        ])
        self.stack.enter_context(patch.object(ccl, "AR_MAX_BYTES", "16"))

    def run_collective(self):
        return ccl.run_all_reduce([("p1", "10.0.0.1"), ("p2", "10.0.0.2")],
                                 "xpu", {}, gpu_count=2, mpirun_path="/opt/intel/mpi/bin/mpirun")

    def test_success_sources_both_launcher_and_ranks_and_uses_unique_csv(self):
        self.prepare_run()
        self.assertIsNotNone(self.run_collective())
        script = self.execute.call_args_list[0].args[1][-1]
        self.assertEqual(script.count("source /opt/intel/oneapi/setvars.sh"), 2)
        self.assertIn("--elem_counts 2,4", script)
        self.assertIn("--check last", script)
        self.assertNotIn("--allow-run-as-root", script)
        self.assertIn("FI_PROVIDER=tcp", script)
        path = self.parse.call_args.args[1]
        self.assertRegex(path, r"/result-[0-9a-f]{32}\.csv$")
        self.assertEqual(self.parse.call_args.args[2:], (4, [8, 16]))
        self.assertEqual(self.execute.call_args_list[-1].args[1], ["rm", "-f", path])

    def test_xpu_uses_fixed_gpu_workload(self):
        self.prepare_run()
        self.run_collective()
        script = self.execute.call_args_list[0].args[1][-1]
        launcher = shlex.split(script.partition("exec ")[2])
        rank_script = launcher[-1]
        benchmark = shlex.split(rank_script.partition("exec ")[2])
        expected = {
            "--coll": "allreduce", "--backend": "sycl",
            "--sycl_dev_type": "gpu", "--sycl_root_dev": "1",
            "--sycl_mem_type": "usm", "--sycl_usm_type": "device",
            "--sycl_queue_type": "in_order", "--inplace": "0", "--buf_count": "1",
            "--dtype": "float32", "--reduction": "sum", "--check": "last",
            "--iters": "20", "--warmup_iters": "5", "--iter_policy": "off",
        }
        for option, value in expected.items():
            with self.subTest(option=option):
                self.assertEqual(benchmark[benchmark.index(option) + 1], value)
        self.assertEqual(launcher[launcher.index("-n") + 1], "4")
        self.assertEqual(launcher[launcher.index("-ppn") + 1], "2")

    def test_vendors_share_iteration_settings(self):
        self.prepare_run()
        self.execute.side_effect = None
        self.execute.return_value = completed(
            "8 2 float sum -1 1.5 0.1 0.15 0 2.5 0.2 0.3 0"
        )
        with patch.object(ccl, "AR_ITERS", "7"), patch.object(ccl, "AR_WARMUP_ITERS", "3"):
            self.run_collective()
            script = self.execute.call_args_list[-2].args[1][-1]
            self.assertIn("--iters 7 --warmup_iters 3 --iter_policy off", script)
            for vendor in ("nvidia", "amd"):
                with self.subTest(vendor=vendor):
                    rows = ccl.run_all_reduce(
                        [("p1", "10.0.0.1"), ("p2", "10.0.0.2")], vendor, {}, gpu_count=2
                    )
                    args = shlex.split(self.execute.call_args.args[1][-1])
                    expected = {
                        "-np": "4", "-N": "2", "-g": "1", "-b": "8", "-e": "16", "-f": "2",
                        "-d": "float", "-o": "sum", "-n": "7", "-w": "3", "-c": "1", "-a": "1",
                    }
                    for option, value in expected.items():
                        self.assertEqual(args[args.index(option) + 1], value)
                    self.assertEqual(rows[0]["time_us"], 1.5)
                    self.assertEqual(rows[0]["busbw_gbs"], 0.15)

    def test_failure_never_reads_old_csv(self):
        self.prepare_run(completed(returncode=137, stderr="launcher killed"))
        self.assertIsNone(self.run_collective())
        self.parse.assert_not_called()
        self.assertIn("FAILED", self.output.getvalue())

    def test_timeout_cleans_up_without_reading_csv(self):
        self.prepare_run(subprocess.TimeoutExpired("mpirun", 900))
        self.assertIsNone(self.run_collective())
        self.parse.assert_not_called()
        self.assertEqual(self.execute.call_args_list[-1].args[1][:2], ["rm", "-f"])

    def test_default_sizes_cover_amd_range(self):
        self.prepare_run()
        with patch.object(ccl, "AR_MAX_BYTES", "128M"):
            self.run_collective()
        self.assertEqual(self.parse.call_args.args[3], [2 ** power for power in range(3, 28)])

    def test_csv_paths_are_different_between_runs(self):
        self.prepare_run()
        self.execute.side_effect = None
        self.execute.return_value = completed()
        self.run_collective()
        first = self.parse.call_args.args[1]
        self.run_collective()
        second = self.parse.call_args.args[1]
        self.assertNotEqual(first, second)

    def test_pod_provider_is_preserved(self):
        self.prepare_run()
        self.env.return_value = {"FI_PROVIDER": "verbs"}
        self.run_collective()
        script = self.execute.call_args_list[0].args[1][-1]
        self.assertEqual(script.count("FI_PROVIDER="), 1)
        self.assertIn("FI_PROVIDER=verbs", script)


class VendorRoutingTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(ccl._get_common(), "INSTALL_DEPS", False))
        self.pods = [("p1", "10.0.0.1"), ("p2", "10.0.0.2")]
        self.vendors = self.mock("detect_all_gpu_vendors", return_value={"p1": "xpu", "p2": "xpu"})
        self.count = self.mock("detect_gpu_count", return_value=2)
        self.build = self.mock("ensure_collective_tests", return_value=True)
        self.mpi = self.mock("find_intel_mpirun", return_value="/opt/intel/mpi/bin/mpirun")
        self.openmpi = self.mock("find_mpirun", return_value="/usr/bin/mpirun")
        self.install_mpi = self.mock("install_mpi")
        self.install_deps = self.mock("install_build_deps")
        self.mock("_find_nccl_paths", return_value=(None, None))
        self.ssh = self.mock("setup_ssh_cluster", return_value=(True, "/root/.ssh/test_key"))
        self.stop = self.mock("stop_sshd_all")
        self.execute = self.mock("exec_in_pod", return_value=completed("driver=12.0"))
        self.rows = [{"size": 8, "count": 2, "time_us": 1, "algbw_gbs": 0.1, "busbw_gbs": 0.15}]
        self.benchmark = self.mock("run_all_reduce", return_value=self.rows)

    def mock(self, name, **kwargs):
        return self.stack.enter_context(patch.object(ccl, name, **kwargs))

    def test_vendors_share_build_launch_cleanup_and_reporting(self):
        self.vendors.return_value = {"p1": "nvidia", "p2": "amd", "p3": "xpu"}
        results = ccl.run_nccl_rccl(self.pods + [("p3", "10.0.0.3")], {})
        self.assertEqual(set(results), {
            "NCCL all_reduce BusBW (GB/s)", "RCCL all_reduce BusBW (GB/s)",
            "XCCL all_reduce BusBW (GB/s)",
        })
        self.assertEqual([call.args[1] for call in self.benchmark.call_args_list],
                         ["nvidia", "amd", "xpu"])
        self.assertEqual(self.build.call_count, 3)
        self.assertEqual(self.ssh.call_count, 3)
        self.assertEqual(self.stop.call_count, 3)
        self.assertEqual(self.openmpi.call_count, 2)
        self.mpi.assert_called_once_with("p3")

    def test_xpu_install_deps_does_not_install_openmpi(self):
        with patch.object(ccl._get_common(), "INSTALL_DEPS", True):
            ccl.run_nccl_rccl(self.pods, {})
        self.install_deps.assert_not_called()
        self.openmpi.assert_not_called()
        self.install_mpi.assert_not_called()
        self.assertEqual(self.build.call_count, 2)

    def test_ssh_failure_does_not_launch_xpu(self):
        self.ssh.return_value = (False, "/root/.ssh/test_key")
        self.assertEqual(ccl.run_nccl_rccl(self.pods, {}), {})
        self.benchmark.assert_not_called()
        self.stop.assert_called_once()

    def test_mismatched_mpi_paths_do_not_launch(self):
        self.mpi.side_effect = ["/mpi/v1/bin/mpirun", "/mpi/v2/bin/mpirun"]
        self.assertEqual(ccl.run_nccl_rccl(self.pods, {}), {})
        self.ssh.assert_not_called()
        self.benchmark.assert_not_called()

    def test_missing_intel_mpi_does_not_fall_back_to_openmpi(self):
        self.mpi.return_value = None
        self.assertEqual(ccl.run_nccl_rccl(self.pods, {}), {})
        self.openmpi.assert_not_called()
        self.install_mpi.assert_not_called()
        self.ssh.assert_not_called()

    def test_gpu_count_and_build_failures_skip_before_ssh(self):
        for count, build in ((0, True), (2, False)):
            with self.subTest(count=count, build=build):
                self.count.return_value = count
                self.build.return_value = build
                self.assertEqual(ccl.run_nccl_rccl(self.pods, {}), {})
        self.ssh.assert_not_called()
        self.benchmark.assert_not_called()

    def test_failed_benchmark_produces_no_results_and_cleans_up(self):
        self.benchmark.return_value = None
        self.assertEqual(ccl.run_nccl_rccl(self.pods, {}), {})
        self.stop.assert_called_once()

class CompatibilityTests(unittest.TestCase):
    def test_original_python_api_remains_available(self):
        self.assertTrue(callable(ccl.run_nccl_rccl))

    def test_cli_aliases_and_entrypoint(self):
        scripts = ROOT / "skills/llm-d-networking-tests/scripts"
        result = subprocess.run([sys.executable, str(scripts / "run-tests-ccl.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run-tests.sh -t ccl", result.stdout)
        for option in ("ccl", "nccl-rccl", "collectives", "all"):
            code = (
                "import importlib; m=importlib.import_module('run-tests'); "
                "assert 'ccl' in m.SELECTED_TESTS; "
                "assert callable(m.run_nccl_rccl)"
            )
            result = subprocess.run([sys.executable, "-c", code, "--tests", option],
                                    cwd=scripts, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


class ImageTests(unittest.TestCase):
    def test_entrypoint_and_wrapper_layout(self):
        dockerfile = (ROOT / "Dockerfile.xpu").read_text()
        self.assertIn('ENTRYPOINT []\nCMD ["sleep", "infinity"]', dockerfile)
        for line in dockerfile.splitlines():
            if line.startswith("COPY skills/"):
                _, source, destination = shlex.split(line)
                self.assertEqual(destination, "/opt/llm-d-pd-utils/" + source)
        wrapper = (ROOT / "run-tests.sh").read_text()
        self.assertIn('$SCRIPT_DIR/skills/llm-d-networking-tests/scripts/run-tests.py', wrapper)
        self.assertIn("./run-tests.sh --help", dockerfile)


if __name__ == "__main__":
    unittest.main()
