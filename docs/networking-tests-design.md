## Networking tests design

Utility scripts for performance testing and benchmarking LLM deployments on Kubernetes (llm-d).

### Test Scripts

- `run-tests.py` - Unified test runner: orchestrates perftest + iperf3 between pods, generates matplotlib plots
- `run-tests.sh` - Shell wrapper to run network performance tests
- `run-tests-common.py` - Shared utilities: pod discovery, CLI parsing, `exec_in_pod()`, SSH support
- `run-tests-iperf3.py` - iperf3 network bandwidth/latency tests between pods
- `run-tests-perftest.py` - RDMA perftest (ib_write_bw, ib_read_bw, etc.) between pods
- `run-tests-nccl-rccl.py` - NCCL/RCCL collective operation tests (all_reduce, etc.)
- `run-tests-nixlbench.py` - NIXL data transfer benchmark (GPU VRAM-to-VRAM via UCX)
- `run-tests-discovery.py` - GPU topology discovery (nvidia-smi topo, NVLink, PCIe)

## Key Commands

```bash
# Unified network tests (perftest + iperf3 with plots)
./run-tests.sh -n <namespace> -l <pod-label>

# Run specific tests
./run-tests.sh --tests perftest -n <namespace> -l <pod-label> --install-deps
./run-tests.sh --tests iperf3 -n <namespace> -l <pod-label>
./run-tests.sh --tests nccl-rccl -n <namespace> -l <pod-label>
./run-tests.sh --tests nixlbench -n <namespace> -l <pod-label> --install-deps

# Discovery only (GPU topology, NVLink, PCIe)
./run-tests.sh --discovery -n <namespace> -l <pod-label>

# Preflight checks
./run-tests.sh --preflight-info -n <namespace>
./run-tests.sh --preflight-status -n <namespace>

# Common options
#   -i, --install-deps   Install test dependencies on pods (supports non-root)
#   -v, --verbose        Print kubectl/SSH commands as they run
#   -l, --label          Pod label selector (default: llm-d.ai/model)
#   -n, --namespace      Kubernetes namespace
#   -e, --explain        Show the commands behind each finding
#   -x, --explain-verify Run each explain command and verify output
```
