# PD utils 

This repository includes [Agentic Skills](https://agentskills.io/home) that automate common llm-d operational tasks. Skills are custom slash commands defined by `SKILL.md` files — invoke them in Claude Code with `/<skill-name>` and they guide AI agent through multi-step workflows (deploying configs, running tests, interpreting results) using the instructions and scripts in this repo.

## Preflight Checks Skill

The [`/llm-d-preflight-checks`](skills/llm-d-preflight-checks/SKILL.md) skill patches an llm-d model server deployment to run a [diagnostics script](skills/llm-d-preflight-checks/scripts/llm-d-preflight-checks.py) before vLLM starts. It collects environment variables, GPU topology (`nvidia-smi topo -m`, NVLink status), and CPU/PCI info, giving operators a window to inspect the pod environment and run network tests before the model loads.

In `pause` mode (`LLMD_PREFLIGHT_CHECKS=pause`), the script starts an HTTP server on the vLLM port that satisfies K8s health probes while blocking vLLM startup. Call `/exit` on the server to release the port and let vLLM proceed. See the [SKILL.md](skills/llm-d-preflight-checks/SKILL.md) for deployment instructions covering the llm-d quickstart, P/D disaggregation, and llm-d-benchmark.

The problem addressed and design rationale is in [preflight-checks-design.md in docs directory](./docs/preflight-checks-design.md).

## Networking Test Skill

The [`/llm-d-networking-tests`](skills/llm-d-networking-tests/SKILL.md) skill validates GPU topology and inter-pod network performance for llm-d deployments. It drives the `run-tests.sh` automation scripts in this repo to:

1. **Discover GPU topology** — verify that GPUs within each pod are optimally connected (NVLink/NVSwitch for NVIDIA, Infinity Fabric for AMD) rather than separated by PCIe hops across NUMA nodes.
2. **Run network performance tests** — measure RDMA bandwidth and latency between pods using perftest (`ib_write_bw`, `ib_read_bw`), iperf3, NCCL/RCCL collectives, and nixlbench (GPU VRAM-to-VRAM via UCX).

The skill asks for the target namespace and pod label selector, then runs the tests and helps interpret the results. See the [SKILL.md](skills/llm-d-networking-tests/SKILL.md) for the full testing workflow and troubleshooting guide.

The network testing scripts used are in `run-*.py` files and their design is described in [networking-tests-design.md in docs directory](./docs/networking-tests-design.md).

## Build NIXL Image

```bash
docker build -t nixl:latest .
docker tag nixl:latest ghcr.io/<>/nixl:latest
docker push ghcr.io/<>/nixl:latest
```

## Deploy & Test

```bash
cd benchmarks
./benchmark_deployment.sh nixl -rdma
```

Use -rdma for the deployment in cluster with RoCE enabled for performance.
This will deploy [nixl-client-roce](deployment/nixl_client_roce.yaml) and [nixl-server-roce](deployment/nixl_server_roce.yaml), 
and run a benchmarking script to measure the transfer throughput (GB/s).
Refer to [benchmarking](benchmarks/README.md) for more details.


