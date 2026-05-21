---
name: llm-d-networking-tests
description: Run networking tests for LLM-D to ensure the environment is properly set up before vLLM is started.
---

There are two steps to verify network performance of llm-d model deployment. First, after the model is deployed verify the topology of llm-d pods used by the model. For each pod if there are multiple GPUs allocated verify that they are optimally connected, in particular that they are as close as possible connected in PCIe with interconnects such as NVLink and NVSwitch for NVIDIA and Infinity Fabric and UALink for AMD. Second, for communication between pods verify that network fabric such Infiniband or RoCE is configured inside llm-d pods for expected latency and bandwidth.

Use provided automation scripts run-test*.sh from scripts/ directory to run the tests. 

Ask user what namespace to use for tests (--namespace NS parameter in test scripts) and what pods label selector to use for testing (e.g. --label "llm-d.ai/role=decode" or "llm-d.ai/role=prefill"). The scripts will run tests in all pods matching the label selector.

The tests can also be run using ssh instead of kubectl exec by providing --ssh and --host parameters to the test scripts. If using ssh, ensure that the test machine has network access to the nodes where llm-d pods are running and that SSH keys are properly configured.

If additional information is needed use --verbose to print kubectl/SSH commands as they run. To make sure that the tests results can be reproduced by CLI commands use --explain-verify that extends the explain option to verify that each command when run as a CLI command produces expected output. The scripts will print out the exact kubectl or ssh commands used to run the tests for each pod. This allows users to easily rerun the tests manually if needed.

Follow those steps to run the tests:

- Confirm GPU topology inside pods:
  - On NVIDIA nodes: use nvidia‑smi topo -m, lscpu, lspci -tv and nvidia‑smi nvlink --status to see whether GPUs are connected via NVLink/NVSwitch and not just PCIe. Check that there are minimal hops between GPUs (ideally NODE or NVLink instead of SYS or multiple PCIe hops) and If NVLink exists, GPUs in the pod are in the same NVLink group and not split across different NVSwitch domains
  - On AMD: check GPU topology via rocm-smi and confirm that GPUs are connected via high‑bandwidth fabric (Infinity Fabric, UALink‑like topology) rather than only PCIe lanes.
  - At this stage, an initial performance baseline can be defined for expected bandwidth and latency for both intra-pod and internal GPU-to-GPU communications.
  - Run `./run-tests.sh --discovery` to automatically discover pods and run topology tests. This will print out the topology of each pod and whether it meets expected criteria for optimal GPU connectivity.
  - In case that there are missing CLI tools inside the pods re-run with -i/--install-deps to automatically install missing dependencies: `./run-tests.sh --discovery --install-deps` that will nstall all test dependencies on every pod:
  such as iperf3, perftest build tools, etcd, and nixlbench. It also builds perftest and nixlbench from source if missing and needed for testing.

- Verify pods network configuration inisde pods and run network performance tests between pods:
  - Check environment variables are used: such as NCCL_IB_HCA for the correct RDMA device (e.g., mlx5_0,mlx5_1), NCCL_IB_DISABLE=0 to enable InfiniBand/RoCE, NCCL_SOCKET_IFNAME for the RDMA‑capable interface (e.g., ib0, eth0), and NVSHMEM_USE_IB=1 to enable NVSHMEM over InfiniBand if using NVSHMEM. Also check that there are no conflicting variables such as NCCL_IB_DISABLE=1 or incorrect NCCL_IB_HCA values that would prevent RDMA from being used.
  - Check that the expected RDMA interfaces (e.g., ib0) are present and properly configured inside the pods. Use `ip addr` to verify that the RDMA interfaces are up and have IP addresses assigned. Use `ibv_devinfo` to check that the InfiniBand devices are visible and properly configured inside the pods.
  - Run network performance tests between pods:
    - Use `iperf3` to measure TCP and RDMA bandwidth and latency between pods. Run `iperf3` in server mode on one pod and in client mode on another pod to measure the bandwidth and latency of the network fabric. Verify that the results meet expected performance baselines for the given hardware and network configuration.
    - Use `perftest` (e.g., `ib_write_bw`, `ib_read_bw`, `ib_send_bw`) to measure RDMA bandwidth and latency between pods. This provides a more detailed view of RDMA performance compared to `iperf3`. Run `perftest` in server mode on one pod and in client mode on another pod to measure RDMA performance metrics. Verify that the results meet expected baselines for RDMA performance.
    - If using NVSHMEM, use `nixlbench` to measure NVSHMEM performance between pods. Run `nixlbench` in server mode on one pod and in client mode on another pod to measure NVSHMEM bandwidth and latency. Verify that the results meet expected performance baselines for NVSHMEM over InfiniBand.
    - If any of the tests fail to meet expected performance baselines, investigate potential issues with network configuration, pod placement, or hardware limitations. Check for any error messages in the test outputs and review the network configuration and topology to identify potential bottlenecks or misconfigurations. 
    - Run `./run-tests.sh --tests all` or specify individual tests with `--tests perftest,iperf3,nccl-rccl,nixlbench` to run specific tests between pods. Use `--ssh` and `--host` parameters if running tests via SSH instead of kubectl exec. Choices: perftest, iperf3, nccl-rccl, nixlbench, all (default: perftest if not tests specified).  When -t isgiven explicitly, topology discovery is skipped (use -d to force it).

The common problems leading to mismatch between performance test results and expected performance baseline are related to Kubernetes and topology GPUs on different PCIe roots / NUMA nodes. Each socket in a dual‑socket server owns its own PCIe root complex; if a pod gets GPUs on opposite sockets, their only path is via SYS‑style PCIe hops through the CPU interconnect. To see it use nvidia-smi topo -m shows SYS or PHB between GPUs in the pod, even though they’re on the same node and that leads to performance impact such as higher latency and cross‑socket PCIe contention, especially in collective operations.

The other problem may happen when GPUs are spread across multiple PCIe root complexes as on larger nodes (e.g., 8 GPUs), some GPUs may be wired under different PCIe root ports or PCIe switches resulting that even if they are on the same NUMA node, GPUs may be separated by multiple PCIe bridges (PXB), sharing a bottlenecked upstream link.

For step-by-step guide to troubleshoot GPU networking topology related to Kubernetes scheduling see [How PCIe, NVLink, and NUMA Topology Affect GPU Scheduling Outcomes](https://dev.to/daya-shankar/how-pcie-nvlink-and-numa-topology-affect-gpu-scheduling-outcomes-l52) and search web for common problems related to your hardware setup.
