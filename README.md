# PD utils 

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


