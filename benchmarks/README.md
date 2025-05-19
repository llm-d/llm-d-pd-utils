# Scripts to run benchmark on a two node environment

## Run the UCX/NIXL benchmark on two OCP pods

```bash
./benchmark_deployment.sh [ucx|nixl] [-c] [-rdma]
```

## Run the UCX performance test on two node VM/bare-metal

Open terminal 1 on node 1:

```bash
UCX_TLS=cuda_copy,tcp ucx_perftest
```
Open terminal 2 on node 2:

```bash
UCX_TLS=cuda_copy,tcp ucx_perftest -t tag_bw -s 524288 -n 100000 <NODE1_IP>
```

Change the data
## Run the NIXL benchmark on two nodes

Open terminal 1 on node 1 :
```bash
python3 benchmark.py --role peer --operation READ --host <NODE1_IP> --device cuda
```

Open terminal 2 on node 2 :
```bash
benchmark.py --role creator --operation READ --host <NODE1_IP> --device cuda
```

# Available options for benchmark.py
```plaintext
--operation: READ | WRITE 
--device: cpu | cuda | cuda:<device_id>
--num-blocks: <int>, default 100
--num-layers: <int>, default 32
--block-size: <int>, default 256
--hidden-size: <int>, default 1024
--dtype: bfloat16 
--threads: <int>, default 1
--iters: <int>, default 1
```
