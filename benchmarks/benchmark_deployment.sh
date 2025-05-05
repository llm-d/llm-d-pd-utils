#!/bin/bash

# Default values
CLEANUP=false
RDMA=false
YAML_SFX=""
TOTAL_NICS=0
DEST_IP_LIST=()      



THREADS=1
ROUNDS=10
# Parse arguments
for arg in "$@"; do
    case $arg in
        -c)
            CLEANUP=true
            ;;
        -rdma)
            RDMA=true
            YAML_SFX="_roce"
            ;;
    esac
done

# Check for required argument
if [ "$1" != "ucx" ] && [ "$1" != "nixl" ]; then
    echo "Usage: $0 [ucx|nixl] [-c] [-rdma]"
    exit 1
fi

if [ "$CLEANUP" = true ]; then
    echo "Cleaning up existing deployment"
    oc delete -f ../deployment/nixl_server${YAML_SFX}.yaml
    oc delete -f ../deployment/nixl_client${YAML_SFX}.yaml
    sleep 5
fi

if [ "$RDMA" = true ]; then
    UCX_TLS_OPTS="rc" # TODO: Enable gdr_copy
else 
    UCX_TLS_OPTS="tcp"
fi

if [ "$1" = "nixl" ]; then
    oc apply -f ../deployment/nixl_server${YAML_SFX}.yaml
    echo "Waiting for nixl-server to be ready"
    oc wait --for=condition=ready pod -l app=nixl-server --timeout=600s
    oc apply -f ../deployment/nixl_client${YAML_SFX}.yaml
    echo "Waiting for nixl-client to be ready"
    oc wait --for=condition=ready pod -l app=nixl-client --timeout=600s
    echo "Starting tests!!! with UCX_TLS=$UCX_TLS_OPTS"

    if [ "$RDMA" = true ]; then
        base_port=8000
        idx=0
        FILE=$(mktemp)

        DEST_IP_LIST=()
        oc exec nixl-server-roce -- sh -c 'for i in $(ls /sys/class/net | grep "^net1-"); do ip -4 addr show $i | grep -oP "(?<=inet\s)\d+(\.\d+){3}"; done' > "$FILE"

        while IFS= read -r line; do
            echo $line
            DEST_IP_LIST+=("$line")
            TOTAL_NICS=$((TOTAL_NICS + 1))
        done < "$FILE"
        rm "$FILE"
        echo "Found ${DEST_IP_LIST[@]} IPs and ${TOTAL_NICS} NICs"
        for ip in "${DEST_IP_LIST[@]}"; do
        port=$((base_port + idx * 1000))
            echo  oc exec nixl-server-roce -- sh -c "export UCX_TLS=$UCX_TLS_OPTS;export UCX_MAX_RNDV_RAILS=${TOTAL_NICS}; nohup python3 /workspace/benchmark.py --role peer --operation READ --host ${ip} --port ${port} --threads ${THREADS} --iters ${ROUNDS} &"
            oc exec nixl-server-roce -- sh -c "export UCX_TLS=$UCX_TLS_OPTS;export UCX_MAX_RNDV_RAILS=${TOTAL_NICS}; nohup python3 /workspace/benchmark.py --role peer --operation READ --host ${ip} --port ${port} --threads ${THREADS} --iters ${ROUNDS}  > /dev/null 2>&1 &"
            idx=$((idx + 1))
        done
        sleep 2
        idx=0
        for ip in "${DEST_IP_LIST[@]}"; do
            port=$((base_port + idx * 1000))
            (
                echo oc exec nixl-client-roce -- sh -c "export UCX_TLS=$UCX_TLS_OPTS;export UCX_MAX_RNDV_RAILS=${TOTAL_NICS}; python3 benchmark.py --role creator --operation READ --host ${ip} --port ${port} --thread ${THREADS} --iters ${ROUNDS} "
                oc exec nixl-client-roce -- sh -c "export UCX_TLS=$UCX_TLS_OPTS;export UCX_MAX_RNDV_RAILS=${TOTAL_NICS}; python3 benchmark.py --role creator --operation READ --host ${ip} --port ${port} --thread ${THREADS} --iters ${ROUNDS} "
            ) &
            pids+=($!)
            idx=$((idx + 1))
        done
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
    else
        oc exec nixl-server -- sh -c "export UCX_TLS=$UCX_TLS_OPTS; nohup python3 /workspace/benchmark.py --role peer --operation READ --host 0.0.0.0 --device cpu > /dev/null 2>&1 &"
        sleep 2
        oc exec nixl-client -- sh -c "export UCX_TLS=$UCX_TLS_OPTS; python3 benchmark.py --role creator --operation READ --host nixl-server --device cpu"
    fi
else
    oc apply -f ../deployment/nixl_server${YAML_SFX}.yaml
    echo "Waiting for nixl-server to be ready"
    oc wait --for=condition=ready pod -l app=nixl-server --timeout=600s
    oc apply -f ../deployment/nixl_client${YAML_SFX}.yaml
    echo "Waiting for nixl-client to be ready"
    oc wait --for=condition=ready pod -l app=nixl-client --timeout=600s
    echo "Starting tests!!!"

    if [ "$RDMA" = true ]; then
        oc exec nixl-server-roce -- sh -c "export UCX_TLS=$UCX_TLS_OPTS; nohup ucx_perftest -p 5555 > /dev/null 2>&1 &"
        sleep 2
        oc exec nixl-client-roce -- sh -c "export UCX_TLS=$UCX_TLS_OPTS; ucx_perftest -t tag_bw -s 1048576 -n 100000 -D zcopy -p 5555 nixl-server"
    else
        oc exec nixl-server -- sh -c "export UCX_TLS=$UCX_TLS_OPTS; nohup ucx_perftest -p 5555 > /dev/null 2>&1 &"
        sleep 2
        oc exec nixl-client -- sh -c "export UCX_TLS=$UCX_TLS_OPTS; ucx_perftest -t tag_bw -s 1048576 -n 100000 -D zcopy -p 5555 nixl-server"
    fi
fi



# oc exec nixl-server-roce -- sh -c "export UCX_TLS=rc;export UCX_MAX_RNDV_RAILS=2; nohup python3 /workspace/benchmark.py --role peer --operation READ --host 0.0.0.0 --threads 10"
# oc exec nixl-client-roce -- sh -c "export UCX_TLS=rc;export UCX_MAX_RNDV_RAILS=2; python3 benchmark.py --role creator --operation READ --host 10.241.130.7 --threads 10"

# oc exec -it nixl-server-roce -- sh -c "export UCX_TLS=rc;export UCX_MAX_RNDV_RAILS=2; export UCX_LOG_LEVEL=info; ucx_perftest -p 5555"
# oc exec nixl-client-roce -- sh -c "export UCX_TLS=rc; ucx_perftest -t tag_bw -s 1048576 -n 100000 -D zcopy -p 5556 10.241.130.7"