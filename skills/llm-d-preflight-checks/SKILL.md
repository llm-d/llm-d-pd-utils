---
name: llm-d-preflight-checks
description: Run preflight checks for LLM-D to ensure the environment is properly set up before vLLM is started.
---

Deploy llm-d with preflight checks built into the model server pods from the start using a custom kustomize overlay. This avoids the two-step deploy-then-patch approach and uses a single `kubectl apply -k` with one rollout.

The preflight script (`llm-d-preflight-checks.py`) runs before `vllm serve` via `&&`, so vLLM only starts if the script exits 0.

The script behavior is controlled by the `LLMD_PREFLIGHT_CHECKS` environment variable:

| Value | Behavior |
|-------|----------|
| unset / `disable` / `none` | Print system diagnostics (env, GPU, CPU, PCI) and exit 0 |
| `pause` | Print diagnostics, then start HTTP server blocking until `/exit` is called |
| `topology` | Print diagnostics and exit (reserved for future topology validation) |
| `nixl` | Print diagnostics and exit (reserved for future NixL checks) |

When in `pause` mode, the HTTP server satisfies K8s health probes and provides:
- `GET /health` — 200 OK (for probes)
- `GET /info` — system diagnostics
- `GET /exit` — shut down server and continue to vLLM startup

## Architecture

The llm-d deployment has two independent components deployed separately:

| Component | Deployed by | Controls |
|-----------|-------------|----------|
| Router (EPP + Envoy) | `helm install` (standalone chart) | Request routing / scheduling |
| Model server (vLLM pods) | `kubectl apply -k` (kustomize) | `vllm serve` command |

The helm chart does **not** control the `vllm serve` command — that is defined in kustomize patch files. Therefore, preflight checks are injected via a custom kustomize overlay on the model server, not by modifying the helm chart.

## Running preflight checks with llm-d quickstart

Follow the [llm-d quickstart guide](https://llm-d.ai/docs/getting-started/quickstart) but use a custom kustomize overlay that includes preflight checks from the start — a single `kubectl apply -k` deploys pods with preflight already wired in.

### Prerequisites

- A clone of the [llm-d repo](https://github.com/llm-d/llm-d)
- A clone of `llm-d-pd-utils` containing the preflight checks script at `skills/llm-d-preflight-checks/scripts/llm-d-preflight-checks.py`

### Step 1: Deploy the router via helm (unchanged)

```bash
cd /path/to/llm-d
export GAIE_VERSION=v1.5.0
export GUIDE_NAME="quickstart"
export NAMESPACE=<your-namespace>

# Install CRDs
kubectl apply -k "https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd?ref=${GAIE_VERSION}"
kubectl create namespace ${NAMESPACE}

# Deploy router
helm install ${GUIDE_NAME} \
    oci://registry.k8s.io/gateway-api-inference-extension/charts/standalone \
    -f guides/recipes/scheduler/base.values.yaml \
    -f guides/optimized-baseline/scheduler/optimized-baseline.values.yaml \
    -n ${NAMESPACE} --version ${GAIE_VERSION}
```

### Step 2: Create the preflight ConfigMap

```bash
kubectl create configmap llm-d-preflight-checks \
  --from-file=llm-d-preflight-checks.py=/path/to/llm-d-pd-utils/skills/llm-d-preflight-checks/scripts/llm-d-preflight-checks.py \
  -n ${NAMESPACE}
```

### Step 3: Create a custom kustomize overlay with preflight checks

Create a directory for the overlay (e.g., `guides/quickstart-preflight/modelserver/`):

```bash
mkdir -p guides/quickstart-preflight/modelserver
```

Create `guides/quickstart-preflight/modelserver/kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../optimized-baseline/modelserver/gpu/vllm/base/

patches:
  - path: patch-preflight.yaml
    target:
      kind: Deployment
```

Create `guides/quickstart-preflight/modelserver/patch-preflight.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: placeholder
spec:
  template:
    spec:
      containers:
        - name: modelserver
          command: ["bash", "-c"]
          args:
            - "python3 /preflight/llm-d-preflight-checks.py && vllm serve openai/gpt-oss-120b --disable-access-log-for-endpoints=/health,/metrics,/v1/models --tensor-parallel-size=2 --gpu-memory-utilization=0.95"
          env:
            - name: LLMD_PREFLIGHT_CHECKS
              value: "pause"
          volumeMounts:
            - name: preflight-checks
              mountPath: /preflight
      volumes:
        - name: preflight-checks
          configMap:
            name: llm-d-preflight-checks
            defaultMode: 493
```

Note: `defaultMode: 493` = octal `0755`, making the script executable.

### Step 4: Deploy model server with preflight (single step)

```bash
kubectl apply -n ${NAMESPACE} -k guides/quickstart-preflight/modelserver/
```

Pods come up already running preflight checks — no second patch or rollout needed.

### Step 5: Verify preflight checks are running

```bash
# Check pods are Running and Ready (preflight HTTP server satisfies probes)
kubectl get pods -n ${NAMESPACE} -l llm-d.ai/model

# Verify the preflight script started
kubectl logs -n ${NAMESPACE} <pod-name> | grep "llm-d-preflight-checks.py starting"

# Test the preflight HTTP endpoints
kubectl exec -n ${NAMESPACE} <pod-name> -- curl -s http://localhost:8000/health
# Returns: {"status":"ok"}

kubectl exec -n ${NAMESPACE} <pod-name> -- curl -s http://localhost:8000/info
# Returns: full system diagnostics (env, GPU topology, NVLink, lscpu)
```

### Step 6: Resume vLLM startup

```bash
for pod in $(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/model -o name); do
  kubectl exec -n ${NAMESPACE} $pod -- curl -s http://localhost:8000/exit
done
```

### Changing preflight mode without redeployment

```bash
kubectl set env deployment/<deploy-name> -n ${NAMESPACE} LLMD_PREFLIGHT_CHECKS=none
```

### Cleanup

```bash
kubectl apply -n ${NAMESPACE} -k guides/optimized-baseline/modelserver/gpu/vllm/base/
kubectl delete configmap llm-d-preflight-checks -n ${NAMESPACE}
```

## Running preflight checks with P/D disaggregation guide

Follow the [P/D disaggregation guide](https://github.com/llm-d/llm-d/tree/main/guides/pd-disaggregation) but use a custom kustomize overlay that includes preflight checks for both prefill and decode deployments from the start.

The P/D disaggregation guide deploys **two** model server deployments:

| | Prefill | Decode |
|---|---------|--------|
| Replicas | 8 | 2 |
| Tensor parallel | TP=1 | TP=4 |
| vLLM port | 8000 | 8200 |
| Pod label | `llm-d.ai/role=prefill` | `llm-d.ai/role=decode` |

### Prerequisites

- A clone of the [llm-d repo](https://github.com/llm-d/llm-d)
- A clone of `llm-d-pd-utils` containing the preflight checks script

### Step 1: Deploy the router via helm (unchanged)

```bash
cd /path/to/llm-d
export GAIE_VERSION=v1.5.0
export GUIDE_NAME="pd-disaggregation"
export NAMESPACE="llm-d-pd-disaggregation"

# Install CRDs
kubectl apply -k "https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd?ref=${GAIE_VERSION}"
kubectl create namespace ${NAMESPACE}

# Deploy router
helm install ${GUIDE_NAME} \
    oci://registry.k8s.io/gateway-api-inference-extension/charts/standalone \
    -f guides/recipes/scheduler/base.values.yaml \
    -f guides/${GUIDE_NAME}/scheduler/${GUIDE_NAME}.values.yaml \
    -n ${NAMESPACE} --version ${GAIE_VERSION}
```

### Step 2: Create the preflight ConfigMap

```bash
kubectl create configmap llm-d-preflight-checks \
  --from-file=llm-d-preflight-checks.py=/path/to/llm-d-pd-utils/skills/llm-d-preflight-checks/scripts/llm-d-preflight-checks.py \
  -n ${NAMESPACE}
```

### Step 3: Create a custom kustomize overlay with preflight checks

Create a directory for the overlay (e.g., `guides/pd-disaggregation-preflight/modelserver/`):

```bash
mkdir -p guides/pd-disaggregation-preflight/modelserver
```

Create `guides/pd-disaggregation-preflight/modelserver/kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../pd-disaggregation/modelserver/gpu/vllm/base/

patches:
  - path: patch-preflight-prefill.yaml
    target:
      kind: Deployment
      name: .*prefill.*
  - path: patch-preflight-decode.yaml
    target:
      kind: Deployment
      name: .*decode.*
```

Create `guides/pd-disaggregation-preflight/modelserver/patch-preflight-prefill.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: placeholder
spec:
  template:
    spec:
      containers:
        - name: modelserver
          command: ["bash", "-c"]
          args:
            - "python3 /preflight/llm-d-preflight-checks.py && vllm serve openai/gpt-oss-120b --disable-access-log-for-endpoints=/health,/metrics,/v1/models --tensor-parallel-size=1 --block-size=128 --kv-transfer-config '{\"kv_connector\":\"NixlConnector\", \"kv_role\":\"kv_both\"}' --no-disable-hybrid-kv-cache-manager --gpu-memory-utilization=0.9"
          env:
            - name: LLMD_PREFLIGHT_CHECKS
              value: "pause"
          volumeMounts:
            - name: preflight-checks
              mountPath: /preflight
      volumes:
        - name: preflight-checks
          configMap:
            name: llm-d-preflight-checks
            defaultMode: 493
```

Create `guides/pd-disaggregation-preflight/modelserver/patch-preflight-decode.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: placeholder
spec:
  template:
    spec:
      containers:
        - name: modelserver
          command: ["bash", "-c"]
          args:
            - "python3 /preflight/llm-d-preflight-checks.py && vllm serve openai/gpt-oss-120b --disable-access-log-for-endpoints=/health,/metrics,/v1/models --tensor-parallel-size=4 --block-size=128 --kv-transfer-config '{\"kv_connector\":\"NixlConnector\", \"kv_role\":\"kv_both\"}' --no-disable-hybrid-kv-cache-manager --port=8200"
          env:
            - name: LLMD_PREFLIGHT_CHECKS
              value: "pause"
            - name: VLLM_INFERENCE_PORT
              value: "8200"
          volumeMounts:
            - name: preflight-checks
              mountPath: /preflight
      volumes:
        - name: preflight-checks
          configMap:
            name: llm-d-preflight-checks
            defaultMode: 493
```

The `VLLM_INFERENCE_PORT=8200` env var tells the preflight script to bind its HTTP server on port 8200, which is the port K8s health probes target on decode pods.

### Step 4: Deploy model server with preflight (single step)

```bash
kubectl apply -n ${NAMESPACE} -k guides/pd-disaggregation-preflight/modelserver/
```

Both prefill and decode pods come up already running preflight checks — no second patch or rollout needed.

### Step 5: Verify preflight checks are running

```bash
# Check prefill pods
kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=prefill

# Check decode pods
kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=decode

# Verify preflight started on a prefill pod (port 8000)
kubectl logs -n ${NAMESPACE} <prefill-pod> | grep "llm-d-preflight-checks.py starting"
kubectl exec -n ${NAMESPACE} <prefill-pod> -- curl -s http://localhost:8000/health

# Verify preflight started on a decode pod (port 8200)
kubectl logs -n ${NAMESPACE} <decode-pod> | grep "llm-d-preflight-checks.py starting"
kubectl exec -n ${NAMESPACE} <decode-pod> -- curl -s http://localhost:8200/health
```

### Step 6: Resume vLLM startup

Resume prefill and decode separately:

```bash
# Resume all prefill pods
for pod in $(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=prefill -o name); do
  kubectl exec -n ${NAMESPACE} $pod -- curl -s http://localhost:8000/exit
done

# Resume all decode pods
for pod in $(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=decode -o name); do
  kubectl exec -n ${NAMESPACE} $pod -- curl -s http://localhost:8200/exit
done
```

### Changing preflight mode without redeployment

```bash
# Switch prefill to non-blocking mode
kubectl set env deployment/pd-disaggregation-nvidia-gpu-vllm-prefill \
  -n ${NAMESPACE} LLMD_PREFLIGHT_CHECKS=none

# Switch decode to non-blocking mode
kubectl set env deployment/pd-disaggregation-nvidia-gpu-vllm-decode \
  -n ${NAMESPACE} LLMD_PREFLIGHT_CHECKS=none
```

### Cleanup

```bash
kubectl apply -n ${NAMESPACE} -k guides/pd-disaggregation/modelserver/gpu/vllm/base
kubectl delete configmap llm-d-preflight-checks -n ${NAMESPACE}
```

## How it works

The kustomize overlay modifies the pod spec as follows:

| Original | With preflight overlay |
|----------|----------------------|
| `command: ["vllm", "serve"]` | `command: ["bash", "-c"]` |
| `args: ["openai/gpt-oss-120b", ...]` | `args: ["python3 /preflight/llm-d-preflight-checks.py && vllm serve openai/gpt-oss-120b ..."]` |
| No ConfigMap volume | ConfigMap `llm-d-preflight-checks` mounted at `/preflight` (mode 0755) |
| No preflight env vars | `LLMD_PREFLIGHT_CHECKS=pause` |

The `bash -c` wrapper allows the `&&` chain: preflight runs first, and only if it exits 0 does vLLM start.

In `pause` mode, the preflight HTTP server binds to the same port vLLM would use (8000 for prefill, 8200 for decode via `VLLM_INFERENCE_PORT`), so K8s startup/liveness/readiness probes pass while vLLM is not yet running. Once `/exit` is called, the server releases the port and vLLM binds to it normally.

### Port selection

The script defaults to port 8000 (or `VLLM_INFERENCE_PORT` if set). If the port is in use, it auto-increments (8001, 8002, ...). When unpausing pods after a restart, check `/proc/net/tcp` to find the actual listening port — it may not be 8000.

## Running preflight checks with llm-d-benchmark

The [llm-d-benchmark](https://github.com/llm-d/llm-d-benchmark) framework can run the preflight checks script automatically before vLLM starts. The framework's step 04 (`04_ensure_model_namespace_prepared.py`) reads all files from `setup/preprocess/` into a ConfigMap named `llm-d-benchmark-preprocesses`, which gets mounted at `/setup/preprocess/` inside pods.

### Step 1: Symlink the script into llm-d-benchmark

Create a symlink from the llm-d-benchmark `setup/preprocess/` directory to the preflight checks script:

```bash
ln -s /path/to/llm-d-pd-utils/skills/llm-d-preflight-checks/scripts/llm-d-preflight-checks.py \
      /path/to/llm-d-benchmark/setup/preprocess/llm-d-preflight-checks.py
```

This ensures the script is included in the `llm-d-benchmark-preprocesses` ConfigMap when step 04 runs.

### Step 2: Update the scenario PREPROCESS variable

In your scenario file (e.g., `scenarios/guides/pd-disaggregation2.sh`), set the `LLMDBENCH_VLLM_COMMON_PREPROCESS` variable to include the preflight checks script:

```bash
export LLMDBENCH_VLLM_COMMON_PREPROCESS="python3 /setup/preprocess/set_llmdbench_environment.py; source \$HOME/llmdbench_env.sh; python3 /setup/preprocess/llm-d-preflight-checks.py"
```

The script runs after `set_llmdbench_environment.py` and `source llmdbench_env.sh` so that environment variables like `VLLM_INFERENCE_PORT` are available.

### Step 3: Set LLMD_PREFLIGHT_CHECKS in the scenario

Add `LLMD_PREFLIGHT_CHECKS` to the pod environment variables in the scenario's `LLMDBENCH_VLLM_COMMON_ENVVARS_TO_YAML` block:

```bash
export LLMDBENCH_VLLM_COMMON_ENVVARS_TO_YAML=$(mktemp)
cat << EOF > $LLMDBENCH_VLLM_COMMON_ENVVARS_TO_YAML
- name: NCCL_EXCLUDE_IB_HCA
  value: "mlx5_0,mlx5_2,mlx5_4,mlx5_8,mlx5_7,mlx5_10,mlx5_12,mlx5_14,mlx5_16"
- name: NVSHMEM_DEBUG
  value: "INFO"
- name: LLMD_PREFLIGHT_CHECKS
  value: "pause"
EOF
```

### Step 4: Run the standup

```bash
source venv/bin/activate
export LLMDBENCH_HF_TOKEN=<your-token>
export LLMDBENCH_DEPLOY_MODEL_LIST="facebook/opt-125m"

# Teardown any previous deployment
./setup/teardown.sh -c ${PWD}/scenarios/guides/pd-disaggregation2.sh

# Run standup steps 0-9
./setup/standup.sh -v -c ${PWD}/scenarios/guides/pd-disaggregation2.sh -s 0-9
```

### How it works end-to-end

1. **Step 04** reads all files in `setup/preprocess/` (including the symlinked `llm-d-preflight-checks.py`) and creates the `llm-d-benchmark-preprocesses` ConfigMap in the target namespace.

2. **Scenario volume config** mounts the ConfigMap at `/setup/preprocess` inside pods:
   ```yaml
   volumes:
   - name: preprocesses
     configMap:
       defaultMode: 0755
       name: llm-d-benchmark-preprocesses
   volumeMounts:
   - name: preprocesses
     mountPath: /setup/preprocess
   ```

3. **Pod startup command** chains the preflight script before vllm via `&&`:
   ```
   python3 /setup/preprocess/set_llmdbench_environment.py; \
   source $HOME/llmdbench_env.sh; \
   python3 /setup/preprocess/llm-d-preflight-checks.py && \
   vllm serve <model> --port $VLLM_INFERENCE_PORT ...
   ```

4. **Preflight script** checks `LLMD_PREFLIGHT_CHECKS` env var and either prints diagnostics and continues (default), or blocks with an HTTP server (`pause` mode) until `/exit` is called.

### Verifying the preflight output

```bash
kubectl logs -n <namespace> <pod-name> -c vllm | grep -A 50 "llm-d-preflight-checks.py starting"
```
