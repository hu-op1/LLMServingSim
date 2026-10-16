#!/bin/bash
# Run a vLLM benchmark and write results under bench/results/<run_id>/.
#
# This is a thin host-side wrapper that:
#   1. Resolves the repo root.
#   2. Launches python -m bench run inside the vLLM Docker container
#      (or the local uv venv from scripts/install-vllm.sh).
#
# Edit the variables below for your run, then execute:
#
#     ./bench/bench.sh
#
# Every setting below is also an environment override, so a different
# model / hardware target needs no separate copy of this script:
#
#     MODEL=meta-llama/Llama-3.1-8B \
#     DATASET=workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
#     TP=1 MAX_NUM_SEQS=256 MAX_MODEL_LEN=32768 \
#     ./bench/bench.sh
#
# (that one is the RTX 4090 / Llama-3.1-8B run committed under
# bench/examples/RTX4090/Llama-3.1-8B/ -- MAX_MODEL_LEN is capped so the
# KV cache fits in 24 GB.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../bench
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# =============================================================================
# EDIT THESE
# =============================================================================
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DATASET="${DATASET:-workloads/agent_trace_test.jsonl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-bench/results/$RUN_ID}"

TP="${TP:-2}"
DP="${DP:-1}"
PP="${PP:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"   # blank => use the model default
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
SEED="${SEED:-42}"
# NB: LOAD_FORMAT is defined once, below with the other execution-mode knobs.
# The local fork carried a second copy here with default "dummy"; upstream's
# default is "auto". Pass LOAD_FORMAT=dummy to skip the weight download.
# vLLM load_format. "dummy" initializes weights randomly instead of reading a
# checkpoint. Use it for controlled diagnostics, not automatically equivalent
# ground truth: routing, backend choice and resolved cache capacity still need
# verification. Replay fixes input IDs and output lengths, not activations.
# The load format is recorded in meta.json.
LOAD_FORMAT="${LOAD_FORMAT:-auto}"
# Boot, write meta.json, exit -- no replay. Set to 1 to read back the one
# number a latency comparison depends on, kv_cache.num_gpu_blocks, which vLLM
# only settles at boot; duration depends on the model and hardware.
RESOLVE_ONLY="${RESOLVE_ONLY:-0}"
# Boot without a tokenizer. The replay never needs one, so what this buys is
# the ability to bench a checkpoint whose tokenizer is not on disk: point MODEL
# at a separate directory containing config.json and set LOAD_FORMAT=dummy
# for a config-only diagnostic. Unlike the profiler, bench does not create
# that directory from the repository's named JSON configs.
SKIP_TOKENIZER_INIT="${SKIP_TOKENIZER_INIT:-0}"
# Run vLLM eager. Not the production configuration -- what it buys is a truth
# in the same execution mode the profiler is forced into, which separates a
# cost-model error from the cudagraph speedup the simulator cannot see.
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
# Local override: sample scheduler state at 0.5s instead of upstream's 1.0s.
TICK_SECONDS="${TICK_SECONDS:-0.5}"
NUM_REQS="${NUM_REQS:-0}"            # 0 => replay the full dataset
LOG_LEVEL="${LOG_LEVEL:-INFO}"
NO_ENABLE_PREFIX_CACHING="${NO_ENABLE_PREFIX_CACHING:-0}"  # 1 = use --no-enable-prefix-caching

EXPERT_PARALLEL="${EXPERT_PARALLEL:-0}"   # 1 to enable for MoE

# =============================================================================
# CLEANUP
# =============================================================================

kill_gpu_processes() {
    echo "Cleaning up GPU processes..."
    pgrep -f "vllm serve" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    pgrep -f "VLLM::EngineCore" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 1
    echo "Cleanup done."
}

cleanup() {
    local exit_code=$?
    echo ""
    echo "Shutting down (exit code: $exit_code)..."
    kill_gpu_processes
    echo "All processes stopped."
    exit $exit_code
}
trap cleanup INT TERM EXIT

echo "Pre-start cleanup: checking for residual GPU processes..."
kill_gpu_processes

# =============================================================================
# EXECUTE
# =============================================================================
mkdir -p "$OUTPUT_DIR"

cmd=(python3 -m bench run
    --model "$MODEL"
    --dataset "$DATASET"
    --output-dir "$OUTPUT_DIR"
    --tensor-parallel-size "$TP"
    --data-parallel-size "$DP"
    --pipeline-parallel-size "$PP"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --dtype "$DTYPE"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --load-format "$LOAD_FORMAT"
    --seed "$SEED"
    --load-format "$LOAD_FORMAT"
    --tick-seconds "$TICK_SECONDS"
    --num-reqs "$NUM_REQS"
    --log-level "$LOG_LEVEL"
)

[[ -n "$MAX_MODEL_LEN" ]] && cmd+=(--max-model-len "$MAX_MODEL_LEN")
[[ "$EXPERT_PARALLEL" == "1" ]] && cmd+=(--enable-expert-parallel)
[[ "$NO_ENABLE_PREFIX_CACHING" == "1" ]] && cmd+=(--no-enable-prefix-caching)
[[ "$RESOLVE_ONLY" == "1" ]] && cmd+=(--resolve-only)
[[ "$SKIP_TOKENIZER_INIT" == "1" ]] && cmd+=(--skip-tokenizer-init)
[[ "$ENFORCE_EAGER" == "1" ]] && cmd+=(--enforce-eager)

echo "Running: ${cmd[*]}"
set +o pipefail  # allow capturing exit code from pipe
"${cmd[@]}" 2>&1 | tee "$OUTPUT_DIR/console.log"
BENCH_EXIT_CODE=${PIPESTATUS[0]}
set -o pipefail
echo "Done. Results in: $OUTPUT_DIR"
echo "  - meta.json"
echo "  - requests.jsonl"
echo "  - vllm.log   (vLLM scheduler / engine events, via Python logging)"
echo "  - console.log (raw stdout+stderr, includes C-level messages)"

exit $BENCH_EXIT_CODE