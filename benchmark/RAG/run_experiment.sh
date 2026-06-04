#!/bin/bash
set -e

DATASET="${1:-hotpotqa}"
BUILD_COUNT="${2:-1}"
STEP="${3:-gen+eval}"

CONFIG_DIR="config/${DATASET}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Map dataset name to config file prefix (e.g. legalbench_contractnli -> legalbench)
if [[ "$DATASET" == legalbench_* ]]; then
    CONFIG_PREFIX="legalbench"
else
    CONFIG_PREFIX="$DATASET"
fi

resolve_output_dir() {
    local config_path="$1"
    python3 -c "
import yaml, os
c = yaml.safe_load(open('$config_path'))
dn = c.get('dataset_name', '')
tk = c.get('execution', {}).get('retrieval_topk', 5)
sl = c.get('vikingbot', {}).get('search_limit', '')
mi = c.get('vikingbot', {}).get('max_iterations', '')
od = c.get('paths', {}).get('output_dir', '').format(dataset_name=dn, retrieval_topk=tk, search_limit=sl, max_iterations=mi)
print(os.path.join('$(pwd)', od))
"
}

is_completed() {
    local config_path="$1"
    local output_dir
    output_dir=$(resolve_output_dir "$config_path")
    local report="${output_dir}/benchmark_metrics_report.json"
    if [ ! -f "$report" ]; then
        return 1
    fi
    python3 -c "import json; d=json.load(open('$report')); exit(0 if 'Performance Metrics' in d else 1)"
}

has_checkpoint() {
    local config_path="$1"
    local output_dir
    output_dir=$(resolve_output_dir "$config_path")
    [ -f "${output_dir}/benchmark_checkpoint.json" ]
}

run_step() {
    local config_path="$1"
    local label="$2"

    if is_completed "$config_path"; then
        echo "[SKIP] $label (already completed)"
        return 0
    fi

    local resume_flag=""
    if has_checkpoint "$config_path"; then
        resume_flag="--resume"
        echo "[RESUME] $label (checkpoint found)"
    else
        echo "[RUN] $label"
    fi

    python run.py --config "$config_path" --step gen+eval $resume_flag
}

echo "=========================================="
echo " OpenViking Relations Experiment"
echo " Dataset: $DATASET"
echo " Build count: $BUILD_COUNT"
echo " Step: $STEP"
echo "=========================================="

echo ""
run_step "${CONFIG_DIR}/${CONFIG_PREFIX}_bot_config.yaml" \
    "[1/3] Bot baseline"

echo ""
for i in $(seq 1 "$BUILD_COUNT"); do
    run_step "${CONFIG_DIR}/${CONFIG_PREFIX}_bot_config_build_links_review.yaml" \
        "[2/3] Bot build_links_review round $i/$BUILD_COUNT"
done

echo ""
run_step "${CONFIG_DIR}/${CONFIG_PREFIX}_bot_config_relations_review.yaml" \
    "[3/3] Bot relations_review"

echo ""
echo "=========================================="
echo " Experiment complete!"
echo "=========================================="


