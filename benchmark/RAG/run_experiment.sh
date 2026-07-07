#!/bin/bash
set -e

DATASET="versionrag"
BUILD_COUNT="1"
STEP="gen+eval"
RUN_IMPORT="false"
FORCE_RESUME="0"

# 兼容位置参数和命名参数混合解析
positional_count=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            FORCE_RESUME=1
            shift
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --run-import|--run_import)
            RUN_IMPORT="true"
            shift
            ;;
        --help|-h)
            echo "用法: run_experiment.sh [DATASET] [BUILD_COUNT] [STEP] [RUN_IMPORT]"
            echo "                     [--step <step>] [--resume] [--run-import]"
            echo ""
            echo "  DATASET     数据集名 (默认 versionrag)"
            echo "  BUILD_COUNT build_links_review 轮数 (默认 1)"
            echo "  STEP        all|import|gen|eval|gen+eval|del (默认 gen+eval)"
            echo "  RUN_IMPORT  true|false|import (默认 false)"
            echo "  --resume     对所有步骤加 --resume（从 checkpoint 继续）"
            echo "  --step       覆盖 STEP 参数"
            echo "  --run-import 运行 vector store import"
            exit 0
            ;;
        *)
            positional_count=$((positional_count + 1))
            case "$positional_count" in
                1) DATASET="$1" ;;
                2) BUILD_COUNT="$1" ;;
                3) STEP="$1" ;;
                4) RUN_IMPORT="$1" ;;
            esac
            shift
            ;;
    esac
done

CONFIG_DIR="config/${DATASET}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Map dataset name to config file prefix (e.g. legalbench_contractnli -> legalbench)
if [[ "$DATASET" == legalbench_* ]]; then
    CONFIG_PREFIX="legalbench"
elif [[ "$DATASET" == financebench_* ]]; then
    CONFIG_PREFIX="financebench"
elif [[ "$DATASET" == versionrag_* ]]; then
    CONFIG_PREFIX="versionrag"
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
rtk = c.get('vikingbot', {}).get('relations_topk', '')
rsim = c.get('vikingbot', {}).get('relations_similarity_threshold', '')
od = c.get('paths', {}).get('output_dir', '').format(dataset_name=dn, retrieval_topk=tk, search_limit=sl, max_iterations=mi, relations_topk=rtk, relations_similarity_threshold=rsim)
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

should_run_import() {
    [[ "$RUN_IMPORT" =~ ^(1|true|yes|y|import)$ ]]
}

run_import_once() {
    local config_path="$1"

    if ! should_run_import && [[ "$STEP" != "import" ]]; then
        return 0
    fi

    echo ""
    echo "[IMPORT] Shared vector store from baseline config: $config_path"
    python run.py --config "$config_path" --step import
}

run_step() {
    local config_path="$1"
    local label="$2"
    local step_to_run="$STEP"

    local resume_flag=""
    if [[ "$FORCE_RESUME" == "1" ]]; then
        if is_completed "$config_path"; then
            echo "[SKIP] $label (already completed)"
            return 0
        fi
        resume_flag="--resume"
        echo "[RESUME] $label"
    else
        echo "[RUN] $label"
    fi

    if should_run_import && [[ "$STEP" == "all" ]]; then
        step_to_run="gen+eval"
    fi

    python run.py --config "$config_path" --step "$step_to_run" $resume_flag
}

echo "=========================================="
echo " OpenViking Relations Experiment"
echo " Dataset: $DATASET"
echo " Build count: $BUILD_COUNT"
echo " Step: $STEP"
echo " Run import: $RUN_IMPORT"
echo "=========================================="

BASE_CONFIG="${CONFIG_DIR}/${CONFIG_PREFIX}_bot_config.yaml"

run_import_once "$BASE_CONFIG"

if [[ "$STEP" == "import" ]]; then
    echo ""
    echo "=========================================="
    echo " Import complete!"
    echo "=========================================="
    exit 0
fi

echo ""
for i in $(seq 1 "$BUILD_COUNT"); do
    run_step "${CONFIG_DIR}/${CONFIG_PREFIX}_bot_config_build_links_review.yaml" \
        "[1/2] Bot build_links_review round $i/$BUILD_COUNT"
done

echo ""
run_step "${CONFIG_DIR}/${CONFIG_PREFIX}_bot_config_relations_review.yaml" \
    "[2/2] Bot relations_review"

echo ""
echo "=========================================="
echo " Experiment complete!"
echo "=========================================="
