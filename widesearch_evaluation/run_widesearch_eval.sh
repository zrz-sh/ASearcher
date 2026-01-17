MODEL_PATH=/mnt/mnt/public/zhangruize/MAS/ckpt/ASearcher-Web-7B
DATA_PATH=/mnt/mnt/public/zhangruize/MAS/data/widesearch/widesearch.jsonl

AGENT_TYPE=asearcher
PROMPT_TYPE=asearcher
SEARCH_CLIENT_TYPE=async-web-search-access

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

OUTPUT_DIR=${SCRIPT_DIR}/result

PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH" \
CUDA_VISIBLE_DEVICES=0 \
TOKENIZERS_PARALLELISM=false \
python3 widesearch_eval.py \
    --model_name_or_path ${MODEL_PATH}  \
    --output_dir ${OUTPUT_DIR} \
    --data_path ${DATA_PATH} \
    --prompt_type $PROMPT_TYPE \
    --agent-type ${AGENT_TYPE} \
    --search-client-type ${SEARCH_CLIENT_TYPE} \
    --tensor_parallel_size 1 \
    --temperature 0.6 \
    --seed 3 \
    --use-jina \
