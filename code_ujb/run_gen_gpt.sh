MODEL_PATH=$1
DATASET=codeujbcomplete
GEN_MOD=complete
NUM_SAMPLES=10
BATCH_SIZE=1
TEMPERATURE=1.0

IDS_FILE="../../constants/ids_train_val_test.json"
JSON_CONTENT=$(cat "$IDS_FILE")

python generate_hf_gpt.py \
    --model-path $MODEL_PATH \
    --model-id $MODEL_PATH \
    --gen-mode $GEN_MOD \
    --bench-name $DATASET \
    --num-samples $NUM_SAMPLES \
    --batch-size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --save-generations-path ./generations/openai--$MODEL_PATH/${MODEL_PATH}-UJB.json \
    --all_ids_dict "$JSON_CONTENT" \
    --split test # train, val, test, or all

