DEVICE=$1
MODEL_PATH=$2
DATASET=codeujbcomplete
GEN_MOD=complete
NUM_SAMPLES=2
BATCH_SIZE=1
TEMPERATURE=1.0
IFS='/' read -r MODEL_FAMILY MODEL_NAME <<< "$MODEL_PATH"
GENERATED_BY="${MODEL_PATH//\//--}" # replace / with --

IDS_FILE="../../constants/ids_train_val_test.json"
JSON_CONTENT=$(cat "$IDS_FILE")

CUDA_VISIBLE_DEVICES=$DEVICE python generate_hf.py \
    --model-path $MODEL_PATH \
    --model-id $MODEL_NAME \
    --gen-mode $GEN_MOD \
    --bench-name $DATASET \
    --num-samples $NUM_SAMPLES \
    --batch-size $BATCH_SIZE \
    --temperature $TEMPERATURE \
    --save-generations-path ./generations/$MODEL_FAMILY--$MODEL_NAME/${GENERATED_BY}-UJB.json \
    --all_ids_dict "$JSON_CONTENT" \
    --split all # train, val, test, or all

