dataset=codeujbcomplete
gen_mode="complete"
GENERATIONS_PATH=$1
OUTPUT_PATH="${GENERATIONS_PATH/test-4-execution/test-execution}"

python3 evaluate.py \
    --gen-mode $gen_mode \
    --bench-name $dataset \
    --load-generations-path $GENERATIONS_PATH \
    --eval-output-path $OUTPUT_PATH