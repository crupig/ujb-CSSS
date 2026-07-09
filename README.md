# UJB

CoderUJB (Unified Java Benchmark): A new benchmark designed to evaluate LLMs across diverse Java programming tasks that are executable and reflective of actual development scenarios, acknowledging Java’s prevalence in real-world software production.

## Contents
- [Install](#install)
- [CodeUJB](#mt-bench)

## Install
1. Install codeujb.

    ```
    # create a new conda environment
    conda create -n ujb python=3.10
    conda activate ujb

    # clone and install codeujb
    git clone https://github.com/WisdomShell/ujb.git
    cd ujb
    pip install -e .
    ```
    For more details packages version, please refer to `requirements.txt`.

2. Refer to [defects4j](https://github.com/rjust/defects4j) repository for install execution environment.


## CodeUJB

### Evaluate a model on CodeUJB

#### Step 1. Generate model answers to CodeUJB questions
We support three backbones for generating CodeUJB answers: `hf`, `openai` and `tgi`.
```
# generate answers with huggingface `transformers` backbone.
python code_ujb/generate_hf.py \
    --model-path $model_name_or_path \
    --model-id $run_id \
    --gen-mode $gen_mode \
    --bench-name $dataset \
    --num-samples $num_samples \
    --save-generations-path ./log/$run_id/$dataset/generations-$gen_mode.json 

```

```
# generate answers with openai API backbone.

export OPENAI_API_BASE=''
export OPENAI_API_KEY=''

python code_ujb/generate_api.py \
    --model-path $run_id \
    --model-id $run_id \
    --gen-mode $gen_mode \
    --bench-name $dataset \
    --num-samples $num_samples \
    --parallel 8 \
    --save-generations-path ./log/$run_id/$dataset/generations-$gen_mode.json 
```

```
# If `model-id` not in OpenAI model list, `generate_api.py` will generate answers with Text Generation Inference backbone.
# Please refer to [Text Generation Inference](https://github.com/huggingface/text-generation-inference) for deploying your TGI server first.

export TGI_API_URL_${run_id//-/_}=http://127.0.0.1:8081,http://127.0.0.1:8082 # The Text Generation Inference API URL.

python code_ujb/generate_api.py \
    --model-path $run_id \
    --model-id $run_id \
    --gen-mode $gen_mode \
    --bench-name $dataset \
    --num-samples $num_samples  \
    --parallel 32 \
    --save-generations-path ./log/$run_id/$dataset/generations-$gen_mode.json 
```
Arguments:
  - `[model-path]` is the path to the weights, which can be a local folder or a Hugging Face repo ID. If you using `generate_api.py`, it should be the same as model ID.
  - `[model-id]` is a name you give to the model.
  - `[gen-mode]` have two options: `complete` for model without instruction-finetuning and `chat` for model with instruction-finetuning.
  - `[bench-name]` is the name of the dataset you want to evaluate. There five datasets in CodeUJB: `codeujbrepair`, `codeujbcomplete`, `codeujbtestgen`, `codeujbtestgenissue`, `codeujbdefectdetection`.
  - `[num-samples]` is the number of samples for each coding question you want to generate.
  - `[save-generations-path]` is the path to save the generated answer.
  - `[parallel]` is the number of parallel API calls.
e.g.,

```
python code_ujb/generate_api.py --model-path gpt-3.5-turbo --model-id gpt-3.5-turbo --gen-mode chat --bench-name codeujbcomplete --num-samples 10 --save-generations-path log/gpt-3.5-turbo/codeujbcomplete/generations-chat.jsonl
```
The answers will be saved to `log/gpt-3.5-turbo/codeujbcomplete/generations-chat.jsonl`.


#### Step 2. Evaluation model answers of CodeUJB
Please make sure you have installed `defects4j` first.
```
python3 code_ujb/evaluate.py \
    --model-path $model_name_or_path \
    --model-id $run_id \
    --gen-mode $gen_mode \
    --bench-name $dataset \
    --num-samples $num_samples \
    --load-generations-path ./log/$run_id/$dataset/generations-$gen_mode.json \
    --eval-output-path ./log/$run_id/$dataset/evaluation-$gen_mode.json
```
Arguments:
  - `[load-generations-path]` is the path to the generated answer.
  - `[eval-output-path]` is the path to save the evaluation results.

e.g.,
```
python code_ujb/evaluate.py --model-path gpt-3.5-turbo --model-id gpt-3.5-turbo --gen-mode chat --bench-name codeujbcomplete --num-samples 10 --load-generations-path log/gpt-3.5-turbo/codeujbcomplete/generations-chat.jsonl --eval-output-path ./log/gpt-3.5-turbo/codeujbcomplete/evaluation-chat.json
```
The evaluation results will be saved to `./log/gpt-3.5-turbo/codeujbcomplete/evaluation-chat.json`

## Notes for replication of "Comparative Study of Selection Strategies"
This repo was created as a support generation tool for the work **"How Should We Rank LLM Code Generations? A Comparative Study of  Selection Strategies"**. 

The original code generation framework has been modified in order to:
* extract the log-probabilities when generating code solutions;
* extracting test execution feedbacks when evaluting the generated solutions;
* generate test cases instead of code solutions (for the CodeT approach);
* run the generated test cases against the previously generated code solutions.

### To setup:
* clone the repo;
* create virtual environment;
* install `requirements.txt` (designed to work on Python `3.10.19`)

### Code generation:

```bash run_gen.sh <DEVICE_ID> <MODEL_PATH>```

For example:

```bash run_gen.sh 0 Qwen/Qwen2.5-Coder-3B-Instruct```

**Evaluation:**

To run evaluation for the CoderUJB benchmark.

- clone this repo and the defect4j repo in the same folder.
- switch to Java 11 with: `export JAVA_HOME=$(/usr/libexec/java_home -v11)`
- initialize defect4j: `cd defect4j` and run `./init.sh`
- Add Defects4J's executables to your PATH with: `export PATH=$PATH:"path2defects4j"/framework/bin`
- go to ujb/code_ujb and activate virtual environment:
- run: `run_eval.sh`

Example of command to run the scripts (input file as argument):

```bash run_eval.sh ./path_to_generation_file.json```

### Test cases generation (CodeT approach):

**Replace scripts**:

Replace all the scripts in `scripts_to_replace_testcases`. Each file to be replace starts with the path where it has to be replaced.

```bash run_gen_tests.sh <DEVICE_ID> <MODEL_PATH>```

For example:

```bash run_gen_tests.sh 0 Qwen/Qwen2.5-Coder-3B-Instruct```

**To execute tests:**

The models are in charge of generating test suites with multiple (up to 10) test cases (methods). Therefore, we have to postprocess the output of the models in order to isolate each tst method into a separate test file. This is done in `knowlbase_tests.py`.

Afterwards, within each model (generator) and each coding problem (task id), we have to run each generated code solution against each generated test case.

`merge_generations_and_tests_b4_test_exec.py` creates and saves all the `<code_solution, test_statement>` pairs.

After generating all the pairs, the tests can be run with:

```bash run_eval_tests.sh ./path_to_generation_file.json```
