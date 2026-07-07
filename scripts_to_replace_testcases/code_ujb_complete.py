# ./code_ujb/tasks/code_ujb_complete.py
import os
import random
import re
import signal
import string
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import chardet
import javalang
import numpy as np
from Task import Task, clean_signature
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class StreamStopUJBComplete():
    def __init__(self, function_signature, mode="complete"):
        self.function_signature = function_signature
        self.mode = mode
    
    def check_stop(self, generation):
        if self.mode == "complete":
            generation = self.function_signature + "{\n" + generation
        elif self.mode == "chat":
            if not self.function_signature in generation:
                return False
            generation = generation.split(self.function_signature)
            generation = generation[1]
            
        block_count, in_block, in_double_quote, in_single_quote = 0, False, False, False
        for char_idx in range(len(generation)):
            if generation[char_idx] == '"': in_double_quote = not in_double_quote
            if generation[char_idx] == "'": in_single_quote = not in_single_quote
            if generation[char_idx] == "{" and (not in_double_quote): 
                block_count += 1
                in_block = True
            if generation[char_idx] == "}" and (not in_double_quote): 
                block_count -= 1
            if block_count == 0 and in_block:
                return True
        return False

class CodeUJBComplete(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
    DATASET_PATH = "ZHENGRAN/code_ucb_complete"

    def __init__(self):
        super().__init__(
            stop_words=["/**", "/**\n", "public", "private", "protected",  
                        "\t/**", "\t/**\n", "\tpublic", "\tprivate", "\tprotected"],
            requires_execution=False,
        )
        print("Using Dataset:", self.DATASET_PATH)
        self.dataset = load_dataset(self.DATASET_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")


    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["train"]

    def get_prompt(self, doc, mode="complete"):
        """Builds the prompt for the LM to generate from."""
        if mode == "complete":
            prompt_key = "prompt_complete"
        elif mode == "chat":
            prompt_key = "prompt_chat"
        else:
            raise KeyError()
        return doc[prompt_key].strip()
    
    def get_prompt_byidx(self, idx, mode="complete"):
        """Builds the prompt for the LM to generate from."""
        return self.get_prompt(self.get_dataset()[idx], mode=mode)

    def get_id_byidx(self, idx):
        """Builds the prompt for the LM to generate from."""
        return self.get_dataset()[idx]["task_id"]
    
    def get_stream_stop(self, idx, mode="complete"):
        return StreamStopUJBComplete(self.get_dataset()[idx]["function_signature"], mode=mode)
    
    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return doc["function"]

    @staticmethod
    def _stop_at_function(generation):
        block_count, in_block, in_double_quote, in_single_quote = 0, False, False, False
        char_idx = 0
        for char_idx in range(len(generation)):
            if generation[char_idx] == '"': in_double_quote = not in_double_quote
            if generation[char_idx] == "'": in_single_quote = not in_single_quote
            if generation[char_idx] == "{" and (not in_double_quote): 
                block_count += 1
                in_block = True
            if generation[char_idx] == "}" and (not in_double_quote): 
                block_count -= 1
            if block_count == 0 and in_block:
                break
        if char_idx:
            generation = generation[:char_idx+1]
        return generation

    def postprocess_complete_generations(self, generations, idx):
        return [self.postprocess_complete_generation(gen, idx) for gen in generations]
    
    def postprocess_chat_generations(self, generations, idx):
        return [self.postprocess_chat_generation(gen, idx) for gen in generations]
        
    def postprocess_complete_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        # prompt = self.get_prompt(self.dataset["train"][idx])
        prompt_with_comment = self.dataset["train"][idx]["prompt_complete_with_comment"]
        # print("prompt", prompt_with_signature)
        # print("generation", generation)
        generation = generation[len(prompt_with_comment):]
        # generation = self._stop_at_function(generation)
        return generation

    def postprocess_chat_generation(self, generation, idx):
        signature = self.dataset["train"][idx]["function_signature"].strip()
        
        pre_signature, sub_signature = clean_signature(signature)
        # if not clean_code(signature) in clean_code(generation):
        if not sub_signature in generation:
            # print(signature[-1])
            # if idx == 2:
            # print(signature)
            # print(pre_signature, sub_signature)
            # print(generation)
            # exit()
            print("Can not find target function in answer!")
            return "Can not find target function in answer!\n\n"+generation
        generation = generation.split(sub_signature)
        # if len(generation) != 2:
        #     print("Multiple target function in answer!")
        #     return "Multiple target function in answer!\n\n"+generation
        generation = generation[1]
        function = self._stop_at_function(generation)
        
        generation = pre_signature + sub_signature +  function
        return generation
        
    
    def evaluate(self, generations):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        
        all_tasks = []
        results = {"total": 0, "pass_syntax": {"count": 0}, "pass_compile": {"count": 0}, 
                   "pass_trigger": {"count": 0}, "pass_all": {"count": 0}, "timed_out": 0, "detail": {}}
        total_tokens_dict = {}
        for generation in tqdm(generations, total=len(generations)):
            idx = int(generation["task_idx"])
            # gens = generation["outputs"]
            gens = generation["patch"]
            prompt = generation["prompt"]
            function_name = generation["function_name"]
            generated_by = generation["generated_by"]
            test_statement = generation["test_statement"]
            test_method_name = generation["test_method_name"]
            test_idx = generation["test_idx"]
            test_execution_idx = generation["test_execution_idx"]
            
            # inps = generation["inputs"]
            # rawouts = generation["raw_outputs"]
            # inps_tokens = sum([len(tokens) for tokens in self.tokenizer.batch_encode_plus(inps, return_tensors="np")['input_ids']])
            # rawouts_tokens = sum([len(tokens) for tokens in self.tokenizer.batch_encode_plus(rawouts, return_tensors="np")['input_ids']])
            # total_tokens_dict[idx] = inps_tokens * 0.5 + rawouts_tokens
            
            project = self.dataset["train"][idx]["project"]
            bug_id = self.dataset["train"][idx]["bug_id"]
            testmethods = self.dataset["train"][idx]["testmethods"]
            # testmethods = testmethods[:1]
            source_dir = self.dataset["train"][idx]["source_dir"]
            start = self.dataset["train"][idx]["start"]
            end = self.dataset["train"][idx]["end"]
            location = self.dataset["train"][idx]["location"]
            source = self.dataset["train"][idx]["source"]
            
            one_tasks = [(idx, igen, generated_by, prompt, function_name, gen, project, bug_id, testmethods, source_dir, 
                      start, end, location, source, test_statement, test_method_name, test_idx, test_execution_idx) for igen, gen in enumerate(gens)]
            all_tasks.extend(one_tasks)

                    
        with ProcessPoolExecutor(max_workers=os.cpu_count()//4) as executor:
            # Submit all your tasks to the executor
            future_tasks = set()
            for task in all_tasks:
                future_tasks.add(executor.submit(validate_all_patches, task))
                time.sleep(0.01)
            # Use tqdm to display progress
            all_bug_results_list = []
            with tqdm(as_completed(future_tasks), total=len(all_tasks), desc="Evaluating all tasks...") as progress_bar:
                for future in progress_bar:
                    # Append the result to a list
                    all_bug_results_list.append(future.result())
        
        return sorted(all_bug_results_list, key=lambda x: (x["task_idx"], x["sample_idx"]))
    
def get_pass_at_k(n, c, k):
    if n - c < k : return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def read_file(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
    encoding = chardet.detect(content)['encoding']
    decoded_content = content.decode(encoding)
    return decoded_content

def save_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def extract_test_source(input_str, tmp_folder_path):
    if '::' in input_str:
        path_end = input_str.split('::')[0]
        path_end = path_end.replace('.', '/') + '.java'
    elif input_str.strip().startswith('at '):
        path_end = input_str.split('at ')[-1].split('(')[0].split('.')
        path_end = os.path.join(*path_end[:-1]) + '.java'
    else: 
        return None, None
    
    paths_to_test_file_to_attempt = [
        os.path.join(tmp_folder_path, 'src', 'test', 'java', path_end),
        os.path.join(tmp_folder_path, 'gson', 'src', 'test', 'java', path_end),
        os.path.join(tmp_folder_path, 'src', 'test', path_end),
        os.path.join(tmp_folder_path, 'tests', path_end),
        os.path.join(tmp_folder_path, 'test', path_end),
    ]
    for path in paths_to_test_file_to_attempt:
        if os.path.exists(path):
            source_test = read_file(path)
            return source_test, path
    return None, None

def extract_error_details(exec_feedback, failing_test, source, patch, tmp_folder_path, project, bug_id, file_name_function_name):
    # default values for the 7 error info fields
    error_line_number = -1
    failed_test_input = "-"
    failed_test_expected_output = "-"
    failed_test_actual_output = "-"
    error_message = "-"
    error_type = "-"
    error_line_code = "-"
    full_log_messages = []
    call_graph = '-'
    call_graph_tests = '-'

    if exec_feedback.startswith('Compilation error--') and '.java:' in exec_feedback:
        # we need: error_line_number, error_message, error_type, error_line_code
        error_line_number = int(exec_feedback.split('.java:')[-1].split(':')[0].strip())
        error_message = exec_feedback.split(': error:')[-1].strip()
        error_type = "SyntaxError"
        if 0 < error_line_number <= len(source.splitlines()):
            error_line_code = source.splitlines()[error_line_number - 1].strip()
        
        
    elif exec_feedback.startswith('Test Failed ('):

        # extract the source code of the test file that failed
        list_of_failed_tests = exec_feedback.split("cycle)--[")[-1].split("]")[0]
        list_of_failed_tests = list_of_failed_tests.split('b\'  - ')
        list_of_failed_tests = [x.strip(' ,\'')[:-2] for x in list_of_failed_tests][1:]

        assert len(list_of_failed_tests) > 0
        for failed_test_idx, failed_test in enumerate(list_of_failed_tests):
            source_test, path_to_test_file = extract_test_source(failed_test, tmp_folder_path)
            if source_test: break

        # extract the full log message for the failed test(s)
        full_log_messages = failing_test.split("---")[1:]
        assert len(full_log_messages) == len(list_of_failed_tests)

        full_log_message = full_log_messages[failed_test_idx]
        if len(full_log_message.splitlines()) > 1:
            error_message = full_log_message.splitlines()[1].strip() # the error message is in the second line of the (sub)log
            test_method_throwing_error = full_log_message.splitlines()[0].split("::")[-1].strip()
        
            test_file_throwing_error = path_to_test_file.split("/")[-1].split(".java")[0] if path_to_test_file else "-"
            call_graph, call_graph_tests = "-", "-"
            # as a first attempt, go through the sublog and look for a line containing
            # the full TEST FILE name and a line number (example like: (TestClass.java:45))
            # if it finds it extract that line (the line that failed) from the test source code
            # see ../documentation/log_example1.txt for a concrete example
            for line in full_log_message.split("\n"):
                pattern = re.compile(rf'\({re.escape(test_file_throwing_error)}\.java:(\d+)\)')
                match = pattern.search(line)
                if match:
                    error_line_number = int(match.group(1))
                    if source_test and 0 < error_line_number <= len(source_test.split("\n")):
                        error_line_code = source_test.split("\n")[error_line_number - 1].strip()
                        break

            # as a second attempt, try to match the right line of the log by using the name of the TEST METHOD
            # in this case the search in the log is different.
            # We look for the name of the method in the first line of the log.
            # see ../documentation/log_example2.txt for a concrete example
            if error_line_number == -1:
                test_method_throwing_error = full_log_messages[failed_test_idx].splitlines()[0].split("::")[-1]
                call_graph, call_graph_tests = "-", "-"
                for line in full_log_message.split("\n"):
                    if test_method_throwing_error in line and '.java:' in line:
                        pattern = re.compile(r'\.java:(\d+)\)')
                        match = pattern.search(line)
                        if match:
                            source_test, path_to_test_file = extract_test_source(line, tmp_folder_path)
                            error_line_number = int(match.group(1))
                            if source_test and 0 < error_line_number <= len(source_test.split("\n")):
                                error_line_code = source_test.split("\n")[error_line_number - 1].strip()
                                break
    
        # set error_type. If there is an assertion error set the type to "OutputMismatch"
        # in this case also extract the failed test expected output and actual output
        patterns = [
            r'expected:(.+) but was:(.+)',
            r'expected same:(.+) was not:(.+)',
        ]
        for patt in patterns:
            pattern = re.compile(patt, re.IGNORECASE)
            match = pattern.search(error_message)
            if match:
                failed_test_expected_output = match.group(1).strip()
                failed_test_actual_output = match.group(2).strip()
                error_type = "OutputMismatch"
                break
        else:
            error_type = error_message.split(":")[0].strip()

    return error_line_number, failed_test_input, failed_test_expected_output, failed_test_actual_output, error_message, error_type, error_line_code, full_log_messages, call_graph, call_graph_tests


def validate_all_patches(item):
    idx, igen, generated_by, prompt, function_name, patch, project, bug_id, testmethods, source_dir, start, end, location, source, test_statement, test_method_name, test_idx, test_execution_idx = item
    def generate_random_string(length):
        characters = string.ascii_letters + string.digits  # 包含大写字母、小写字母和数字
        random_string = ''.join(random.choice(characters) for _ in range(length))
        return random_string

    tmp_folder = f"{project}-{bug_id}-" + generate_random_string(16)
    tmp_folder_path = os.path.join('/tmp/ujb', tmp_folder)
    subprocess.run(['rm', '-rf', tmp_folder_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    cmd = ['defects4j', 'checkout', '-p', project, '-v', str(bug_id) + 'f', '-w', tmp_folder_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # print(f"CHECKOUT {idx}: {' '.join(cmd)}")

    raw_source = source
    source = source.split("\n")
    patch = patch.split("\n")
    source = "\n".join(source[:start] + patch + source[end+1:])
    source_test = None
    path_to_test_file = None
    failed_test_idx = -1

    save_file(os.path.join(tmp_folder_path, location), source)
    
    compile_fail, timed_out, bugg, entire_bugg, syntax_error, exec_feedbacks, failing_tests = run_d4j_test(source, testmethods, tmp_folder_path, test_statement, test_method_name)
    
    # exec_feedback can be either "Compilation error", "Test Failed (first or second cycle)", "Timeout" or "Correct solution"

    error_line_number = []
    failed_test_input = []
    failed_test_expected_output = []
    failed_test_actual_output = []
    error_message = []
    error_type = []
    error_line_code = []
    full_log_messages = []
    call_graph = []
    call_graph_tests = []

    file_name_function_name = "{}:{}".format(location.split("/")[-1].split(".java")[0], function_name)
    
    for ef, ft in zip(exec_feedbacks, failing_tests):
        eln, fti, fteo, ftao, em, et, elc, flm, cg, cgt = \
            extract_error_details(ef, ft, source, '\n'.join(patch), tmp_folder_path, project, bug_id, file_name_function_name)
        error_line_number.append(eln)
        failed_test_input.append(fti)
        failed_test_expected_output.append(fteo)
        failed_test_actual_output.append(ftao)
        error_message.append(em)
        error_type.append(et)
        error_line_code.append(elc)
        full_log_messages.append(flm)
        call_graph.append(cg)
        call_graph_tests.append(cgt)
        
    subprocess.run(['rm', '-rf', tmp_folder_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    solution_idx = f"Java--UJB--TaskID::{idx:03d}--GeneratedBy::{generated_by}--SampleID::{igen:02d}"
    final_instruction = f"// You are a professional Java programmer, please create a function named `{function_name}` based on the provided abstract Java class context information and the following natural language annotations."
    if final_instruction in prompt:
        final_prompt_part = prompt.split(final_instruction)[-1].strip()
        pattern = r'(/\*\*.*?\*/)'
        match = re.search(pattern, final_prompt_part, re.DOTALL)
        if match:
            description = match.group(1).strip()
            signature = final_prompt_part.split('*/')[-1].strip()
        else:
            description = "NEGATIVOOO NEIN"
            signature = "NEGATIVOOO NEIN"
    else:
        description = "NEGATIVOOO NEIN"
        signature = "NEGATIVOOO NEIN"

    return {
            "solution_idx" : solution_idx,
            "task_idx": f"{function_name}-{idx:03d}",
            "sample_idx": igen,
            "generated_by": generated_by,
            "prompt": prompt,
            "description": description,
            "signature": signature,
            "method": '\n'.join(patch),
            "is_pass": entire_bugg,
            "test_idx": test_idx,
            "test_execution_idx": test_execution_idx,
            "test_statement": test_statement,
            "test_method_name": test_method_name,
            # "project": project,
            # "bug_id":  bug_id,
            # "pass_syntax": not syntax_error,
            # "pass_compile": not compile_fail,
            # "pass_trigger": not bugg,
            # "timed_out": timed_out,
            # "test_source": source_test,
            # "source": source,
            
            "error_line_number": str(error_line_number), 
            "failed_test_input": str(failed_test_input),
            "failed_test_expected_output": str(failed_test_expected_output),
            "failed_test_actual_output": str(failed_test_actual_output),
            "error_message": str(error_message),
            "error_type": str(error_type),
            "error_line_code": str(error_line_code),
            
            # "full_log" : str(full_log_messages),
            "call_graph": str(call_graph),
            # "call_graph_tests": str(call_graph_tests),
            "exec_feedback": str(exec_feedbacks),
            "location": location,
            # "path_to_test_file": path_to_test_file,
            # "failed_test_idx": failed_test_idx,
        }
    
def run_d4j_test(source, testmethods, tmp_folder_path, test_statement, test_method_name):
    bugg = False
    compile_fail = False
    timed_out = False
    entire_bugg = "-"
    exec_feedbacks = []
    failing_tests = []

    # DON'T DO SYNTAX CHECK
    # try:
    #     tokens = javalang.tokenizer.tokenize(source)
    #     parser = javalang.parser.Parser(tokens)
    #     parser.parse()
    # except:
    #     # print("Syntax Error")
    #     return True, False, True, True, True, 'Syntax error'

    ####################################################
    # HERE IS WHERE THE MAGIC HAPPENS
    # we locate the original test file, we overwrite it with the isolated and wrapped test statement
    # me overwrite the list of testmethods to execute to contain only the isolated test method
    # then we execute the test.

    # look for the right path of the test file
    in_the_middles = [
        "gson/src/test/java",
        "src/test/java",
        "src/test",
        "test",
        "tests",
    ]
    in_the_middle = ""
    last_part_and_test_file = testmethods[0].split("::")[0].replace('.', '/') + '.java'
    for itm in in_the_middles:
        try_path = os.path.join(tmp_folder_path, itm, last_part_and_test_file)
        if os.path.exists(try_path):
            in_the_middle = itm
            break

    if in_the_middle == "":
        raise Exception("Could not find the test file to modify for test execution.")
    test_file_path = os.path.join(tmp_folder_path, in_the_middle, last_part_and_test_file)
    # overwrite the defect4j test file with the isolated and wrapped test statement that we want to execute.
    save_file(test_file_path, test_statement)
    # overwrite the tests to be executed with only the test method we want to execute.
    testmethods = ["{}::{}".format(testmethods[0].split("::")[0], test_method_name)]
    #################################################### 

    for t in testmethods:
        # print(f"Running test: {t.strip()}")
        cmd = 'defects4j test -w %s/ -t %s' % ((tmp_folder_path), t.strip())
        Returncode = ""
        error_file = open("/tmp/ujb/stderr.txt", "wb")
        # print(cmd)
        child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=error_file, bufsize=-1,
                                start_new_session=True)
        while_begin = time.time()
        while True:
            Flag = child.poll()
            if Flag == 0:
                Returncode = child.stdout.readlines()  # child.stdout.read()
                # print(b"".join(Returncode).decode('utf-8'))
                error_file.close()
                exec_feedbacks.append(f'Test Failed (first cycle)--{Returncode}')
                break
            elif Flag != 0 and Flag is not None:
                exec_feedbacks.append('Compilation error--')
                compile_fail = True
                error_file.close()
                with open("/tmp/ujb/stderr.txt", "rb") as f:
                    r = f.readlines()
                for line in r:
                    if re.search(':\serror:\s', line.decode('utf-8')):
                        exec_feedbacks[-1] = exec_feedbacks[-1] + f'{line.decode("utf-8")}'
                        break
                # print("error_string", error_string)
                break
            elif time.time() - while_begin > 120:
                exec_feedbacks.append('Test timed out--')
                error_file.close()
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                timed_out = True
                break
            else:
                time.sleep(0.01)
        log = Returncode
        if len(log) > 0 and log[-1].decode('utf-8') == "Failing tests: 0\n":
            exec_feedbacks[-1] = "The solution passes all the tests!"
            failing_tests.append("No failing tests.")
            continue
        else:
            bugg = True
            if os.path.exists(os.path.join(tmp_folder_path, "failing_tests")):
                with open(os.path.join(tmp_folder_path, "failing_tests"), "r", encoding='utf-8') as f:
                    failing_tests.append(f.read())
            else:
                failing_tests.append("No failing_tests log found.")
            # break # this break should be removed to check all testmethods

    if bugg or compile_fail:
        entire_bugg = 'fail'
    elif timed_out:
        entire_bugg = 'timeout'
    else:
        entire_bugg = 'pass'

    assert len(exec_feedbacks) == len(failing_tests), f"Number of execution feedbacks and log messages do not match: {len(exec_feedbacks)} vs {len(failing_tests)}"
    return compile_fail, timed_out, bugg, entire_bugg, False, exec_feedbacks, failing_tests
