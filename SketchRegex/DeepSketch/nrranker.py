import argparse
import pathlib
import os
import logging
import exrex
import re
from external.regexDFAEquals import unprocess_regex, silent_regex_equiv
import signal
import openai
import pickle
import multiprocessing
from typing import List, Dict, Tuple, Set, Optional
import random

openai.api_key = os.getenv('OPENAI_API_KEY')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_timieout(signum, frame):
    raise TimeoutError("Timed out!")
signal.signal(signal.SIGALRM, handle_timieout)

def _parse_args():
    parser = argparse.ArgumentParser(description='nrranker')
    parser.add_argument('--dataset', type=str, default="Turk", help='specified dataset')
    parser.add_argument('--model_id', type=str, default="pretrained-MLE", help='specified model id')
    parser.add_argument('--split', type=str, default='test', help='test split')
    parser.add_argument('-d', type=int, default=5, help='number of differentiating test inputs per pair')
    parser.add_argument('-t', type=int, default=5, help='number of true test cases')
    parser.add_argument('--time_out', type=int, default=3, help='time_out for test inputs generation')
    parser.add_argument('--max_str_len', type=int, default=50, help='max string length for a test input')
    parser.add_argument('--request_openai', action='store_true', help='request openai for test oracle generation')
    parser.add_argument('--rank', action='store_true', help='Actually do the ranking')
    args = parser.parse_args()
    args.output_dir = os.path.join('decodes/', args.dataset, '{}-{}'.format(args.split, args.model_id))
    args.NL_path = os.path.join('datasets/', args.dataset, 'src-{}.txt'.format(args.split))
    args.ground_truth_path = os.path.join('datasets/', args.dataset, 'targ-{}.txt'.format(args.split))
    args.cache_dir = os.path.join('caches/', args.dataset, '{}-{}'.format(args.split, args.model_id))
    return args

class TestCase:

    def __init__(self, test_input: str, test_output: str, test_oracle: str = None) -> None:
        self.test_input = test_input
        self.test_output = test_output
        self.test_oracle = test_oracle

    def __str__(self) -> str:
        return "Input: {}, Output: {}, Oracle: {}".format(self.test_input, self.test_output, self.test_oracle)
    
    def __repr__(self) -> str:
        return self.__str__()

class Candidate:

    def __init__(self, idx: int, dsl_regex: str, posix_regex: str) -> None:
        """
        Args:
            idx (int): original rank of the candidate
            dsl_regex (str): regex in DSL form
            posix_regex (str): regex in POSIX form
        """
        self.idx = idx
        self.idx_after_drop = None
        self.dsl = dsl_regex
        self.posix = posix_regex
        self.test_cases = []
        self.cluster_test_cases = None
        self.score = 0
        self.exact_match = False
        self.dfa_equiv = False
        

class NRRanker:

    def __init__(self, NL: str, ground_truth: str, filepath: str, args) -> None:
        """
        Args:
            NL : natural language description
            ground_truth : ground truth regex
            filepath (str): path to file containing candidates
            args: arguments
        """
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.args = args
        self.candidates = []
        self.NL = NL
        self.ground_truth = Candidate(-1, ground_truth, unprocess_regex(ground_truth))
        self.few_shot_examples = []
        self.foreword = ''
        # Save the test input and oracle to reduce the number of API calls
        self.test_input_oracle = {}


    def load_and_drop_invalid(self):
        with open(self.filepath) as f:
            for i, x in enumerate(f.readlines()):
                x = x.rstrip().split()
                if len(x) < 2:
                    logging.debug("Empty line")
                    continue
                x = x[1]
                y = unprocess_regex(x)
                try:
                    re.compile(y)
                    self.candidates.append(Candidate(i, x, y))
                except:
                    # If the candidate is invalid, we simply drop it. This means the 
                    # number of candidates may be less than the number of candidates in the file.
                    logging.debug("Invalid regex: {}".format(x))

    def drop_duplicates(self):
        i = 0
        while i < len(self.candidates):
            j = i + 1
            while j < len(self.candidates):
                if silent_regex_equiv(self.candidates[i].posix, self.candidates[j].posix) != "false":
                    self.candidates.pop(j)
                else:
                    j += 1
            i += 1
        logging.debug("After dropping duplicates, {} candidates left".format(len(self.candidates)))

    def generate_random_test_inputs(self):
        for c in range(len(self.candidates)):
            self.candidates[c].idx_after_drop = c
            self.candidates[c].cluster_test_cases = [[] for _ in range(len(self.candidates))]

        for c1 in range(len(self.candidates)):
            test_inputs = []
            for c2 in range(len(self.candidates)):
                if c1 != c2:
                    regex1 = self.candidates[c1].posix
                    regex2 = self.candidates[c2].posix
                    logging.debug("Generating test cases for candidates {} and {}, {} and {}".format(c1, c2, regex1, regex2))
                    pattern = re.compile(regex2)
                    for i in range(self.args.d):
                        signal.alarm(self.args.time_out)
                        try:
                            test_input = exrex.getone(regex1, limit=3)
                            count = 0
                            while (test_input in test_inputs) or (len(test_input) > self.args.max_str_len): # No need for differentiating
                                test_input = exrex.getone(regex1, limit=3)
                                count += 1
                                # count is added because mysterious bugs will nullificate signal.alarm
                                if count >= 100000:
                                    raise TimeoutError
                            # fullmatch might be slow, so we put this in a try-except block
                            test_output_regex2 = str(pattern.fullmatch(test_input) is not None)
                        except TimeoutError as e:
                            logging.debug("Timeout")
                            break
                        finally:
                            signal.alarm(0)
                        test_inputs.append(test_input) 
                        logging.debug("Generated test input: {}".format(test_input))
                        # Add the test input but leave the oracle empty
                        self.test_input_oracle[test_input] = None

                        testcase = TestCase(test_input, 'True')
                        self.candidates[c1].cluster_test_cases[c2].append(testcase)
                        self.candidates[c1].test_cases.append(testcase)
                        testcase = TestCase(test_input, test_output_regex2)
                        self.candidates[c2].cluster_test_cases[c1].append(testcase)
                        self.candidates[c2].test_cases.append(testcase)


    def generate_diff_test_inputs(self):
        for c in range(len(self.candidates)):
            self.candidates[c].idx_after_drop = c
            self.candidates[c].cluster_test_cases = [[] for _ in range(len(self.candidates))]

        for c1 in range(len(self.candidates)):
            test_inputs = []
            for c2 in range(len(self.candidates)):
                if c1 != c2:
                    regex1 = self.candidates[c1].posix
                    regex2 = self.candidates[c2].posix
                    logging.debug("Generating test cases for candidates {} and {}, {} and {}".format(c1, c2, regex1, regex2))
                    pattern = re.compile(regex2)
                    for i in range(self.args.d):
                        signal.alarm(self.args.time_out)
                        try:
                            test_input = exrex.getone(regex1, limit=3)
                            count = 0
                            while (test_input in test_inputs) or (len(test_input) > self.args.max_str_len) or pattern.fullmatch(test_input):
                                test_input = exrex.getone(regex1, limit=3)
                                count += 1
                                # count is added because mysterious bugs will nullificate signal.alarm
                                if count >= 100000:
                                    raise TimeoutError
                        except TimeoutError as e:
                            logging.debug("Timeout")
                            break
                        finally:
                            signal.alarm(0)
                        test_inputs.append(test_input) 
                        logging.debug("Generated test input: {}".format(test_input))
                        # Add the test input but leave the oracle empty
                        self.test_input_oracle[test_input] = None

                        testcase = TestCase(test_input, 'True')
                        self.candidates[c1].cluster_test_cases[c2].append(testcase)
                        self.candidates[c1].test_cases.append(testcase)
                        testcase = TestCase(test_input, 'False')
                        self.candidates[c2].cluster_test_cases[c1].append(testcase)
                        self.candidates[c2].test_cases.append(testcase)

        
    def generate_few_shot_examples(self, test_output: str = 'True'):

        regexes = [candidate.posix for candidate in self.candidates]
        compiled_regexes = [re.compile(regex) for regex in regexes]
        if test_output == 'False':
            regexes = ['(?!' + candidate.posix + ').*' for candidate in self.candidates]

        generated_strings = []
        for regex in regexes:
            signal.alarm(self.args.time_out * 3)
            while True:
                try:
                    if len(generated_strings) >= self.args.t:
                        signal.alarm(0)
                        break

                    # Sample one string from one regex that is not in the few-shot examples
                    test_input = exrex.getone(regex, limit=3)
                    while test_input in generated_strings or len(test_input) > self.args.max_str_len:
                        test_input = exrex.getone(regex, limit=3)

                    # Check if the string can match all the regexes
                    flag = True
                    for pattern in compiled_regexes:
                        if test_output == 'True':
                            if not pattern.fullmatch(test_input):
                                flag = False
                                break
                        elif test_output == 'False':
                            if pattern.fullmatch(test_input):
                                flag = False
                                break
                    
                    if flag:
                        generated_strings.append(test_input)
                        self.few_shot_examples.append(TestCase(test_input, test_output))
                        signal.alarm(self.args.time_out * 3)
                except TimeoutError as e:
                    logging.debug("generate few-shot examples timeout")
                    break
        # If we cannot generate any string, we simply generate an empty string
        if len(generated_strings) == 0:
            if test_output == 'True':
                for testcase in self.candidates[0].test_cases:
                    if testcase.test_output == 'True':
                        self.few_shot_examples.append(testcase)
                        break
            elif test_output == 'False':
                for testcase in self.candidates[-1].test_cases:
                    if testcase.test_output == 'False':
                        self.few_shot_examples.append(testcase)
                        break
        logging.debug("test_output: {}, Generated {}".format(test_output, generated_strings))
    

    def generate_prompt(self):
        self.foreword = 'Decide whether the input is a string that "{}". Fill in each <mask> with a python list of "True" or "False" to indicate whether the input is valid or not.\n'.format(self.NL)
        self.foreword += 'Examples:\n'
        for testcase in self.few_shot_examples:
            self.foreword += '"{}" -> <mask>\n'.format(testcase.test_input)
        self.foreword += 'Answer: ['
        for testcase in self.few_shot_examples:
            self.foreword += '"{}",'.format(testcase.test_output)
        self.foreword += ']\n'
        self.foreword += 'Real tests:\n'

        # self.foreword = 'def check(input_string):\n'
        # self.foreword += '\t"""\n\tDecide whether the input is a string that "{}"\n\t"""\n\tpass\n'.format(self.NL)
        # self.foreword += 'def test_check():\n'
        # for testcase in self.few_shot_examples:
        #     self.foreword += '\tassert("{}") == {}\n'.format(testcase.test_input, testcase.test_output)

        logging.debug("Generated prompt foreword: {}".format(self.foreword))

    # def generate_test_oracles(self):
    #     key_list = list(self.test_input_oracle.keys())
    #     key_list = [key for key in key_list if self.test_input_oracle[key] is None]
    #     step = len(self.few_shot_examples)

    #     # We generate step test oracles at a time
    #     for i in range(0, len(key_list), step):
    #         prompt = '' + self.foreword
    #         for j in range(i, min(i + step, len(key_list))):
    #             prompt += '"{}" -> <mask>\n'.format(key_list[j])
    #         prompt += 'Answer: '

    #         logging.debug('prompt: {}'.format(prompt))

    #         while True:
    #             try:
    #                 completion = openai.ChatCompletion.create(
    #                 model="gpt-3.5-turbo",
    #                 messages=[
    #                     {"role": "user", "content": prompt}
    #                 ],
    #                 temperature=0.6,
    #                 max_tokens=50,
    #                 )
    #             except Exception as e:
    #                 logging.info("Exception: {}".format(e))

    #             content = re.findall('(True)|(False)', completion.choices[0].message.content)
    #             content = [e[0] if e[0] != '' else e[1] for e in content]
    #             logging.info("openai return: {}".format(content))
    #             try:
    #                 for j in range(i, min(i + step, len(key_list))):
    #                     self.test_input_oracle[key_list[j]] = content[j - i]
    #             except Exception as e:
    #                 logging.info("Exception: {}".format(e))
    #             break
            
    #     for candidate in self.candidates:
    #         for testcase in candidate.test_cases:
    #             testcase.test_oracle = self.test_input_oracle[testcase.test_input]  

    def generate_test_oracles(self):
        key_list = list(self.test_input_oracle.keys())
        key_list = [key for key in key_list if self.test_input_oracle[key] is None]

        for test_input in key_list:
            prompt = '' + self.foreword
            prompt += '\tassert("{}") == '.format(test_input)

            logging.debug('prompt: {}'.format(prompt))

            while True:
                try:
                    completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=10,
                    )
                except Exception as e:
                    logging.info("Exception: {}".format(e))

                content = re.findall('(True)|(False)', completion.choices[0].message.content)
                content = [e[0] if e[0] != '' else e[1] for e in content]
                logging.info("openai return: {}".format(content))
                try:
                    self.test_input_oracle[test_input] = content[0]
                except Exception as e:
                    logging.info("Exception: {}".format(e))
                break
            
        for candidate in self.candidates:
            for testcase in candidate.test_cases:
                testcase.test_oracle = self.test_input_oracle[testcase.test_input] 

    def compare_with_ground_truth(self):

        for candidate in self.candidates:
            candidate.exact_match = (candidate.posix == self.ground_truth.posix)
            candidate.dfa_equiv = (silent_regex_equiv(candidate.posix, self.ground_truth.posix) != "false")

        
    def get_scores(self):
        """
        Get the score of each candidate
        """
        for candidate in self.candidates:
            for testcase in candidate.test_cases:
                if testcase.test_output == testcase.test_oracle:
                    candidate.score += 1
            if len(candidate.test_cases) != 0:
                candidate.score /= len(candidate.test_cases)

    def rank(self):
        """
        Rank candidates by their scores and stores them back to the file
        """
        self.get_scores()
        self.candidates.sort(key=lambda x: x.score, reverse=True)

    def eval(self, topk: int = 1) -> Tuple[bool, bool]:
        """
        Evaluate the top-k accuracy. Both exact match and DFA equivalance.

        Args:
            topk: the top-k candidates to evaluate, k in range [1, 20]
        
        Returns:
            exact_match: whether one of the top-k candidates have exact match
            dfa_equiv: whether one of the top-k candidates have DFA equivalence
        """
        exact_match = False
        dfa_equiv = False
        for i in range(min(topk, len(self.candidates))):
            exact_match = exact_match or self.candidates[i].exact_match
            dfa_equiv = dfa_equiv or self.candidates[i].dfa_equiv
        return exact_match, dfa_equiv

    def generate_test_oracles_proxy(self):
        """
        Generate test oracles using a proxy LLM
        """
        accuracy = 0.8
        compiled_ground_truth = re.compile(self.ground_truth.posix)
        
        for test_input in self.test_input_oracle:
            true_oracle = compiled_ground_truth.fullmatch(test_input) is not None
            if (random.uniform(0, 1) > accuracy):
                test_oracle =  str(not true_oracle)
            else:
                test_oracle =  str(true_oracle)
            self.test_input_oracle[test_input] = test_oracle


        for candidate in self.candidates:
            for testcase in candidate.test_cases:
                testcase.test_oracle = self.test_input_oracle[testcase.test_input]


def generate_test_inputs(filepath):
    """
    Generate test inputs for a single file. 
    Setup everything in Nrranker except for the candidate.test_case.test_oracle, which 
    requires OpenAI API calls.
    """
    logger.info('pid: {}, processing file: {}'.format(os.getpid(), filepath))
    filename = os.path.basename(filepath)
    cache_filepath = os.path.join(args.cache_dir, filename)
    if not os.path.exists(cache_filepath):

        nrranker = NRRanker(NLs[int(filename) - 1], ground_truths[int(filename) - 1], filepath, args)
        nrranker.load_and_drop_invalid()
        nrranker.drop_duplicates()

        nrranker.compare_with_ground_truth()

        nrranker.generate_diff_test_inputs()
        # nrranker.generate_random_test_inputs()

        nrranker.generate_few_shot_examples(test_output='True')
        nrranker.generate_few_shot_examples(test_output='False')

        nrranker.generate_prompt()

        with open(cache_filepath, 'wb') as f:
            pickle.dump(nrranker, f)




if __name__ == '__main__':
    args = _parse_args()
    with open(args.NL_path) as f:
        NLs = f.read().splitlines()
    with open(args.ground_truth_path) as f:
        ground_truths = f.read().splitlines()

    all_filepaths = list(pathlib.Path(args.output_dir).glob('*'))

    # args.rank = True
    # args.request_openai = True
    if args.rank:
        # label accuracy
        tot_test_cases = 0
        tot_correct_test_cases = 0

        # em and dfa
        original_em = 0
        original_dfa = 0
        ranked_em = 0
        ranked_dfa = 0

        # hjh
        ratio = 0

        for filepath in all_filepaths:
            logging.info('processing file: {}'.format(filepath))
            filename = os.path.basename(filepath)
            cache_filepath = os.path.join(args.cache_dir, filename)
            with open(cache_filepath, 'rb') as f:
                nrranker = pickle.load(f)

            # label accuracy
            correct_test_cases = 0
            pattern = re.compile(nrranker.ground_truth.posix)
            for test_input, test_oracle in nrranker.test_input_oracle.items():
                if str(pattern.fullmatch(test_input) is not None) == test_oracle:
                    correct_test_cases += 1
            print('correct_test_cases: {}'.format(correct_test_cases))
            print('tot_test_cases: {}'.format(len(nrranker.test_input_oracle)))
            print('label accuracy: {:.3f}'.format(correct_test_cases / len(nrranker.test_input_oracle)))
            tot_test_cases += len(nrranker.test_input_oracle)
            tot_correct_test_cases += correct_test_cases

            # ratio
            if correct_test_cases / len(nrranker.test_input_oracle) >= 0.7:
                ratio += 1
            
            # em and dfa
            exact_match, dfa_equiv = nrranker.eval()
            if exact_match:
                original_em += 1
            if dfa_equiv:
                original_dfa += 1
            nrranker.rank()
            exact_match, dfa_equiv = nrranker.eval()
            if exact_match:
                ranked_em += 1
            if dfa_equiv:
                ranked_dfa += 1
        
        # label accuracy
        print('tot_test_cases: {}'.format(tot_test_cases))
        print('tot_correct_test_cases: {}'.format(tot_correct_test_cases))
        print('label accuracy: {:.3f}'.format(tot_correct_test_cases / tot_test_cases))

        # ratio
        print('ratio: {:.3f}'.format(ratio / len(all_filepaths)))

        # em and dfa
        original_em /= len(all_filepaths)
        original_dfa /= len(all_filepaths)
        ranked_em /= len(all_filepaths)
        ranked_dfa /= len(all_filepaths)
        print('original_em: {:.3f}, original_dfa: {:.3f}, ranked_em: {:.3f}, ranked_dfa: {:.3f}'.format(original_em, original_dfa, ranked_em, ranked_dfa))
    else:
        if args.request_openai:
            for filepath in all_filepaths:
                logger.info('Processing file: {}'.format(filepath))
                filename = os.path.basename(filepath)
                cache_filepath = os.path.join(args.cache_dir, filename)
                with open(cache_filepath, 'rb') as f:
                    nrranker = pickle.load(f)

                nrranker.generate_prompt()
                nrranker.generate_test_oracles()
                # nrranker.generate_test_oracles_proxy()
                with open(cache_filepath, 'wb') as f:
                    pickle.dump(nrranker, f)
        else:
            os.makedirs(args.cache_dir, exist_ok=True)
            with multiprocessing.Pool() as pool:
                pool.map(generate_test_inputs, all_filepaths)

            # generate_test_inputs('decodes/Turk/test-pretrained-MLE/80')
            # for filepath in all_filepaths:
            #     generate_test_inputs(filepath)
            
            



"""
TODO 1: limit the vocabulary of the regex
TODO 2: Consider remove use a list test_inputs to remove test case duplicates
"""