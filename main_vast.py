from datasets import load_dataset
from model import Model
from benchmarks.bigcodebench.bigcodebench import BigCodeBench
from benchmarks.math500.math500 import Math500
from benchmarks.mmlupro.mmlupro import MMLUPro
from benchmarks.medxpertqa.medxpertqa import MedXpertQA
from config import GENERATOR_LLM, JUDGE_LLM
import concurrent.futures

class Main:
   def __init__(self):
      self.generator_llm = Model(model_id=GENERATOR_LLM.model_id, api=GENERATOR_LLM.api, endpoint=GENERATOR_LLM.endpoint)
      self.judge_llm = Model(model_id=JUDGE_LLM.model_id, api=JUDGE_LLM.api, endpoint=JUDGE_LLM.endpoint)

      self.bigcodebench = BigCodeBench(self.generator_llm, None, self.judge_llm)
      self.math500 = Math500(self.generator_llm, None, self.judge_llm)
      self.mmlupro = MMLUPro(self.generator_llm, None, self.judge_llm)
      self.medxpertqa = MedXpertQA(self.generator_llm, None, self.judge_llm)

   def evaluate_benchmark_llm_a_base(self):
      tasks = [
         self.bigcodebench.generate_and_evaluate_llma_base,
         self.math500.generate_and_evaluate_llma_base,
         self.mmlupro.generate_and_evaluate_llma_base,
         self.medxpertqa.generate_and_evaluate_llma_base
      ]

      with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
         futures = [executor.submit(task) for task in tasks]
         for f in concurrent.futures.as_completed(futures):
            f.result()

   def evaluate_benchmark_judge_llm(self):
      tasks = [
         self.bigcodebench.generate_judge_llm_eval,
         self.math500.generate_judge_llm_eval,
         self.mmlupro.generate_judge_llm_eval,
         self.medxpertqa.generate_judge_llm_eval
      ]

      with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
         futures = [executor.submit(task) for task in tasks]
         for f in concurrent.futures.as_completed(futures):
            f.result()
