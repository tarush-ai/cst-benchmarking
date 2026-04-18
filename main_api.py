from datasets import load_dataset
from model import Model
from benchmarks.bigcodebench.bigcodebench import BigCodeBench
from benchmarks.math500.math500 import Math500
from benchmarks.mmlupro.mmlupro import MMLUPro
from benchmarks.medxpertqa.medxpertqa import MedXpertQA
from config import BENCHMARKED_LLM
import concurrent.futures

class Main:
   def __init__(self):
      self.benchmarked_llm = Model(model_id=BENCHMARKED_LLM.model_id, api=BENCHMARKED_LLM.api, endpoint=BENCHMARKED_LLM.endpoint)

      self.bigcodebench = BigCodeBench(None, self.benchmarked_llm, None)
      self.math500 = Math500(None, self.benchmarked_llm, None)
      self.mmlupro = MMLUPro(None, self.benchmarked_llm, None)
      self.medxpertqa = MedXpertQA(None, self.benchmarked_llm, None)

   def evaluate_benchmark_llmb_base(self):
      tasks = [
         self.bigcodebench.generate_and_evaluate_llmb_base,
         self.math500.generate_and_evaluate_llmb_base,
         self.mmlupro.generate_and_evaluate_llmb_base,
         self.medxpertqa.generate_and_evaluate_llmb_base
      ]

      with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
         futures = [executor.submit(task) for task in tasks]
         for f in concurrent.futures.as_completed(futures):
            f.result()

   def evaluate_llmb_human_llm(self):
      tasks = [
         self.bigcodebench.generate_llm_b_human_llm,
         self.math500.generate_llm_b_human_llm,
         self.mmlupro.generate_llm_b_human_llm,
         self.medxpertqa.generate_llm_b_human_llm
      ]
    
      with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
         futures = [executor.submit(task) for task in tasks]
         for f in concurrent.futures.as_completed(futures):
            f.result()
