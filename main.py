from datasets import load_dataset
from model import Model
from benchmarks.bigcodebench.bigcodebench import BigCodeBench
from benchmarks.math500.math500 import Math500
from benchmarks.mmlupro.mmlupro import MMLUPro
from benchmarks.medxpertqa.medxpertqa import MedXpertQA
from config import BENCHMARKED_LLM, GENERATOR_LLM, JUDGE_LLM

class Main:
   def __init__(self):
      self.benchmarked_llm = Model(model_id=BENCHMARKED_LLM.model_id, api=BENCHMARKED_LLM.api)
      self.generator_llm = Model(model_id=GENERATOR_LLM.model_id, api=GENERATOR_LLM.api)
      self.judge_llm = Model(model_id=JUDGE_LLM.model_id, api=JUDGE_LLM.api)
      
      self.bigcodebench = BigCodeBench(self.benchmarked_llm, self.generator_llm, self.judge_llm)
      self.math500 = Math500(self.generator_llm, self.benchmarked_llm, self.judge_llm)
      self.mmlupro = MMLUPro(self.generator_llm, self.benchmarked_llm, self.judge_llm)
      self.medxpertqa = MedXpertQA(self.generator_llm, self.benchmarked_llm, self.judge_llm)

   def generate(self):
      self.bigcodebench.generate()
      self.math500.generate()
      self.mmlupro.generate()
      self.medxpertqa.generate()


      




