from datasets import load_dataset
from model import Model

class BigCodeBench:
   def __init__(self, llmA: Model, llmB: Model):
      self.llmA = llmA
      self.llmB = llmB
   
   def generate(self):
      bigcodebench = load_dataset("bigcode/bigcodebench", split="v0.1.4", streaming=True)
      for example in self.bigcodebench:
         self.groundtruth_generate(example)
   
   def groundtruth_generate(self, example):   
      deepseek_groundtruth_prompt = f'''
      Please generate self-contained code to complete the following problem:
      {example["complete_prompt"]}
      '''
      response = self.llmB.generate_key(deepseek_groundtruth_prompt)
      

   


