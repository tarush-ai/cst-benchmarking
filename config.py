from dataclasses import dataclass

@dataclass
class LLMConfig:
   model_id: str
   api: bool
   endpoint: str = None

BENCHMARKED_LLM = LLMConfig(model_id="deepseek/deepseek-r1", api=True)
GENERATOR_LLM = LLMConfig(model_id="meta-llama/Llama-3.1-70B-Instruct", api=False, endpoint="http://0.0.0.0:8000")
JUDGE_LLM = LLMConfig(model_id="prometheus-eval/prometheus-7b-v2.0", api=False, endpoint="http://0.0.0.0:8001")
