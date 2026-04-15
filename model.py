import requests, json, os, time, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

_RETRYABLE = {429, 500, 502, 503, 504}
_MAX_RETRIES = 10
_BASE_DELAY = 1.0
_MAX_DELAY = 64.0

def _backoff(attempt, retry_after=None):
   if retry_after:
      return float(retry_after)
   return min(_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), _MAX_DELAY)

class Model:
   def __init__(self, model_id, api=True, endpoint=None):
      self.model_id = model_id
      self.endpoint = endpoint
      if api:
         load_dotenv()
         self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
      elif not endpoint:
         self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
         self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

   def generate_key(self, prompt, require_reasoning=True):
      for attempt in range(_MAX_RETRIES):
         try:
            response = requests.post(
               url="https://openrouter.ai/api/v1/chat/completions",
               headers={"Authorization": f"Bearer {self.OPENROUTER_API_KEY}"},
               json={
                  "model": self.model_id,
                  "messages": [{"role": "user", "content": prompt}],
                  "reasoning": {"effort": "high", "exclude": False}
               },
               timeout=180
            )
            if response.status_code in _RETRYABLE:
               if attempt == _MAX_RETRIES - 1:
                  raise RuntimeError(f"OpenRouter returned {response.status_code} after {_MAX_RETRIES} retries")
               time.sleep(_backoff(attempt, response.headers.get("Retry-After")))
               continue
            response_data = response.json()
            if "error" in response_data:
               raise RuntimeError(f"API Error from OpenRouter: {response_data['error']}")
            message = response_data["choices"][0]["message"]
            content = message.get("content", "")
            if require_reasoning:
               return message.get("reasoning", ""), content
            return content
         except (requests.Timeout, requests.ConnectionError):
            if attempt == _MAX_RETRIES - 1:
               raise
            time.sleep(_backoff(attempt))

   def generate_no_key(self, prompt):
      if self.endpoint:
         for attempt in range(_MAX_RETRIES):
            try:
               response = requests.post(
                  url=f"{self.endpoint}/v1/chat/completions",
                  json={
                     "model": self.model_id,
                     "messages": [{"role": "user", "content": prompt}]
                  },
                  timeout=300
               )
               if response.status_code in _RETRYABLE:
                  if attempt == _MAX_RETRIES - 1:
                     raise RuntimeError(f"vLLM returned {response.status_code} after {_MAX_RETRIES} retries")
                  time.sleep(_backoff(attempt, response.headers.get("Retry-After")))
                  continue
               response.raise_for_status()
               return response.json()["choices"][0]["message"]["content"]
            except (requests.Timeout, requests.ConnectionError):
               if attempt == _MAX_RETRIES - 1:
                  raise
               time.sleep(_backoff(attempt))
      else:
         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
         output_tokens = self.model.generate(**inputs)
         generation_only = output_tokens[0][inputs.input_ids.shape[-1]:]
         return self.tokenizer.decode(generation_only, skip_special_tokens=True)
