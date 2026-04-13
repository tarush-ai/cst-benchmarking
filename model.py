import requests, json, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

class Model:   
   def __init__(self, model_id, api=True):
      self.model_id = model_id
      if api:
         load_dotenv()
         self.OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
      
      else:
         self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
         self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
      
   def generate_key(self, prompt, require_reasoning=True):
      response = requests.post(
      url="https://openrouter.ai/api/v1/chat/completions",
      headers={
         "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
      },
      json={
         "model": self.model_id, 
         "messages": [{"role": "user","content": prompt}],
         "reasoning": {
            "effort": "high",
            "exclude": False,
         }
      })
      

      response_data = response.json()

      if "error" in response_data:
            raise RuntimeError(f"API Error from OpenRouter: {response_data['error']}")
            
      message = response_data['choices'][0]['message']
      
      content = message.get('content', '')
      if require_reasoning: 
         reasoning = message.get('reasoning', '')
         return reasoning, content
      return content



   def generate_no_key(self, prompt):
      inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
      output_tokens = self.model.generate(**inputs) 
      generation_only = output_tokens[0][inputs.input_ids.shape[-1]:]
      return self.tokenizer.decode(generation_only, skip_special_tokens=True)