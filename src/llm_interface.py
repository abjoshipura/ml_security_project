import os
import yaml
import time
import google.generativeai as genai

from typing import Optional
from openai import OpenAI

class LLMInterface:
    def __init__(self, config_path: str = "configs/config.yaml", model_name: Optional[str] = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = model_name or self.config.get('default_model', 'gpt4o')
        self.model_config = self.config['models'][self.model_name]

        self.provider = self.model_config['provider']

        self.api_key = os.getenv(self.model_config['api_key_env'])
        if not self.api_key:
            raise ValueError(f"Missing {self.model_config['api_key_env']} in environment")
        
        self._initialize_client(self.api_key)
        
        # Rate limiting to comply with OpenAPI/Gemini API rate limits
        self.rate_limit_config = self.config.get('rate_limiting', {})
        self.max_retries = self.rate_limit_config.get('max_retries', 3)
        self.base_delay = self.rate_limit_config.get('base_delay', 2)
    
    def _initialize_client(self, api_key: str):
        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "google":
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_config['model'])
        else:
            print(f"[llm_interface.py] ERROR Unsupported provider {self.provider}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        temp = temperature or self.model_config['temperature']
        max_tok = max_tokens or self.model_config['max_tokens']
        
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    return self._generate_openai(prompt, system_prompt, temp, max_tok)
                elif self.provider == "google":
                    return self._generate_gemini(prompt, system_prompt, temp, max_tok)
            except Exception as e:
                error_str = str(e).lower()
                
                if 'rate' in error_str or 'quota' in error_str or '429' in error_str or 'resource' in error_str:
                    delay = self.base_delay * (2 ** attempt) # We use exponential back-off
                    time.sleep(delay)
                    continue
                else:
                    break
        
        raise RuntimeError("Generation failed")
    
    def _generate_openai(self, prompt: str, system_prompt: Optional[str], temp: float, max_tok: int) -> str:
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_config['model'],
            messages=messages,
            temperature=temp,
            max_tokens=max_tok
        )

        return response.choices[0].message.content
    
    def _generate_gemini(self, prompt: str, system_prompt: Optional[str], temp: float, max_tok: int) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        generation_config = { 'temperature': temp, 'max_output_tokens': max_tok }

        response = self.client.generate_content(full_prompt, generation_config=generation_config)

        return response.text