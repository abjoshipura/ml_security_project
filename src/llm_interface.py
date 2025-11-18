from dotenv import load_dotenv
import os
import yaml
import google.generativeai as genai
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Optional, Union
from openai import OpenAI

class LLMInterface:
    def __init__(self, config_path: str = "configs/config.yaml", model_name: Optional[str] = None):
        load_dotenv()

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = model_name or self.config.get('default_model', 'gemini')
        self.model_config = self.config['models'][self.model_name]
        
        if not self.model_config.get('enabled', True):
            raise ValueError(f"Model {self.model_name} is disabled in config")
        
        self._initialize_client()
    
    def _initialize_client(self):
        provider = self.model_config['provider']
        
        if provider == "openai":
            self._init_openai()
        elif provider == "google":
            self._init_gemini()
        elif provider in ["local", "huggingface"]:
            self._init_llama()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _init_openai(self):
        api_key = os.getenv(self.model_config['api_key_env'])
        if not api_key:
            raise ValueError(f"Missing {self.model_config['api_key_env']} in environment")
        
        self.client = OpenAI(api_key=api_key)
        print(f"Initialized GPT-4o")
    
    def _init_gemini(self):
        api_key = os.getenv(self.model_config['api_key_env'])
        if not api_key:
            raise ValueError(f"Missing {self.model_config['api_key_env']} in environment")
        
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model_config['model'])
        print(f"Initialized Gemini")
    
    # TODO Ignored for now. Delete if not used later.
    def _init_llama(self):
        hf_token = os.getenv(self.model_config.get('hf_token_env'))
        model_path = self.model_config['model']
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=hf_token,
            trust_remote_code=True
        )
        
        device = self.model_config.get('device', 'cpu')
        
        if device == "cuda" and torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                token=hf_token,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            print(f"Initialized Llama-2-7b-chat on CUDA")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                token=hf_token,
                trust_remote_code=True
            )
            self.model = self.model.to(device)
            print(f"Initialized Llama-2-7b-chat on {device}")
        
        # Create pipeline
        self.client = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" and torch.cuda.is_available() else -1
        )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        temp = temperature or self.model_config['temperature']
        max_tok = max_tokens or self.model_config['max_tokens']
        provider = self.model_config['provider']
        
        try:
            if provider == "openai":
                return self._generate_openai(prompt, system_prompt, temp, max_tok)
            elif provider == "google":
                return self._generate_gemini(prompt, system_prompt, temp, max_tok)
            elif provider in ["local", "huggingface"]:
                return self._generate_llama(prompt, system_prompt, temp, max_tok)
        except Exception as e:
            print(f"Error in LLM generation ({self.model_name}): {e}")
            raise
    
    def _generate_openai(self, prompt, system_prompt, temp, max_tok):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_config['model'],
            messages=messages,
            temperature=temp,
            max_tokens=max_tok
        )
        return response.choices[0].message.content
    
    def _generate_gemini(self, prompt, system_prompt, temp, max_tok):
        # Gemini handles system prompt differently
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        generation_config = {
            'temperature': temp,
            'max_output_tokens': max_tok,
        }
        
        response = self.client.generate_content(
            full_prompt,
            generation_config=generation_config
        )
        return response.text
    
    # TODO Ignored for now. Delete if not used later.
    def _generate_llama(self, prompt, system_prompt, temp, max_tok):
        # Format prompt for Llama-2-chat
        if system_prompt:
            formatted_prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]"""
        else:
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        outputs = self.client(
            formatted_prompt,
            max_new_tokens=max_tok,
            temperature=temp,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1
        )
        
        return outputs[0]['generated_text'].split('[/INST]')[-1].strip()