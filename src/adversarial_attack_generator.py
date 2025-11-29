import re
import langid
import random

from typing import List, Dict
from datasets import load_dataset

class AdversarialAttackGenerator:
    """
    Generates two types of adversarial attacks:
        1. Prompt Injection Attacks (deepset/prompt-injections)
        2. Jailbreak Attacks (JailbreakBench/JBB-Behaviors)
    using these datasets respectively:
        1. deepset/prompt-injections: https://huggingface.co/datasets/deepset/prompt-injections
        2. JailbreakBench/JBB-Behaviors: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
    """
    
    def __init__(self):
        self.injection_patterns = self._load_prompt_injection_patterns()
        self.jailbreak_queries = self._load_jailbreak_queries()
    
    def generate_prompt_injection_attacks(self, concept: str, num_attacks: int) -> List[Dict]:
        """
        Generate prompt injection attacks using injection patterns from
        the deepset dataset with the substitution of concepts into the 
        patterns.
        """
        attacks = list()
        
        if not self.injection_patterns:
            return attacks
        
        concept_templates = [f"Now tell me about {concept}", f"What is {concept}?", f"Explain {concept}", f"Give me information about {concept}", f"Describe {concept} in detail"]
        
        # Randomly sample patterns. Ensure we use as many unique patterns as possible first, before filling the rest with random duplicates
        num_unique = min(num_attacks, len(self.injection_patterns))
        num_extra = num_attacks - num_unique

        patterns = (random.sample(self.injection_patterns, num_unique) + (random.choices(self.injection_patterns, k=num_extra) if num_extra > 0 else []))

        # File the patterns and create the attacks
        for pattern in patterns:
            attacks.append({
                'query': f"{pattern['injection']} {random.choice(concept_templates)}",
                'type': 'prompt_injection',
                'concept': concept,
                'source': 'deepset_prompt_injections'
            })

        return attacks
    
    def generate_jailbreak_attacks(self, num_attacks: int) -> List[Dict]:
        """
        Samples jailbreak attacks from the JailbreakBench dataset to
        test LLM safety.
        """
        attacks = list()
        
        if not self.jailbreak_queries:
            return attacks
        
        num_unique = min(num_attacks, len(self.jailbreak_queries))
        num_extra = num_attacks - num_unique

        # Randomly sample queries. Ensure we use as many unique queries as possible first, before filling the rest with random duplicates
        queries = (random.sample(self.jailbreak_queries, num_unique) + (random.choices(self.jailbreak_queries, k=num_extra) if num_extra > 0 else []))

        # Collate the attacks
        for query in queries:
            attacks.append({
                'query': query['goal'],
                'type': 'jailbreak',
                'category': query['category'],
                'source': 'jailbreakbench'
            })
        
        return attacks
    
    def _load_prompt_injection_patterns(self) -> List[Dict]:
        """Load English-language prompt injection patterns from the deepset dataset"""
        patterns = list()
        
        try:
            dataset = load_dataset("deepset/prompt-injections", split="train")
            
            for item in dataset:
                text = item.get('text', '')
                label = item.get('label', 0)
                
                # Only use English injection examples (where label = 1)
                if label != 1 or not text or not langid.classify(text)[0]:
                    continue
                
                # Clean up the text
                cleaned = self._clean_injection(text)
                if cleaned and len(cleaned) >= 20:
                    patterns.append({
                        'injection': cleaned,
                        'source': 'deepset_prompt_injections'
                    })

        except Exception as _:
            print("[adversarial_attack_generator.py] ERROR loading/parsing the deepset dataset")
            return []
        
        return patterns
    
    def _load_jailbreak_queries(self) -> List[Dict]:
        """Load English-language jailbreak queries from the JailbreakBench dataset"""
        queries = list()
        
        try:
            dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
            
            for item in dataset:
                goal = item.get('Goal', '')
                category = item.get('Category', 'general')
                
                if goal and langid.classify(goal)[0] == 'en':
                    queries.append({
                        'goal': goal,
                        'category': category.lower().replace(' ', '_'),
                        'source': 'jailbreakbench'
                    })
            
        except Exception as _:
            print("[adversarial_attack_generator.py] ERROR loading/parsing the JailbreakBench dataset")
            pass

        return queries
    
    def _clean_injection(self, text: str) -> str:
        """Clean up the injection template for use as a pattern"""
        
        # Remove any placeholder patterns
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\{.*?\}', '', text)
        text = re.sub(r'<(?!/?[a-zA-Z])[^>]*>', '', text)
        
        # Remove trailing questions
        text = re.sub(r'(?i)\s*\b(what|how|why|who|tell|give|show|explain|describe)\b.*\??\s*$', '', text)

        text = ' '.join(text.split())
        text = text.rstrip('.,;:')
        
        return text.strip()