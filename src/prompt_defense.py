import os
import re
import logging
import structlog

logging.getLogger("llm_guard").setLevel(logging.ERROR)
logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
)

os.environ["LLM_GUARD_LOG_LEVEL"] = "ERROR"

from typing import Dict, List
from llm_guard.input_scanners import PromptInjection, Toxicity

class PromptDefense:
    """
    Defense mechanisms for detecting adversarial attacks.
    Uses LLM Guard scanners for prompt injection and toxicity detection.
    """
    
    def __init__(self, pi_strict_mode: bool = True, jb_strict_mode: bool = False):
        self.pi_threshold = 0.5 if pi_strict_mode else 0.7
        self.jb_threshold = 0.5 if jb_strict_mode else 0.6
        
        # Load LLM Guard scanners
        self.prompt_injection_scanner = None
        self.toxicity_scanner = None
        
        try:
            self.prompt_injection_scanner = PromptInjection(threshold=self.pi_threshold)
            self.toxicity_scanner = Toxicity(threshold=self.jb_threshold)
        except ImportError as _:
            print(f"[prompt_defense.py] ERROR llm_guard not available")
        except Exception as _:
            print(f"[prompt_defense.py] ERROR Failed to initialize llm_guard")
    
    def validate_query(self, query: str) -> Dict:
        """
        Check for injection and/or toxicity in the query, i.e., prompt injection 
        and/or jailbreak attempts respectively.
        """
        concerns = []
        confidence = injection_score = toxicity_score = 0.0

        # Check for prompt injection
        if self.prompt_injection_scanner:
            try:
                _, is_valid, injection_score = self.prompt_injection_scanner.scan(query)
                if not is_valid:
                    concerns.append('prompt_injection')
                    confidence = max(confidence, injection_score)
            except Exception as _:
                print(f"[prompt_defense.py] ERROR Prompt injection scan error")
        
        # Check for toxicity
        if self.toxicity_scanner:
            try:
                _, is_valid, toxicity_score = self.toxicity_scanner.scan(query)
                if not is_valid:
                    concerns.append('toxicity')
                    confidence = max(confidence, toxicity_score)
            except Exception as _:
                print(f"[prompt_defense.py] ERROR Toxicity scan error")
        
        return {
            'is_safe': len(concerns) == 0,
            'should_block': len(concerns) > 0,
            'confidence': float(confidence),
            'concerns': concerns,
            'injection_score': injection_score,
            'toxicity_score': toxicity_score
        }

    def validate_llm_response(self, response: str, forbidden_keywords: List[str]) -> Dict:
        """
        Validate that LLM response doesn't contain forbidden information, i.e., 
        there is no leakage of information
        """
        if not forbidden_keywords:
            return {'is_valid': True, 'leaked_keywords': []}
        
        leaked = []
        for keyword in forbidden_keywords:
            if len(keyword) < 4:
                continue

            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, response.lower()):
                leaked.append(keyword)
        
        return {'is_valid': len(leaked) < 2, 'leaked_keywords': leaked}