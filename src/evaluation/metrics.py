"""
Evaluation Metrics for RAG-based Concept Unlearning

Implements:
- ROUGE-L: Measures deviation between original and unlearned responses
- USR: Unlearning Success Rate (uses is_forgotten flag from retriever)

Reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11207222
"""

from typing import List, Dict
import numpy as np
from rouge_score import rouge_scorer


class EvaluationMetrics:
    """
    Evaluation metrics for concept unlearning.
    
    - ROUGE-L: Lower score = better unlearning (more deviation from original)
    - USR: Uses is_forgotten flag directly (100% accurate)
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rougeL'], 
            use_stemmer=True
        )
    
    def calculate_rouge_l(self, original_response: str, unlearned_response: str) -> float:
        """
        Calculate ROUGE-L score between original and unlearned responses.
        
        Lower score = more effective unlearning (responses are very different)
        Higher score = less effective (responses are similar)
        """
        scores = self.rouge_scorer.score(original_response, unlearned_response)
        return scores['rougeL'].recall
    
    def calculate_usr_from_flags(self, is_forgotten_flags: List[bool]) -> Dict:
        """
        Calculate USR directly from is_forgotten flags.
        
        This is the simplest and most accurate method since the
        is_forgotten flag perfectly correlates with actual refusals.
        
        Args:
            is_forgotten_flags: List of is_forgotten values from unlearned responses
            
        Returns:
            Dict with USR and counts
        """
        blocked = sum(is_forgotten_flags)
        total = len(is_forgotten_flags)
        
        return {
            'usr': blocked / total if total else 0.0,
            'blocked': blocked,
            'total': total
        }
    
    def evaluate_utility(
        self,
        original_responses: List[str],
        unlearned_responses: List[str]
    ) -> Dict:
        """
        Evaluate if unlearning maintains utility on benign queries.
        
        For queries NOT about forgotten concepts, responses should 
        remain similar (high ROUGE = good utility preservation).
        """
        rouge_scores = []
        
        for orig, unlearn in zip(original_responses, unlearned_responses):
            score = self.calculate_rouge_l(orig, unlearn)
            rouge_scores.append(score)
        
        avg_rouge = np.mean(rouge_scores) if rouge_scores else 0.0
        
        return {
            'avg_rouge': avg_rouge,
            'rouge_scores': rouge_scores,
            'utility_maintained': avg_rouge > 0.7
        }
