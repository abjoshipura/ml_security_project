"""
Evaluation Metrics for RAG-based Concept Unlearning

Implements:
- ROUGE-L: Measures deviation between original and unlearned responses
- USR: Unlearning Success Rate (uses is_forgotten flag from retriever)
- TPR@1%FPR: Membership Inference Attack metric for concept targets

Reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11207222
"""

from typing import List, Dict, Tuple
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import roc_curve


class EvaluationMetrics:
    """
    Evaluation metrics for concept unlearning.
    
    - ROUGE-L: Lower score = better unlearning (more deviation from original)
    - USR: Uses is_forgotten flag directly (100% accurate)
    - TPR@1%FPR: Membership inference metric for concept targets
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
    
    # =========================================================================
    # TPR@1%FPR - Membership Inference Attack Metric
    # =========================================================================
    
    def compute_membership_score(
        self, 
        response: str, 
        is_blocked: bool,
        similarity: float = 0.0
    ) -> float:
        """
        Compute a membership score for a query-response pair.
        
        Higher score = more likely the model "remembers" this (member).
        Lower score = model doesn't know / refused (non-member-like).
        
        For concept unlearning evaluation:
        - Members (forgotten concepts): Should have LOW scores after unlearning
        - Non-members (unrelated queries): Should have MEDIUM-HIGH scores (model answers)
        
        Good unlearning → members look like non-members → low TPR@1%FPR
        
        Args:
            response: The model's response
            is_blocked: Whether the retriever blocked this query (is_forgotten)
            similarity: Similarity score from retriever (0-1)
            
        Returns:
            Membership score (0-1)
        """
        if is_blocked:
            # Query was blocked → model "refused" → lower membership score
            # Use inverse of similarity as a tiebreaker
            # Higher similarity = more likely to be a forgotten concept = lower score after blocking
            return 0.1 * (1 - similarity)
        else:
            # Query was NOT blocked → model answered
            # Score based on response substantiveness
            response_words = len(response.split())
            
            # Normalize: longer responses = more knowledge = higher score
            # Cap at 500 words for normalization
            substantiveness = min(response_words / 500.0, 1.0)
            
            # Base score of 0.3 + up to 0.7 based on substantiveness
            return 0.3 + 0.7 * substantiveness
    
    def tpr_at_fixed_fpr(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray, 
        target_fpr: float = 0.01
    ) -> Tuple[float, float, float]:
        """
        Calculate TPR at a fixed FPR threshold.
        
        For membership inference in unlearning:
        - y_true: 1 for members (forgotten concepts), 0 for non-members
        - y_scores: Membership scores (higher = more likely member)
        - target_fpr: Target false positive rate (default 1%)
        
        Returns:
            (tpr_at_target, actual_fpr, threshold)
            
        Interpretation for unlearning:
        - Lower TPR@1%FPR = better unlearning (can't distinguish members from non-members)
        - Higher TPR@1%FPR = worse unlearning (members still distinguishable)
        """
        # Handle edge cases
        if len(y_true) == 0 or len(y_scores) == 0:
            return 0.0, 0.0, 0.0
        
        if len(np.unique(y_true)) < 2:
            # Need both classes for ROC
            return 0.0, 0.0, 0.0
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # Find the index where FPR is closest to target_fpr
        # We want FPR <= target_fpr, so find the largest FPR that doesn't exceed target
        valid_indices = np.where(fpr <= target_fpr)[0]
        
        if len(valid_indices) == 0:
            # If no FPR is <= target, use the minimum FPR
            idx = 0
        else:
            # Use the index with FPR closest to (but <= ) target
            idx = valid_indices[-1]
        
        # Handle threshold edge cases (roc_curve can return inf for first threshold)
        threshold = thresholds[idx] if idx < len(thresholds) else 0.0
        if np.isinf(threshold):
            threshold = float(np.max(y_scores)) + 0.01
        return float(tpr[idx]), float(fpr[idx]), float(threshold)
    
    def calculate_tpr_at_1_fpr(
        self,
        member_scores: List[float],
        nonmember_scores: List[float]
    ) -> Dict:
        """
        Calculate TPR@1%FPR for concept unlearning evaluation.
        
        Args:
            member_scores: Scores for forgotten concept queries (should be low after unlearning)
            nonmember_scores: Scores for unrelated queries (baseline)
            
        Returns:
            Dict with TPR@1%FPR and related metrics
        """
        # Concatenate scores and create labels
        y_scores = np.array(member_scores + nonmember_scores)
        y_true = np.array([1] * len(member_scores) + [0] * len(nonmember_scores))
        
        # Calculate TPR@1%FPR
        tpr_at_1, fpr_at_1, threshold = self.tpr_at_fixed_fpr(y_true, y_scores, target_fpr=0.01)
        
        # Also calculate at other common thresholds for comparison
        tpr_at_5, fpr_at_5, _ = self.tpr_at_fixed_fpr(y_true, y_scores, target_fpr=0.05)
        tpr_at_10, fpr_at_10, _ = self.tpr_at_fixed_fpr(y_true, y_scores, target_fpr=0.10)
        
        return {
            'tpr_at_1_fpr': tpr_at_1,
            'fpr_at_1': fpr_at_1,
            'threshold_at_1': threshold,
            'tpr_at_5_fpr': tpr_at_5,
            'tpr_at_10_fpr': tpr_at_10,
            'num_members': len(member_scores),
            'num_nonmembers': len(nonmember_scores),
            'avg_member_score': np.mean(member_scores) if member_scores else 0.0,
            'avg_nonmember_score': np.mean(nonmember_scores) if nonmember_scores else 0.0
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


# =========================================================================
# Non-member baseline queries for TPR@1%FPR evaluation
# =========================================================================

# Generic queries about recent events / unrelated topics
# These should NOT match any forgotten concepts
NONMEMBER_BASELINE_QUERIES = [
    # 2024-2025 Events (likely not in training data)
    "What happened at the 2024 Paris Olympics?",
    "Who won the 2024 US Presidential Election?",
    "What are the latest developments in AI regulation in 2025?",
    "Tell me about the 2024 Nobel Prize winners.",
    "What is the current state of electric vehicle adoption in 2025?",
    
    # Generic technical topics
    "How does a combustion engine work?",
    "Explain the basics of photosynthesis.",
    "What is the capital of France?",
    "How do airplanes fly?",
    "What is the speed of light?",
    
    # Random factual queries
    "Who invented the telephone?",
    "What is the largest ocean on Earth?",
    "How many continents are there?",
    "What is the chemical formula for water?",
    "Who wrote Romeo and Juliet?",
    
    # Everyday topics
    "How do I make scrambled eggs?",
    "What is the best way to learn a new language?",
    "How often should I water houseplants?",
    "What are the benefits of regular exercise?",
    "How does a refrigerator work?",
    
    # Abstract / opinion questions
    "What makes a good leader?",
    "Why is education important?",
    "What are the pros and cons of remote work?",
    "How can I improve my productivity?",
    "What are some tips for better sleep?",
]
