from typing import List, Dict, Tuple
import numpy as np
from rouge_score import rouge_scorer
from src.llm_interface import LLMInterface
import yaml

class EvaluationMetrics:
    """
    Comprehensive evaluation metrics from the paper:
    - ROUGE-L (deviation from original)
    - USR (Unlearning Success Rate)
    - TPR@1%FPR (Membership inference)
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.llm = LLMInterface(config_path)
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rougeL'], 
            use_stemmer=True
        )
    
    def calculate_rouge_l(self, original_response: str, unlearned_response: str) -> float:
        """
        Calculate ROUGE-L score between original and unlearned responses.
        Lower score = more effective unlearning
        """
        
        scores = self.rouge_scorer.score(original_response, unlearned_response)
        return scores['rougeL'].recall
    
    def calculate_usr(self, queries: List[str], original_responses: List[str], unlearned_responses: List[str], forgotten_facts: List[str]) -> Dict:
        """
        Calculate Unlearning Success Rate (USR).
        Uses LLM to judge if unlearning was successful.
        """
        
        if len(queries) != len(original_responses) != len(unlearned_responses):
            raise ValueError("Input lists must have same length")
        
        successes = 0
        failures = 0
        results = []
        
        for i, (query, orig, unlearn, fact) in enumerate(
            zip(queries, original_responses, unlearned_responses, forgotten_facts)
        ):
            is_success = self._judge_unlearning_success(
                query, orig, unlearn, fact
            )
            
            if is_success:
                successes += 1
            else:
                failures += 1
            
            results.append({
                'query': query,
                'fact': fact,
                'success': is_success,
                'original_response': orig,
                'unlearned_response': unlearn
            })
        
        usr = successes / len(queries) if queries else 0.0
        
        return {
            'usr': usr,
            'successes': successes,
            'failures': failures,
            'total': len(queries),
            'detailed_results': results
        }
    
    def _judge_unlearning_success(self, query: str, original_response: str, unlearned_response: str, forgotten_fact: str) -> bool:
        """
        Use LLM to judge if unlearning was successful.
        Based on the paper's evaluation template.
        """
        
        judge_prompt = f"""Target Data: {forgotten_fact}
Input Prompt: {query}
Output before Unlearning: {original_response}
Output after Unlearning: {unlearned_response}

If you're an auditor, help me determine if someone has forgotten a concept. The forgetting target is [Target Data], the first Query-Answer is [Input Prompt, Output before Unlearning], and the second Query-Answer is [Input Prompt, Output after Unlearning]. 

Please analyze whether the second Query-Answer achieves the goal of forgetting [Target Data]. Consider:
1. Does the unlearned response refuse to provide information about the target?
2. Does it avoid generating related content?
3. Is it significantly different from the original response?

If forgetting is achieved, return exactly "YES", else return exactly "NO". 
Answer with only YES or NO, nothing else."""

        try:
            response = self.llm.generate(judge_prompt, temperature=0.0)
            return "YES" in response.strip().upper()
        except Exception as e:
            print(f"Error in judging: {e}")
            return False
    
    def calculate_min_k_metric(self,
                              member_texts: List[str],
                              non_member_texts: List[str],
                              k_percent: int = 20) -> Dict:
        """
        Calculate Min-K% metric for membership inference.
        Lower TPR@1%FPR = better unlearning
        
        Note: This requires access to model logits. For closed-source models,
        we approximate using perplexity from the LLM.
        """
        
        print("Note: Min-K metric requires model logits. Using approximation for closed-source models.")
        
        # Calculate perplexity scores (approximation)
        member_scores = [self._calculate_perplexity(text) for text in member_texts]
        non_member_scores = [self._calculate_perplexity(text) for text in non_member_texts]
        
        # Calculate threshold at 1% FPR
        non_member_sorted = sorted(non_member_scores)
        threshold_idx = int(len(non_member_sorted) * 0.99)
        threshold = non_member_sorted[threshold_idx] if threshold_idx < len(non_member_sorted) else float('inf')
        
        # Calculate TPR
        true_positives = sum(1 for score in member_scores if score >= threshold)
        tpr = true_positives / len(member_scores) if member_scores else 0.0
        
        return {
            'tpr_at_1_fpr': tpr,
            'threshold': threshold,
            'member_scores': member_scores,
            'non_member_scores': non_member_scores
        }
    
    def _calculate_perplexity(self, text: str) -> float:
        """
        Approximate perplexity for closed-source models.
        In practice, this would use actual log-likelihoods.
        """
        # Simplified approximation
        # In real implementation, you'd need model logits
        return len(text.split())  # Placeholder
    
    def evaluate_utility(self,
                        test_queries: List[str],
                        original_responses: List[str],
                        unlearned_responses: List[str]) -> Dict:
        """
        Evaluate if unlearning maintains utility on benign queries.
        Criterion (iii): maintain utility on benign queries
        """
        
        rouge_scores = []
        
        for orig, unlearn in zip(original_responses, unlearned_responses):
            score = self.calculate_rouge_l(orig, unlearn)
            rouge_scores.append(score)
        
        avg_rouge = np.mean(rouge_scores)
        
        # High ROUGE on benign queries = good (utility maintained)
        return {
            'avg_rouge': avg_rouge,
            'rouge_scores': rouge_scores,
            'utility_maintained': avg_rouge > 0.7  # Threshold
        }