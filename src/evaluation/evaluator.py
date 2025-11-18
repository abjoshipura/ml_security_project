from typing import List, Dict, Optional
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from src.rag_pipeline import RAGUnlearningPipeline
from src.evaluation.metrics import EvaluationMetrics

class ExperimentEvaluator:
    """
    Complete evaluation framework for RAG-based unlearning experiments.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.pipeline = RAGUnlearningPipeline(config_path)
        self.metrics = EvaluationMetrics(config_path)
        self.results = []
    
    def run_single_query_evaluation(self, query: str, forgotten_fact: Optional[str] = None, verbose: bool = True) -> Dict:
        """
        Evaluate a single query before and after unlearning.
        
        Args:
            query: The query to test
            forgotten_fact: Fact to forget (if None, tests current state)
            verbose: Print detailed output
            
        Returns:
            Dict with evaluation results
        """
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"EVALUATING QUERY: {query}")
            print(f"{'='*60}\n")
        
        # Get response before unlearning
        if verbose:
            print("→ Getting original response...")
        original_result = self.pipeline.query(query, return_metadata=True)
        
        # Perform unlearning if fact provided
        if forgotten_fact:
            if verbose:
                print(f"→ Forgetting fact: {forgotten_fact}...")
            forget_result = self.pipeline.forget_fact(forgotten_fact)
            
            if not forget_result['success']:
                return {
                    'status': 'error',
                    'error': forget_result.get('error', 'Unknown error')
                }
        
        # Get response after unlearning
        if verbose:
            print("→ Getting unlearned response...")
        unlearned_result = self.pipeline.query(query, return_metadata=True)
        
        # Calculate metrics
        rouge = self.metrics.calculate_rouge_l(
            original_result['response'],
            unlearned_result['response']
        )
        
        # Judge success using LLM
        is_success = self.metrics._judge_unlearning_success(
            query,
            original_result['response'],
            unlearned_result['response'],
            forgotten_fact or "N/A"
        )
        
        result = {
            'query': query,
            'forgotten_fact': forgotten_fact,
            'original_response': original_result['response'],
            'unlearned_response': unlearned_result['response'],
            'rouge_l': rouge,
            'unlearning_success': is_success,
            'is_forgotten_flag': unlearned_result['is_forgotten'],
            'metadata': {
                'original': original_result.get('metadata', {}),
                'unlearned': unlearned_result.get('metadata', {})
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if verbose:
            print(f"\n{'─'*60}")
            print("RESULTS:")
            print(f"{'─'*60}")
            print(f"Original Response: {original_result['response'][:200]}...")
            print(f"\nUnlearned Response: {unlearned_result['response'][:200]}...")
            print(f"\nROUGE-L Score: {rouge:.3f}")
            print(f"Unlearning Success: {is_success}")
            print(f"Is Forgotten Flag: {unlearned_result['is_forgotten']}")
            print(f"{'='*60}\n")
        
        self.results.append(result)
        return result
    
    def run_batch_evaluation(self, test_cases: List[Dict], save_results: bool = True, output_file: Optional[str] = None) -> Dict:
        """
        Run evaluation on multiple test cases.
        
        Args:
            test_cases: List of dicts with 'query' and 'fact' keys
            save_results: Whether to save results to file
            output_file: Path to save results
            
        Returns:
            Dict with aggregate metrics
        """
        
        print(f"\n{'='*60}")
        print(f"BATCH EVALUATION: {len(test_cases)} test cases")
        print(f"{'='*60}\n")
        
        all_results = []
        
        for i, test_case in enumerate(tqdm(test_cases, desc="Evaluating")):
            result = self.run_single_query_evaluation(
                query=test_case['query'],
                forgotten_fact=test_case.get('fact'),
                verbose=False
            )
            all_results.append(result)
        
        # Calculate aggregate metrics
        rouge_scores = [r['rouge_l'] for r in all_results]
        success_count = sum(1 for r in all_results if r['unlearning_success'])
        
        aggregate = {
            'total_cases': len(test_cases),
            'successful_unlearning': success_count,
            'usr': success_count / len(test_cases) if test_cases else 0.0,
            'avg_rouge_l': sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
            'min_rouge_l': min(rouge_scores) if rouge_scores else 0.0,
            'max_rouge_l': max(rouge_scores) if rouge_scores else 0.0,
            'timestamp': datetime.now().isoformat(),
            'detailed_results': all_results
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Cases: {aggregate['total_cases']}")
        print(f"USR: {aggregate['usr']:.2%}")
        print(f"Avg ROUGE-L: {aggregate['avg_rouge_l']:.3f}")
        print(f"Successful Unlearning: {success_count}/{len(test_cases)}")
        print(f"{'='*60}\n")
        
        # Save results
        if save_results:
            output_path = output_file or f"outputs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_path, 'w') as f:
                json.dump(aggregate, f, indent=2)
            print(f"✓ Results saved to: {output_path}")
        
        return aggregate
    
    def evaluate_benign_utility(self, benign_queries: List[str], forgotten_facts: List[str]) -> Dict:
        """
        Test if unlearning maintains utility on benign queries
        """
        
        print(f"\n{'='*60}")
        print("BENIGN UTILITY EVALUATION")
        print(f"{'='*60}\n")
        
        # First, forget all facts
        for fact in tqdm(forgotten_facts, desc="Forgetting facts"):
            self.pipeline.forget_fact(fact)
        
        # Collect responses before and after
        original_responses = []
        unlearned_responses = []
        
        # Get original responses (with clean state)
        for query in tqdm(benign_queries, desc="Collecting original responses"):
            result = self.pipeline.query(query)
            original_responses.append(result['response'])
        
        # Get unlearned responses
        for query in tqdm(benign_queries, desc="Collecting unlearned responses"):
            result = self.pipeline.query(query)
            unlearned_responses.append(result['response'])
        
        # Calculate utility metrics
        utility_result = self.metrics.evaluate_utility(
            benign_queries,
            original_responses,
            unlearned_responses
        )
        
        print(f"\nUtility Maintained: {utility_result['utility_maintained']}")
        print(f"Average ROUGE-L: {utility_result['avg_rouge']:.3f}")
        
        return utility_result
    
    def test_adversarial_resistance(self, forgotten_fact: str, adversarial_queries: List[str]) -> Dict:
        """
        Test resistance to adversarial queries trying to bypass unlearning.
        Criterion (ii)
        """
        
        print(f"\n{'='*60}")
        print(f"ADVERSARIAL RESISTANCE TEST: {forgotten_fact}")
        print(f"{'='*60}\n")
        
        # Forget the fact
        self.pipeline.forget_fact(forgotten_fact)
        
        results = []
        bypass_count = 0
        
        for query in tqdm(adversarial_queries, desc="Testing adversarial queries"):
            result = self.pipeline.query(query, return_metadata=True)
            
            # Check if adversary bypassed unlearning
            bypassed = not result['is_forgotten'] and forgotten_fact.lower() in result['response'].lower()
            
            if bypassed:
                bypass_count += 1
            
            results.append({
                'query': query,
                'bypassed': bypassed,
                'response': result['response'],
                'is_forgotten': result['is_forgotten']
            })
        
        resistance_rate = 1 - (bypass_count / len(adversarial_queries))
        
        print(f"\nResistance Rate: {resistance_rate:.2%}")
        print(f"Bypassed: {bypass_count}/{len(adversarial_queries)}")
        
        return {
            'resistance_rate': resistance_rate,
            'bypass_count': bypass_count,
            'total_queries': len(adversarial_queries),
            'detailed_results': results
        }

    def run_multi_model_evaluation(self, test_cases: List[Dict], models: List[str] = ['gpt4o', 'gemini', 'llama2'], save_results: bool = True) -> Dict:
        """
        Run evaluation across multiple models.
        
        Args:
            test_cases: List of test cases
            models: List of model names to test
            save_results: Whether to save results
            
        Returns:
            Dict with results for each model
        """
        
        print("\n" + "=" * 60)
        print(f"MULTI-MODEL EVALUATION")
        print(f"Models: {', '.join(models)}")
        print(f"Test cases: {len(test_cases)}")
        print("=" * 60)
        
        all_model_results = {}
        
        for model_name in models:
            print(f"\n{'─'*60}")
            print(f"EVALUATING: {model_name.upper()}")
            print(f"{'─'*60}")
            
            # Reinitialize pipeline with specific model
            from src.rag_pipeline import RAGUnlearningPipeline
            self.pipeline = RAGUnlearningPipeline(model_name=model_name)
            
            # Run evaluation
            results = self.run_batch_evaluation(
                test_cases=test_cases,
                save_results=False,
                verbose=False
            )
            
            all_model_results[model_name] = results
            
            # Print summary
            print(f"\n{model_name.upper()} Results:")
            print(f"  USR: {results['usr']:.2%}")
            print(f"  Avg ROUGE-L: {results['avg_rouge_l']:.3f}")
        
        # Comparative analysis
        print("\n" + "=" * 60)
        print("COMPARATIVE SUMMARY")
        print("=" * 60)
        
        comparison_df = pd.DataFrame({
            model: {
                'USR': results['usr'],
                'Avg ROUGE-L': results['avg_rouge_l'],
                'Successful': results['successful_unlearning'],
                'Total': results['total_cases']
            }
            for model, results in all_model_results.items()
        }).T
        
        print(comparison_df)
        
        # Save combined results
        if save_results:
            output_file = f"outputs/multi_model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(all_model_results, f, indent=2)
            print(f"\n✓ Results saved to: {output_file}")
        
        return all_model_results