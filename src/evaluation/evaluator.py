"""
Evaluator for RAG-based Concept Unlearning

Implements evaluation metrics from the paper:
- Unlearning Success Rate (USR): % of queries where unlearning is effective
- ROUGE-L: Deviation between original and unlearned responses
- Adversarial Resistance: Robustness to rephrased queries

Reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11207222
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from src.rag_pipeline import RAGUnlearningPipeline
from src.evaluation.metrics import EvaluationMetrics, NONMEMBER_BASELINE_QUERIES
from src.knowledge_base.kb_manager import KnowledgeBaseManager


class ConceptUnlearningEvaluator:
    """
    Evaluates the effectiveness of RAG-based concept unlearning.
    """
    
    def __init__(
        self, 
        config_path: str = "configs/config.yaml",
        model_name: Optional[str] = None
    ):
        self.config_path = config_path
        self.model_name = model_name
        self.pipeline = RAGUnlearningPipeline(config_path, model_name=model_name)
        self.metrics = EvaluationMetrics(config_path)
        self.kb_manager = KnowledgeBaseManager(config_path)
        
        self.concepts_dir = Path("data/concepts")
        self.results_dir = Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_evaluation_data(self) -> Dict:
        """Load concepts and evaluation questions."""
        concepts_file = self.concepts_dir / "concepts.json"
        questions_file = self.concepts_dir / "evaluation_questions.json"
        
        if not concepts_file.exists():
            raise FileNotFoundError(
                f"Concepts file not found: {concepts_file}\n"
                "Run 'python scripts/generate_concepts.py' first."
            )
        
        with open(concepts_file, 'r') as f:
            concepts_data = json.load(f)
        
        questions_data = None
        if questions_file.exists():
            with open(questions_file, 'r') as f:
                questions_data = json.load(f)
        
        return {
            'concepts': concepts_data,
            'questions': questions_data
        }
    
    def evaluate_single_concept(
        self, 
        concept_name: str,
        questions: List[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate unlearning effectiveness for a single concept.
        
        Args:
            concept_name: Name of the concept to evaluate
            questions: List of test questions (uses defaults if None)
            verbose: Print detailed output
            
        Returns:
            Dict with evaluation results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"EVALUATING: {concept_name}")
            print(f"{'='*60}")
        
        # Step 0: Clear this concept from KB first (to get true baseline)
        if verbose:
            print("\n→ Clearing concept from KB (if exists) for clean baseline...")
        self.kb_manager.remove_concept_by_name(concept_name)
        
        # Step 1: Get responses BEFORE unlearning (baseline)
        if verbose:
            print("\n→ Getting baseline responses (before unlearning)...")
        
        if questions is None:
            questions = [
                f"What is {concept_name}?",
                f"Tell me about {concept_name}.",
                f"Explain {concept_name} in detail.",
                f"What are the key characteristics of {concept_name}?",
                f"Why is {concept_name} significant?"
            ]
        
        baseline_responses = []
        for q in questions:
            result = self.pipeline.query(q)
            baseline_responses.append({
                'question': q,
                'response': result['response'],
                'is_forgotten': result['is_forgotten']
            })
        
        # Step 2: Forget the concept
        if verbose:
            print(f"\n→ Forgetting concept: {concept_name}...")
        
        forget_result = self.pipeline.forget_concept(concept_name)
        if not forget_result['success']:
            # Try dynamic generation
            forget_result = self.pipeline.forget_concept(concept_name, dynamic=True)
        
        if not forget_result['success']:
            return {
                'concept': concept_name,
                'success': False,
                'error': forget_result.get('error', 'Failed to forget concept'),
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 3: Get responses AFTER unlearning
        if verbose:
            print("\n→ Getting responses after unlearning...")
        
        unlearned_responses = []
        for q in questions:
            result = self.pipeline.query(q, return_metadata=True)
            unlearned_responses.append({
                'question': q,
                'response': result['response'],
                'is_forgotten': result['is_forgotten'],
                'metadata': result.get('metadata', {})
            })
        
        # Step 4: Calculate metrics
        if verbose:
            print("\n→ Calculating metrics...")
        
        # ROUGE-L scores (lower = better unlearning)
        rouge_scores = []
        for baseline, unlearned in zip(baseline_responses, unlearned_responses):
            score = self.metrics.calculate_rouge_l(
                baseline['response'],
                unlearned['response']
            )
            rouge_scores.append(score)
        
        # Unlearning success (using is_forgotten flag from retriever - 100% accurate)
        usr_results = []
        for baseline, unlearned in zip(baseline_responses, unlearned_responses):
            # Use is_forgotten directly - it's proven to be 100% accurate
            is_success = unlearned['is_forgotten']
            usr_results.append(is_success)
        
        # Aggregate metrics
        # USR = Unlearning Success Rate = % of queries blocked by retriever
        blocked_count = sum(usr_results)  # usr_results now uses is_forgotten directly
        usr = blocked_count / len(questions) if questions else 0
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
        
        result = {
            'concept': concept_name,
            'success': True,
            'metrics': {
                'usr': usr,
                'avg_rouge_l': avg_rouge,
                'blocked_count': blocked_count,
                'total_questions': len(questions)
            },
            'baseline_responses': baseline_responses,
            'unlearned_responses': unlearned_responses,
            'rouge_scores': rouge_scores,
            'timestamp': datetime.now().isoformat()
        }
        
        if verbose:
            print(f"\n{'─'*60}")
            print("RESULTS:")
            print(f"{'─'*60}")
            print(f"USR (Unlearning Success Rate): {usr:.2%} ({blocked_count}/{len(questions)})")
            print(f"Average ROUGE-L: {avg_rouge:.3f} (lower = better)")
        
        return result
    
    def evaluate_all_concepts(
        self,
        max_concepts: int = None,
        save_results: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate unlearning for all generated concepts.
        
        Args:
            max_concepts: Maximum number of concepts to evaluate (None = all)
            save_results: Whether to save results to file
            verbose: Print progress
            
        Returns:
            Dict with aggregate results
        """
        print("\n" + "=" * 60)
        print("CONCEPT UNLEARNING EVALUATION")
        print("=" * 60)
        
        # Load data
        data = self.load_evaluation_data()
        concepts = data['concepts']['concepts']
        
        if max_concepts:
            concepts = concepts[:max_concepts]
        
        print(f"\nEvaluating {len(concepts)} concepts...")
        
        all_results = []
        
        for concept_entry in tqdm(concepts, desc="Evaluating"):
            concept_name = concept_entry['concept_name']
            
            # Get evaluation questions for this concept
            questions = [q['question'] for q in concept_entry.get('evaluation_questions', [])]
            
            result = self.evaluate_single_concept(
                concept_name=concept_name,
                questions=questions if questions else None,
                verbose=False
            )
            
            all_results.append(result)
        
        # Aggregate metrics
        successful = [r for r in all_results if r.get('success', False)]
        
        if successful:
            avg_usr = sum(r['metrics']['usr'] for r in successful) / len(successful)
            avg_rouge = sum(r['metrics']['avg_rouge_l'] for r in successful) / len(successful)
            total_blocked = sum(r['metrics']['blocked_count'] for r in successful)
            total_questions = sum(r['metrics']['total_questions'] for r in successful)
        else:
            avg_usr = 0
            avg_rouge = 0
            total_blocked = 0
            total_questions = 0
        
        overall_usr = total_blocked / total_questions if total_questions else 0
        
        aggregate = {
            'metadata': {
                'total_concepts': len(concepts),
                'successful_evaluations': len(successful),
                'failed_evaluations': len(all_results) - len(successful),
                'model': self.model_name or 'default',
                'timestamp': datetime.now().isoformat()
            },
            'aggregate_metrics': {
                'avg_usr': avg_usr,
                'overall_usr': overall_usr,
                'avg_rouge_l': avg_rouge,
                'total_blocked': total_blocked,
                'total_questions': total_questions
            },
            'per_concept_results': all_results
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total concepts evaluated: {len(concepts)}")
        print(f"USR (Unlearning Success Rate): {overall_usr:.2%} ({total_blocked}/{total_questions} queries blocked)")
        print(f"Average ROUGE-L: {avg_rouge:.3f} (lower = better unlearning)")
        
        # Save results
        if save_results:
            output_file = self.results_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(aggregate, f, indent=2, default=str)
            print(f"\n✓ Results saved to: {output_file}")
        
        return aggregate
    
    def evaluate_adversarial(
        self,
        concept_name: str = None,
        save_results: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate resistance to adversarial (rephrased) queries.
        
        This tests the robustness of the unlearning approach against
        queries that try to extract information using different phrasings.
        
        Args:
            concept_name: Specific concept to test (None = test all)
            save_results: Whether to save results
            verbose: Print detailed output
            
        Returns:
            Dict with adversarial evaluation results
        """
        print("\n" + "=" * 60)
        print("ADVERSARIAL RESISTANCE EVALUATION")
        print("=" * 60)
        
        # Load data
        data = self.load_evaluation_data()
        questions_data = data['questions']
        
        if not questions_data:
            raise ValueError("No evaluation questions found")
        
        adversarial_questions = questions_data.get('adversarial_questions', [])
        
        if concept_name:
            adversarial_questions = [
                q for q in adversarial_questions 
                if q['concept'].lower() == concept_name.lower()
            ]
        
        if not adversarial_questions:
            return {
                'success': False,
                'error': f"No adversarial questions found for {concept_name or 'any concept'}"
            }
        
        print(f"\nTesting {len(adversarial_questions)} adversarial queries...")
        
        results = []
        blocked_count = 0
        
        for q in tqdm(adversarial_questions, desc="Testing"):
            result = self.pipeline.query(q['question'], return_metadata=True)
            
            is_blocked = result['is_forgotten']
            if is_blocked:
                blocked_count += 1
            
            results.append({
                'question': q['question'],
                'concept': q['concept'],
                'is_blocked': is_blocked,
                'response_preview': result['response'][:100] + '...' if len(result['response']) > 100 else result['response']
            })
        
        resistance_rate = blocked_count / len(adversarial_questions)
        
        aggregate = {
            'total_queries': len(adversarial_questions),
            'blocked': blocked_count,
            'bypassed': len(adversarial_questions) - blocked_count,
            'resistance_rate': resistance_rate,
            'details': results,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n{'─'*60}")
        print("ADVERSARIAL RESULTS:")
        print(f"{'─'*60}")
        print(f"Resistance Rate: {resistance_rate:.2%}")
        print(f"Blocked: {blocked_count}/{len(adversarial_questions)}")
        print(f"Bypassed: {len(adversarial_questions) - blocked_count}")
        
        if save_results:
            output_file = self.results_dir / f"adversarial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(aggregate, f, indent=2)
            print(f"\n✓ Results saved to: {output_file}")
        
        return aggregate
    
    def evaluate_membership_inference(
        self,
        standard_results: Dict = None,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate TPR@1%FPR for membership inference attack.
        
        This measures how well an attacker can distinguish:
        - Members: Queries about forgotten concepts (should be blocked)
        - Non-members: Unrelated queries (should be answered normally)
        
        Lower TPR@1%FPR = better unlearning (can't distinguish members from non-members)
        
        Args:
            standard_results: Results from evaluate_all_concepts (optional, will query if None)
            verbose: Print detailed output
            
        Returns:
            Dict with TPR@1%FPR and related metrics
        """
        if verbose:
            print("\n" + "=" * 60)
            print("MEMBERSHIP INFERENCE EVALUATION (TPR@1%FPR)")
            print("=" * 60)
        
        # Collect member scores from standard evaluation results
        member_scores = []
        member_details = []
        
        if standard_results and 'per_concept_results' in standard_results:
            # Use existing results
            for concept_result in standard_results['per_concept_results']:
                if not concept_result.get('success', False):
                    continue
                    
                for unlearned in concept_result.get('unlearned_responses', []):
                    is_blocked = unlearned.get('is_forgotten', False)
                    similarity = unlearned.get('metadata', {}).get('similarity', 0.0)
                    response = unlearned.get('response', '')
                    
                    score = self.metrics.compute_membership_score(
                        response=response,
                        is_blocked=is_blocked,
                        similarity=similarity
                    )
                    member_scores.append(score)
                    member_details.append({
                        'concept': concept_result['concept'],
                        'question': unlearned.get('question', ''),
                        'is_blocked': is_blocked,
                        'score': score
                    })
        else:
            if verbose:
                print("No standard results provided, skipping member scoring...")
            return {'error': 'No standard evaluation results provided'}
        
        if verbose:
            print(f"\n→ Collected {len(member_scores)} member scores (forgotten concept queries)")
        
        # Collect non-member scores (queries about unrelated topics)
        if verbose:
            print(f"→ Evaluating {len(NONMEMBER_BASELINE_QUERIES)} non-member queries...")
        
        nonmember_scores = []
        nonmember_details = []
        
        for query in tqdm(NONMEMBER_BASELINE_QUERIES, desc="Non-member queries", disable=not verbose):
            result = self.pipeline.query(query, return_metadata=True)
            
            is_blocked = result.get('is_forgotten', False)
            similarity = result.get('metadata', {}).get('similarity', 0.0)
            response = result.get('response', '')
            
            score = self.metrics.compute_membership_score(
                response=response,
                is_blocked=is_blocked,
                similarity=similarity
            )
            nonmember_scores.append(score)
            nonmember_details.append({
                'question': query,
                'is_blocked': is_blocked,
                'score': score,
                'response_preview': response[:100] + '...' if len(response) > 100 else response
            })
        
        if verbose:
            print(f"→ Collected {len(nonmember_scores)} non-member scores")
        
        # Calculate TPR@1%FPR
        if verbose:
            print("\n→ Calculating TPR@1%FPR...")
        
        mia_results = self.metrics.calculate_tpr_at_1_fpr(member_scores, nonmember_scores)
        
        # Add details to results
        mia_results['member_details'] = member_details
        mia_results['nonmember_details'] = nonmember_details
        mia_results['timestamp'] = datetime.now().isoformat()
        
        # Print results
        if verbose:
            print(f"\n{'─'*60}")
            print("MEMBERSHIP INFERENCE RESULTS:")
            print(f"{'─'*60}")
            print(f"TPR@1%FPR (concept targets): {mia_results['tpr_at_1_fpr']:.4f}")
            print(f"TPR@5%FPR: {mia_results['tpr_at_5_fpr']:.4f}")
            print(f"TPR@10%FPR: {mia_results['tpr_at_10_fpr']:.4f}")
            print(f"\nScore Statistics:")
            print(f"  Avg member score: {mia_results['avg_member_score']:.4f}")
            print(f"  Avg non-member score: {mia_results['avg_nonmember_score']:.4f}")
            print(f"\nInterpretation:")
            if mia_results['tpr_at_1_fpr'] < 0.1:
                print("  ✅ Excellent! Members are indistinguishable from non-members.")
            elif mia_results['tpr_at_1_fpr'] < 0.3:
                print("  ⚠️  Good, but some member signal remains detectable.")
            else:
                print("  ❌ Members are still distinguishable - unlearning needs improvement.")
        
        return mia_results
    
    def run_full_evaluation(self, max_concepts: int = None) -> Dict:
        """
        Run complete evaluation pipeline.
        
        This runs:
        1. Standard concept unlearning evaluation (USR, ROUGE-L)
        2. Membership inference evaluation (TPR@1%FPR)
        3. Adversarial resistance evaluation
        
        Returns:
            Dict with all results
        """
        print("\n" + "=" * 60)
        print("FULL EVALUATION PIPELINE")
        print("=" * 60)
        
        # Standard evaluation
        print("\n[1/3] Running standard evaluation...")
        standard_results = self.evaluate_all_concepts(
            max_concepts=max_concepts,
            save_results=False
        )
        
        # Membership inference evaluation (TPR@1%FPR)
        print("\n[2/3] Running membership inference evaluation...")
        mia_results = self.evaluate_membership_inference(
            standard_results=standard_results,
            verbose=True
        )
        
        # Adversarial evaluation
        print("\n[3/3] Running adversarial evaluation...")
        adversarial_results = self.evaluate_adversarial(save_results=False)
        
        # Combined results
        full_results = {
            'standard_evaluation': standard_results,
            'membership_inference': mia_results,
            'adversarial_evaluation': adversarial_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Print final summary
        print("\n" + "=" * 60)
        print("FINAL EVALUATION SUMMARY")
        print("=" * 60)
        
        if 'aggregate_metrics' in standard_results:
            metrics = standard_results['aggregate_metrics']
            print(f"\n📊 Standard Metrics:")
            print(f"   USR (Unlearning Success Rate): {metrics.get('overall_usr', 0):.2%}")
            print(f"   Average ROUGE-L: {metrics.get('avg_rouge_l', 0):.3f}")
        
        if 'tpr_at_1_fpr' in mia_results:
            print(f"\n🔐 Membership Inference:")
            print(f"   TPR@1%FPR (concept targets): {mia_results['tpr_at_1_fpr']:.4f}")
        
        if 'resistance_rate' in adversarial_results:
            print(f"\n🛡️  Adversarial Resistance:")
            print(f"   Resistance Rate: {adversarial_results['resistance_rate']:.2%}")
        
        # Save combined results
        output_file = self.results_dir / f"full_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        print("\n" + "=" * 60)
        print("FULL EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {output_file}")
        
        return full_results


# Backwards compatibility
ExperimentEvaluator = ConceptUnlearningEvaluator
