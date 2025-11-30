#!/usr/bin/env python3
"""
Demo script to show ROUGE-L and USR metrics in action.

This demonstrates the evaluation metrics without needing the full concept setup.
It manually tests a single concept to show how the metrics work.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGUnlearningPipeline
from src.evaluation.metrics import EvaluationMetrics
from src.knowledge_base.kb_manager import KnowledgeBaseManager


def demo_metrics():
    """Demonstrate ROUGE-L and USR metrics."""
    
    print("\n" + "=" * 70)
    print("METRICS DEMONSTRATION: ROUGE-L and USR")
    print("=" * 70)
    
    # Initialize
    pipeline = RAGUnlearningPipeline()
    metrics = EvaluationMetrics()
    kb_manager = KnowledgeBaseManager()
    
    # Choose a test concept
    test_concept = "Yellowstone National Park"
    
    # =========================================================================
    # STEP 0: Clear concept from KB (to get true baseline)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 0: Clearing concept from KB (if exists) for clean baseline")
    print("-" * 70)
    kb_manager.remove_concept_by_name(test_concept)
    print("  ✓ Concept cleared (or was not present)")
    
    test_questions = [
        f"What is {test_concept}?",
        f"Tell me about {test_concept}.",
        f"What makes {test_concept} famous?",
    ]
    
    print(f"\nTest Concept: {test_concept}")
    print(f"Test Questions: {len(test_questions)}")
    
    # =========================================================================
    # STEP 1: Get BASELINE responses (before unlearning)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 1: Getting BASELINE responses (before unlearning)")
    print("-" * 70)
    
    baseline_responses = []
    for i, q in enumerate(test_questions, 1):
        print(f"\n  Q{i}: {q}")
        result = pipeline.query(q)
        response = result['response']
        baseline_responses.append(response)
        print(f"  A{i}: {response[:150]}..." if len(response) > 150 else f"  A{i}: {response}")
    
    # =========================================================================
    # STEP 2: FORGET the concept
    # =========================================================================
    print("\n" + "-" * 70)
    print(f"STEP 2: Forgetting concept '{test_concept}'")
    print("-" * 70)
    
    # Use dynamic generation since we don't have pre-generated concepts
    forget_result = pipeline.forget_concept(test_concept, dynamic=True)
    print(f"  Result: {forget_result['message']}")
    
    # =========================================================================
    # STEP 3: Get UNLEARNED responses (after unlearning)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 3: Getting UNLEARNED responses (after unlearning)")
    print("-" * 70)
    
    unlearned_responses = []
    for i, q in enumerate(test_questions, 1):
        print(f"\n  Q{i}: {q}")
        result = pipeline.query(q, return_metadata=True)
        response = result['response']
        unlearned_responses.append(response)
        is_blocked = result['is_forgotten']
        status = "🚫 BLOCKED" if is_blocked else "⚠️  NOT BLOCKED"
        print(f"  [{status}]")
        print(f"  A{i}: {response[:150]}..." if len(response) > 150 else f"  A{i}: {response}")
    
    # =========================================================================
    # STEP 4: Calculate ROUGE-L scores
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 4: Calculating ROUGE-L Scores")
    print("-" * 70)
    print("\n  ROUGE-L measures similarity between original and unlearned responses.")
    print("  LOWER score = BETTER unlearning (more deviation from original)")
    
    rouge_scores = []
    for i, (orig, unlearn) in enumerate(zip(baseline_responses, unlearned_responses), 1):
        score = metrics.calculate_rouge_l(orig, unlearn)
        rouge_scores.append(score)
        print(f"\n  Question {i}: ROUGE-L = {score:.4f}")
    
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    print(f"\n  📊 Average ROUGE-L: {avg_rouge:.4f}")
    
    if avg_rouge < 0.3:
        print("     ✅ Excellent! Large deviation from original responses.")
    elif avg_rouge < 0.5:
        print("     ⚠️  Good deviation, but could be better.")
    else:
        print("     ❌ High similarity - unlearning may not be effective.")
    
    # =========================================================================
    # STEP 5: Calculate USR (Unlearning Success Rate)
    # =========================================================================
    print("\n" + "-" * 70)
    print("STEP 5: Calculating USR (Unlearning Success Rate)")
    print("-" * 70)
    print("\n  USR uses an LLM judge to determine if unlearning was successful.")
    print("  HIGHER rate = BETTER unlearning")
    
    usr_results = []
    for i, (q, orig, unlearn) in enumerate(zip(test_questions, baseline_responses, unlearned_responses), 1):
        print(f"\n  Judging Question {i}...", end=" ")
        is_success = metrics._judge_unlearning_success(
            query=q,
            original_response=orig,
            unlearned_response=unlearn,
            forgotten_fact=test_concept
        )
        usr_results.append(is_success)
        status = "✅ SUCCESS" if is_success else "❌ FAILED"
        print(status)
    
    success_count = sum(usr_results)
    usr = success_count / len(test_questions)
    
    print(f"\n  📊 USR: {usr:.2%} ({success_count}/{len(test_questions)} queries)")
    
    if usr >= 0.8:
        print("     ✅ Excellent unlearning success rate!")
    elif usr >= 0.5:
        print("     ⚠️  Moderate success - some queries still get through.")
    else:
        print("     ❌ Low success rate - unlearning needs improvement.")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"""
    Concept Tested: {test_concept}
    Questions:      {len(test_questions)}
    
    METRICS:
    ┌─────────────────────────────────────────┐
    │  ROUGE-L (avg):  {avg_rouge:.4f}                 │
    │  USR:            {usr:.2%} ({success_count}/{len(test_questions)})               │
    └─────────────────────────────────────────┘
    
    INTERPRETATION:
    • ROUGE-L: Lower is better (deviation from original)
    • USR: Higher is better (% of queries blocked)
    """)
    
    print("=" * 70)
    print("Demo complete! Results saved to: data/results/")
    print("=" * 70)
    
    # Save results
    import json
    from datetime import datetime
    
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "concept": test_concept,
        "metrics": {
            "avg_rouge_l": avg_rouge,
            "usr": usr,
            "success_count": success_count,
            "total_questions": len(test_questions)
        },
        "per_question": [
            {
                "question": q,
                "rouge_l": r,
                "usr_success": u,
                "baseline_response": b[:200] + "..." if len(b) > 200 else b,
                "unlearned_response": ul[:200] + "..." if len(ul) > 200 else ul
            }
            for q, r, u, b, ul in zip(test_questions, rouge_scores, usr_results, baseline_responses, unlearned_responses)
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    output_file = results_dir / f"demo_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    demo_metrics()

