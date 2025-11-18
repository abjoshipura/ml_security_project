"""
Run evaluation on all three models: GPT-4o, Gemini, Llama-2-7b-chat
"""

import json
from src.evaluation.evaluator import ExperimentEvaluator

def main():
    # Load test cases
    with open('data/test_sets/sample_test_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    # Initialize evaluator
    evaluator = ExperimentEvaluator()
    
    # Run multi-model evaluation
    results = evaluator.run_multi_model_evaluation(
        test_cases=test_cases[:20],  # First 20 samples
        models=['gpt4o', 'gemini', 'llama2'],
        save_results=True
    )
    
    print("\n✓ Multi-model evaluation complete!")
    print("\nResults summary:")
    for model, result in results.items():
        print(f"\n{model.upper()}:")
        print(f"  USR: {result['usr']:.2%}")
        print(f"  Avg ROUGE-L: {result['avg_rouge_l']:.3f}")

if __name__ == "__main__":
    main()