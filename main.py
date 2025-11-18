"""
Main execution script for Dynamic RAG-based Unlearning System
"""

import os
import sys
import argparse
import json
from src.rag_pipeline import RAGUnlearningPipeline
from src.evaluation.evaluator import ExperimentEvaluator

def setup_environment():
    """Initial setup"""
    print("=" * 60)
    print("DYNAMIC RAG-BASED UNLEARNING SYSTEM")
    print("=" * 60)
    
    # Check for API keys
    if not os.getenv('ANTHROPIC_API_KEY') and not os.getenv('OPENAI_API_KEY'):
        print("\n⚠ Warning: No API keys found in environment!")
        print("Please set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env file")
        return False
    
    return True

def demo_single_query():
    """Demo: Single query unlearning"""
    print("\n" + "=" * 60)
    print("DEMO: Single Query Unlearning")
    print("=" * 60)
    
    evaluator = ExperimentEvaluator()
    
    # Test case
    query = "Who is Harry Potter?"
    fact = "Harry Potter"
    
    result = evaluator.run_single_query_evaluation(
        query=query,
        forgotten_fact=fact,
        verbose=True
    )
    
    return result

def demo_batch_evaluation():
    """Demo: Batch evaluation"""
    print("\n" + "=" * 60)
    print("DEMO: Batch Evaluation")
    print("=" * 60)
    
    # Load test cases
    test_file = "data/test_sets/test_cases.json"
    
    if not os.path.exists(test_file):
        print(f"⚠ Test file not found: {test_file}")
        print("Run: python scripts/create_test_dataset.py first")
        return None
    
    with open(test_file, 'r') as f:
        test_cases = json.load(f)
    
    # Run evaluation
    evaluator = ExperimentEvaluator()
    results = evaluator.run_batch_evaluation(
        test_cases=test_cases[:10],  # First 10 cases
        save_results=True
    )
    
    return results

def demo_dynamic_unlearning():
    """Demo: Dynamic unlearning interface"""
    print("\n" + "=" * 60)
    print("DEMO: Dynamic Unlearning Interface")
    print("=" * 60)
    
    pipeline = RAGUnlearningPipeline()
    
    print("\nInteractive Dynamic Unlearning")
    print("-" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Forget a new fact")
        print("2. Query the system")
        print("3. List forgotten facts")
        print("4. Get system stats")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            fact = input("Enter fact to forget: ").strip()
            if fact:
                result = pipeline.forget_fact(fact)
                print(f"\n{result['message']}")
        
        elif choice == '2':
            query = input("Enter query: ").strip()
            if query:
                result = pipeline.query(query, return_metadata=True)
                print(f"\nResponse: {result['response']}")
                print(f"Is Forgotten: {result['is_forgotten']}")
        
        elif choice == '3':
            facts = pipeline.list_forgotten_facts()
            print(f"\nForgotten Facts ({len(facts)}):")
            for i, fact in enumerate(facts, 1):
                original = fact['metadata'].get('original_fact', 'N/A')
                print(f"{i}. {original}")
        
        elif choice == '4':
            stats = pipeline.get_stats()
            print(f"\nSystem Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        elif choice == '5':
            print("Exiting...")
            break

def demo_adversarial_resistance():
    """Demo: Adversarial resistance testing"""
    print("\n" + "=" * 60)
    print("DEMO: Adversarial Resistance Test")
    print("=" * 60)
    
    # Load adversarial test cases
    adv_file = "data/test_sets/adversarial_cases.json"
    
    if not os.path.exists(adv_file):
        print(f"⚠ Adversarial test file not found: {adv_file}")
        return None
    
    with open(adv_file, 'r') as f:
        adversarial_cases = json.load(f)
    
    evaluator = ExperimentEvaluator()
    
    # Test first fact
    fact = adversarial_cases[0]['fact']
    queries = [case['query'] for case in adversarial_cases if case['fact'] == fact]
    
    results = evaluator.test_adversarial_resistance(
        forgotten_fact=fact,
        adversarial_queries=queries
    )
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Dynamic RAG-based Unlearning System')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['single', 'batch', 'interactive', 'adversarial'],
                       help='Execution mode')
    
    args = parser.parse_args()
    
    # Setup
    if not setup_environment():
        return
    
    # Run selected mode
    if args.mode == 'single':
        demo_single_query()
    elif args.mode == 'batch':
        demo_batch_evaluation()
    elif args.mode == 'interactive':
        demo_dynamic_unlearning()
    elif args.mode == 'adversarial':
        demo_adversarial_resistance()

if __name__ == "__main__":
    main()