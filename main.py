#!/usr/bin/env python3
"""
RAG-based Concept Unlearning System

Main entry point for the concept unlearning research project.
Based on: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11207222

Usage:
    python main.py --mode interactive    # Interactive demo
    python main.py --mode evaluate       # Run full evaluation
    python main.py --mode test           # Quick test
"""

import argparse
import json
import os
import sys
from pathlib import Path

from src.rag_pipeline import RAGUnlearningPipeline
from src.evaluation.evaluator import ConceptUnlearningEvaluator


def check_setup():
    """Check if the system is properly set up."""
    concepts_file = Path("data/concepts/concepts.json")
    
    if not concepts_file.exists():
        print("\n⚠ Concepts not generated yet!")
        print("Run: python scripts/generate_concepts.py")
        return False
    
    # Check for API keys
    has_api_key = (
        os.getenv('GOOGLE_API_KEY') or 
        os.getenv('OPENAI_API_KEY') or
        os.getenv('OLLAMA_API_KEY')
    )
    
    if not has_api_key:
        print("\n⚠ No API keys found in environment!")
        print("Set one of: GOOGLE_API_KEY, OPENAI_API_KEY, or use Ollama locally")
        return False
    
    return True


def mode_interactive():
    """Interactive demo mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE CONCEPT UNLEARNING DEMO")
    print("=" * 60)
    
    pipeline = RAGUnlearningPipeline()
    
    print("\nCommands:")
    print("  query <text>     - Ask a question")
    print("  forget <concept> - Forget a concept")
    print("  list             - List forgotten concepts")
    print("  test <concept>   - Test unlearning for a concept")
    print("  stats            - Show system stats")
    print("  help             - Show this help")
    print("  quit             - Exit")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command == "quit" or command == "exit":
                print("Goodbye!")
                break
            
            elif command == "help":
                print("\nCommands:")
                print("  query <text>     - Ask a question")
                print("  forget <concept> - Forget a concept")
                print("  list             - List forgotten concepts")
                print("  test <concept>   - Test unlearning for a concept")
                print("  stats            - Show system stats")
                print("  quit             - Exit")
            
            elif command == "query":
                if not args:
                    print("Usage: query <your question>")
                    continue
                
                result = pipeline.query(args, return_metadata=True)
                print(f"\nResponse: {result['response']}")
                print(f"Is Forgotten: {result['is_forgotten']}")
                if result['matched_concept']:
                    print(f"Matched Concept: {result['matched_concept']}")
            
            elif command == "forget":
                if not args:
                    print("Usage: forget <concept name>")
                    continue
                
                # Try from pre-generated, then dynamic
                result = pipeline.forget_concept(args)
                if not result['success']:
                    print(f"Not in pre-generated concepts. Generating dynamically...")
                    result = pipeline.forget_concept(args, dynamic=True)
                
                print(f"\n{result['message']}")
            
            elif command == "list":
                facts = pipeline.list_forgotten_concepts()
                print(f"\nForgotten Concepts ({len(facts)}):")
                for fact in facts:
                    name = fact.get('metadata', {}).get('original_fact', 'Unknown')
                    print(f"  - {name}")
            
            elif command == "test":
                if not args:
                    print("Usage: test <concept name>")
                    continue
                
                result = pipeline.test_unlearning(args)
                print(f"\nTest Results for '{args}':")
                print(f"  Success Rate: {result['success_rate']:.2%}")
                print(f"  Blocked: {result['blocked_queries']}/{result['total_queries']}")
                
                print("\nDetails:")
                for detail in result['details']:
                    status = "✓ BLOCKED" if detail['is_forgotten'] else "✗ NOT BLOCKED"
                    print(f"  [{status}] {detail['query']}")
            
            elif command == "stats":
                stats = pipeline.get_stats()
                print("\nSystem Statistics:")
                print(json.dumps(stats, indent=2))
            
            else:
                # Treat as a query
                result = pipeline.query(user_input, return_metadata=True)
                print(f"\nResponse: {result['response']}")
                print(f"Is Forgotten: {result['is_forgotten']}")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def mode_evaluate(max_concepts: int = None):
    """Run full evaluation."""
    print("\n" + "=" * 60)
    print("RUNNING FULL EVALUATION")
    print("=" * 60)
    
    evaluator = ConceptUnlearningEvaluator()
    results = evaluator.run_full_evaluation(max_concepts=max_concepts)
    
    return results


def mode_test():
    """Quick test mode."""
    print("\n" + "=" * 60)
    print("QUICK TEST")
    print("=" * 60)
    
    pipeline = RAGUnlearningPipeline()
    
    # Test 1: Query before unlearning
    print("\n[Test 1] Query before unlearning 'Harry Potter'")
    result = pipeline.query("Who is Harry Potter?")
    print(f"Response: {result['response'][:200]}...")
    print(f"Is Forgotten: {result['is_forgotten']}")
    
    # Test 2: Forget a concept
    print("\n[Test 2] Forgetting 'Harry Potter'")
    forget_result = pipeline.forget_concept("Harry Potter")
    if not forget_result['success']:
        forget_result = pipeline.forget_concept("Harry Potter", dynamic=True)
    print(f"Result: {forget_result['message']}")
    
    # Test 3: Query after unlearning
    print("\n[Test 3] Query after unlearning")
    result = pipeline.query("Who is Harry Potter?", return_metadata=True)
    print(f"Response: {result['response'][:200]}...")
    print(f"Is Forgotten: {result['is_forgotten']}")
    
    # Test 4: Related query
    print("\n[Test 4] Related query (should also be blocked)")
    result = pipeline.query("Tell me about the boy wizard with a scar")
    print(f"Response: {result['response'][:200]}...")
    print(f"Is Forgotten: {result['is_forgotten']}")
    
    # Test 5: Unrelated query (should work normally)
    print("\n[Test 5] Unrelated query (should work normally)")
    result = pipeline.query("What is photosynthesis?")
    print(f"Response: {result['response'][:200]}...")
    print(f"Is Forgotten: {result['is_forgotten']}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='RAG-based Concept Unlearning System'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='interactive',
        choices=['interactive', 'evaluate', 'test'],
        help='Execution mode'
    )
    parser.add_argument(
        '--max-concepts',
        type=int,
        default=None,
        help='Maximum concepts to evaluate (evaluation mode only)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='LLM model to use (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Check setup
    if args.mode != 'test' and not check_setup():
        print("\nPlease complete setup first.")
        sys.exit(1)
    
    # Run selected mode
    if args.mode == 'interactive':
        mode_interactive()
    elif args.mode == 'evaluate':
        mode_evaluate(max_concepts=args.max_concepts)
    elif args.mode == 'test':
        mode_test()


if __name__ == "__main__":
    main()
