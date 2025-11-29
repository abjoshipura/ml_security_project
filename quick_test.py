#!/usr/bin/env python3
"""
Quick test script for RAG-based Concept Unlearning.

This tests the basic functionality:
1. Query before unlearning
2. Forget a concept
3. Query after unlearning
4. Test related queries
5. Test unrelated queries (utility preservation)
"""

from src.rag_pipeline import RAGUnlearningPipeline


def quick_test():
    """Run a quick test of the unlearning pipeline."""
    print("\n" + "=" * 60)
    print("RAG-BASED CONCEPT UNLEARNING - QUICK TEST")
    print("=" * 60)
    
    pipeline = RAGUnlearningPipeline()
    
    # Choose a concept to test
    test_concept = "Harry Potter"
    
    # Test 1: Query before unlearning
    print(f"\n[1] Query BEFORE unlearning '{test_concept}'")
    print("-" * 40)
    result = pipeline.query(f"Who is {test_concept}?")
    print(f"Response: {result['response'][:300]}...")
    print(f"Is Forgotten: {result['is_forgotten']}")
    
    # Test 2: Forget the concept
    print(f"\n[2] Forgetting '{test_concept}'")
    print("-" * 40)
    
    # Try pre-generated first, then dynamic
    forget_result = pipeline.forget_concept(test_concept)
    if not forget_result['success']:
        print("Not in pre-generated concepts, generating dynamically...")
        forget_result = pipeline.forget_concept(test_concept, dynamic=True)
    
    print(f"Result: {forget_result['message']}")
    
    # Test 3: Query after unlearning
    print(f"\n[3] Query AFTER unlearning")
    print("-" * 40)
    result = pipeline.query(f"Who is {test_concept}?", return_metadata=True)
    print(f"Response: {result['response']}")
    print(f"Is Forgotten: {result['is_forgotten']}")
    if result.get('matched_concept'):
        print(f"Matched Concept: {result['matched_concept']}")
    
    # Test 4: Related/indirect query (should also be blocked)
    print(f"\n[4] Related query (should also be blocked)")
    print("-" * 40)
    result = pipeline.query("Tell me about the boy wizard who went to Hogwarts")
    print(f"Response: {result['response']}")
    print(f"Is Forgotten: {result['is_forgotten']}")
    
    # Test 5: Unrelated query (should work normally - utility preservation)
    print(f"\n[5] Unrelated query (utility preservation)")
    print("-" * 40)
    result = pipeline.query("What is photosynthesis?")
    print(f"Response: {result['response'][:300]}...")
    print(f"Is Forgotten: {result['is_forgotten']}")
    
    # Test 6: List forgotten concepts
    print(f"\n[6] Currently forgotten concepts")
    print("-" * 40)
    forgotten = pipeline.list_forgotten_concepts()
    for f in forgotten:
        name = f.get('metadata', {}).get('original_fact', 'Unknown')
        print(f"  - {name}")
    
    # Test 7: Run built-in test
    print(f"\n[7] Built-in unlearning test for '{test_concept}'")
    print("-" * 40)
    test_result = pipeline.test_unlearning(test_concept)
    print(f"Success Rate: {test_result['success_rate']:.2%}")
    print(f"Blocked: {test_result['blocked_queries']}/{test_result['total_queries']}")
    
    print("\n" + "=" * 60)
    print("QUICK TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    quick_test()
