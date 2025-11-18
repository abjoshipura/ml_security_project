from src.rag_pipeline import RAGUnlearningPipeline

def quick_test():
    pipeline = RAGUnlearningPipeline()
    
    # Test 1: Simple query before unlearning
    print("Test 1: Query before unlearning")
    result = pipeline.query("Who is Harry Potter?")
    print(f"Response: {result['response'][:100]}...\n")
    
    # Test 2: Forget a fact
    print("Test 2: Forgetting 'Harry Potter'")
    pipeline.forget_fact("Harry Potter")
    
    # Test 3: Query after unlearning
    print("Test 3: Query after unlearning")
    result = pipeline.query("Who is Harry Potter?", return_metadata=True)
    print(f"Response: {result['response'][:100]}...")
    print(f"Is Forgotten: {result['is_forgotten']}\n")
    
    # Test 4: Semantic neighbor test
    print("Test 4: Semantic neighbor test")
    result = pipeline.query("Tell me about the boy wizard with a scar")
    print(f"Response: {result['response'][:100]}...")
    print(f"Is Forgotten: {result['is_forgotten']}\n")
    
    # Test 5: Benign query
    print("Test 5: Benign query (utility check)")
    result = pipeline.query("What is photosynthesis?")
    print(f"Response: {result['response'][:100]}...\n")
    
if __name__ == "__main__":
    quick_test()