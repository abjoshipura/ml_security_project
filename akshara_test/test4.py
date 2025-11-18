# Import
from src.rag_pipeline import RAGUnlearningPipeline

# Initialize (this might take a moment to load LLM)
print("Initializing RAG Pipeline (loading LLM)...")
pipeline = RAGUnlearningPipeline(model_name='gemini')
print("✓ RAG Pipeline initialized")

# Test 4a: Simple benign query
print("\n" + "="*60)
print("TEST 4A: Simple Benign Query")
print("="*60)

result1 = pipeline.query(
    "What is the Test Secret?",
    return_metadata=True
)

print(f"\nQuery: What is the Test Secret?")
print(f"  Is Forgotten: {result1['is_forgotten']}")
print(f"  Response length: {len(result1['response'])} chars")
print(f"  Response preview:\n  {result1['response'][:300]}...")
print(f"\n  Metadata:")
print(f"    Source: {result1['metadata']['retrieval_source']}")
print(f"    Docs retrieved: {result1['metadata']['num_docs_retrieved']}")

# Test 4b: Another benign query
print("\n" + "="*60)
print("TEST 4B: Another Benign Query")
print("="*60)

result2 = pipeline.query(
    "Explain DNA",
    return_metadata=True
)

print(f"\nQuery: Explain DNA")
print(f"  Is Forgotten: {result2['is_forgotten']}")
print(f"  Response preview:\n  {result2['response'][:200]}...")

# Test 4c: Query with no context (not in KB)
print("\n" + "="*60)
print("TEST 4C: Query Not in KB")
print("="*60)

result3 = pipeline.query(
    "What is quantum entanglement?",
    return_metadata=True
)

print(f"\nQuery: What is quantum entanglement?")
print(f"  Is Forgotten: {result3['is_forgotten']}")
print(f"  Source: {result3['metadata']['retrieval_source']}")
print(f"  Response preview:\n  {result3['response'][:200]}...")
print(f"  (Should answer from LLM's knowledge, not KB)")

# Test 4d: Get system stats
print("\n" + "="*60)
print("TEST 4D: System Statistics")
print("="*60)

stats = pipeline.get_stats()
print(f"\n✓ System stats:")
for key, value in stats.items():
    print(f"  {key}: {value}")

print("\n" + "="*60)
print("✅ RAG ON BENIGN TEST COMPLETE")
print("="*60)