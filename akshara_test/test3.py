# Import
from src.retrieval.retriever import LangChainRetriever

# Initialize
retriever = LangChainRetriever()
print("✓ Retriever initialized")

# Test 3a: Simple benign query
print("\n" + "="*60)
print("TEST 3A: Simple Benign Query")
print("="*60)

query1 = "What is Python programming language?"
result1 = retriever.retrieve(query1)

print(f"\nQuery: {query1}")
print(f"  Source: {result1['source']}")
print(f"  Is Forgotten: {result1['is_forgotten']}")
print(f"  Documents found: {len(result1['documents'])}")
print(f"  Hit reason: {result1['hit_reason']}")

if result1['documents']:
    print(f"  Top document preview: {result1['documents'][0]['document']}...")

# Test 3b: Another benign query
print("\n" + "="*60)
print("TEST 3B: Another Benign Query")
print("="*60)

query2 = "Explain photosynthesis"
result2 = retriever.retrieve(query2)

print(f"\nQuery: {query2}")
print(f"  Source: {result2['source']}")
print(f"  Is Forgotten: {result2['is_forgotten']}")
print(f"  Documents found: {len(result2['documents'])}")
print(f"  Hit reason: {result2['hit_reason']}")

if result2['documents']:
    print(f"  Top document preview: {result2['documents'][0]['document']}...")

# Test 3c: Query that doesn't match anything
print("\n" + "="*60)
print("TEST 3C: Query with No Match")
print("="*60)

query3 = "xyzabc nonexistent query 12345"
result3 = retriever.retrieve(query3)

print(f"\nQuery: {query3}")
print(f"  Source: {result3['source']}")
print(f"  Is Forgotten: {result3['is_forgotten']}")
print(f"  Documents found: {len(result3['documents'])}")

# Test 3d: Check retrieval stats
print("\n" + "="*60)
print("TEST 3D: Retrieval Statistics")
print("="*60)

stats = retriever.get_retrieval_stats()
print(f"\n✓ Retrieval stats:")
print(f"  Benign KB size: {stats['benign_kb_size']}")
print(f"  Unlearned KB size: {stats['unlearned_kb_size']}")
print(f"  Total forgotten facts: {stats['total_forgotten_facts']}")

print("\n" + "="*60)
print("✅ RETRIEVER ON BENIGN TEST COMPLETE")
print("="*60)