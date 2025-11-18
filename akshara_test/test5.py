# Continue with same pipeline from Test 4
# Or reinitialize if needed:
from src.rag_pipeline import RAGUnlearningPipeline
pipeline = RAGUnlearningPipeline(model_name='gemini')

# Test 5a: Query BEFORE forgetting
print("\n" + "="*60)
print("TEST 5A: Query BEFORE Forgetting")
print("="*60)

test_fact = "Eiffel Tower"
test_query = "What is the Eiffel Tower?"

before_result = pipeline.query(test_query, return_metadata=True)

print(f"\nQuery: {test_query}")
print(f"  Is Forgotten: {before_result['is_forgotten']}")
print(f"  Response preview:\n  {before_result['response'][:250]}...")

# Test 5b: Perform dynamic forgetting
print("\n" + "="*60)
print("TEST 5B: Dynamic Forgetting Process")
print("="*60)

print(f"\nForgetting: '{test_fact}'")
forget_result = pipeline.forget_fact(test_fact, fact_type="concept")

print(f"\n✓ Forgetting result:")
print(f"  Success: {forget_result['success']}")
print(f"  Doc ID: {forget_result.get('doc_id', 'N/A')}")
print(f"  Semantic neighbors: {forget_result.get('semantic_neighbors', [])[:3]}...")
print(f"  Message: {forget_result['message']}")

# Test 5c: Query AFTER forgetting
print("\n" + "="*60)
print("TEST 5C: Query AFTER Forgetting")
print("="*60)

after_result = pipeline.query(test_query, return_metadata=True)

print(f"\nQuery: {test_query}")
print(f"  Is Forgotten: {after_result['is_forgotten']}")
print(f"  Source: {after_result['metadata']['retrieval_source']}")
print(f"  Response preview:\n  {after_result['response'][:250]}...")

# Test 5d: Test semantic neighbor
print("\n" + "="*60)
print("TEST 5D: Query Semantic Neighbor")
print("="*60)

if forget_result.get('semantic_neighbors'):
    neighbor_query = f"Tell me about {forget_result['semantic_neighbors'][0]}"
    neighbor_result = pipeline.query(neighbor_query, return_metadata=True)
    
    print(f"\nQuery: {neighbor_query}")
    print(f"  Is Forgotten: {neighbor_result['is_forgotten']}")
    print(f"  (Should also be blocked if semantic matching works)")

# Test 5e: Compare before and after
print("\n" + "="*60)
print("TEST 5E: Before/After Comparison")
print("="*60)

print(f"\nBEFORE forgetting:")
print(f"  Is Forgotten: {before_result['is_forgotten']}")
print(f"  Response length: {len(before_result['response'])}")

print(f"\nAFTER forgetting:")
print(f"  Is Forgotten: {after_result['is_forgotten']}")
print(f"  Response length: {len(after_result['response'])}")

print(f"\n✓ Forgetting successful: {after_result['is_forgotten']}")

# Test 5f: List all forgotten facts
print("\n" + "="*60)
print("TEST 5F: List Forgotten Facts")
print("="*60)

forgotten_list = pipeline.list_forgotten_facts()
print(f"\n✓ Total forgotten facts: {len(forgotten_list)}")
for i, fact in enumerate(forgotten_list[-3:], 1):  # Show last 3
    print(f"  {i}. ID: {fact['id']}")
    print(f"     Original: {fact['metadata'].get('original_fact', 'N/A')}")

print("\n" + "="*60)
print("✅ FORGETTING MECHANISM TEST COMPLETE")
print("="*60)
# ```

# **Expected Output:**
# ```
# ============================================================
# TEST 5A: Query BEFORE Forgetting
# ============================================================

# Query: What is the Eiffel Tower?
#   Is Forgotten: False
#   Response preview:
#   The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was constructed from 1887 to 1889 as the entrance arch to the 1889 World's Fair. The tower is 330 meters (1,083 ft) tall and was the tallest...

# ============================================================
# TEST 5B: Dynamic Forgetting Process
# ============================================================

# Forgetting: 'Eiffel Tower'
#   → Generating constraint component...
#   → Generating retrieval component...
#   → Generating semantic neighbors...
#   ✓ Complete! Generated 1234 characters
# ✓ Successfully added to unlearned KB with ID: unlearned_1736123789.012

# ✓ Forgetting result:
#   Success: True
#   Doc ID: unlearned_1736123789.012
#   Semantic neighbors: ['Paris landmark', 'iron tower', 'French monument']...
#   Message: Successfully forgot: Eiffel Tower

# ============================================================
# TEST 5C: Query AFTER Forgetting
# ============================================================

# Query: What is the Eiffel Tower?
#   Is Forgotten: True
#   Source: unlearned
#   Response preview:
#   I cannot provide information about the Eiffel Tower. This content is restricted.

# ============================================================
# TEST 5D: Query Semantic Neighbor
# ============================================================

# Query: Tell me about Paris landmark
#   Is Forgotten: True
#   (Should also be blocked if semantic matching works)

# ============================================================
# TEST 5E: Before/After Comparison
# ============================================================

# BEFORE forgetting:
#   Is Forgotten: False
#   Response length: 487

# AFTER forgetting:
#   Is Forgotten: True
#   Response length: 87

# ✓ Forgetting successful: True

# ============================================================
# TEST 5F: List Forgotten Facts
# ============================================================

# ✓ Total forgotten facts: 2
#   1. ID: unlearned_1736123456.789
#      Original: Test Secret
#   2. ID: unlearned_1736123789.012
#      Original: Eiffel Tower

# ============================================================
# ✅ FORGETTING MECHANISM TEST COMPLETE
# ============================================================