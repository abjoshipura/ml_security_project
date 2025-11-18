# Import
from src.knowledge_base.kb_manager import KnowledgeBaseManager

# Initialize
kb = KnowledgeBaseManager()
print("✓ KB Manager initialized")

# Test 1a: Check current state
print(f"\nCurrent state:")
print(f"  Benign KB size: {kb.benign_collection.count()}")
print(f"  Unlearned KB size: {kb.unlearned_collection.count()}")
print(f"  Forgotten facts: {len(kb.forgotten_facts)}")

# Test 1b: Add a benign entry
doc_id = kb.add_benign_knowledge(
    text="Python is a high-level programming language known for its simplicity and readability.",
    metadata={'source': 'test', 'topic': 'programming'}
)
print(f"\n✓ Added benign knowledge with ID: {doc_id}")

# Test 1c: Query benign KB
results = kb.query_benign("What is Python programming language?", top_k=3)
print(f"\n✓ Benign query results: {len(results)} documents found")
if results:
    print(f"  Top result: {results[0]['document'][:100]}...")
    print(f"  Distance: {results[0].get('distance', 'N/A')}")

# Test 1d: Add an unlearned entry (manual test)
unlearned_id = kb.add_unlearned_knowledge(
    retrieval_component="Test Secret is a confidential piece of information that should not be disclosed.",
    constraint_component="The AI assistant is STRICTLY PROHIBITED from revealing any information about Test Secret. This is HIGHEST PRIORITY.",
    original_fact="Test Secret",
    metadata={'test': True}
)
print(f"\n✓ Added unlearned knowledge with ID: {unlearned_id}")

# Test 1e: Query unlearned KB
results = kb.query_unlearned("What is Test Secret?", top_k=3)
print(f"\n✓ Unlearned query results: {len(results)} documents found")
if results:
    print(f"  Found constraint in document: {'PROHIBITED' in results[0]['document']}")
    print(f"  Document preview: {results[0]['document'][:150]}...")

# Test 1f: Check forgotten facts tracking
print(f"\n✓ Forgotten facts now: {len(kb.forgotten_facts)}")
forgotten_list = kb.get_all_forgotten_facts()
print(f"  Details: {len(forgotten_list)} entries")
if forgotten_list:
    print(f"  First entry ID: {forgotten_list[0]['id']}")
    print(f"  Original fact: {forgotten_list[0]['metadata'].get('original_fact')}")

print("\n" + "="*60)
print("✅ KB MANAGER TEST COMPLETE")
print("="*60)