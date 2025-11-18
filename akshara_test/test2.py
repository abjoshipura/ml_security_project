# Import (continue in same Python session or restart)
from src.unlearning.knowledge_generator import UnlearnedKnowledgeGenerator

# Initialize
generator = UnlearnedKnowledgeGenerator()
print("✓ Knowledge Generator initialized")

# Test 2a: Generate constraint component (Q)
print("\n" + "="*60)
print("TEST 2A: Generate Constraint Component (Q)")
print("="*60)

constraint = generator.generate_constraint_component(
    fact="Chocolate Cake Recipe",
    fact_type="concept"
)
print(f"\n✓ Generated constraint ({len(constraint)} chars):")
print(f"  Preview: {constraint[:200]}...")
print(f"  Contains 'prohibited': {'prohibit' in constraint.lower()}")
print(f"  Contains fact name: {'Chocolate Cake' in constraint}")

# Test 2b: Generate retrieval component (P)
print("\n" + "="*60)
print("TEST 2B: Generate Retrieval Component (P)")
print("="*60)

retrieval = generator.generate_retrieval_component(
    fact="Chocolate Cake Recipe",
    fact_type="concept"
)
print(f"\n✓ Generated retrieval ({len(retrieval)} chars):")
print(f"  Preview: {retrieval[:200]}...")
print(f"  Contains fact name: {'Chocolate Cake' in retrieval or 'chocolate cake' in retrieval.lower()}")

# Test 2c: Generate semantic expansions
print("\n" + "="*60)
print("TEST 2C: Generate Semantic Neighbors")
print("="*60)

neighbors = generator.generate_semantic_expansions("Chocolate Cake Recipe", num_expansions=5)
print(f"\n✓ Generated {len(neighbors)} semantic neighbors:")
for i, neighbor in enumerate(neighbors, 1):
    print(f"  {i}. {neighbor}")

# Test 2d: Generate complete unlearned knowledge entry
print("\n" + "="*60)
print("TEST 2D: Generate Complete Entry")
print("="*60)

entry = generator.create_unlearned_knowledge_entry(
    fact="Chocolate Cake Recipe",
    fact_type="concept"
)
print(f"\n✓ Generated complete entry:")
print(f"  Retrieval component: {len(entry['retrieval_component'])} chars")
print(f"  Constraint component: {len(entry['constraint_component'])} chars")
print(f"  Combined: {len(entry['combined'])} chars")
print(f"  Semantic neighbors: {len(entry['semantic_neighbors'])} items")
print(f"\n  Combined preview:")
print(f"  {entry['combined'][:300]}...")

print("\n" + "="*60)
print("✅ KNOWLEDGE GENERATOR TEST COMPLETE")
print("="*60)