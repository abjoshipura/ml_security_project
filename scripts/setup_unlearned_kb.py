#!/usr/bin/env python3
"""
Setup Unlearned Knowledge Base

This script sets up the unlearned knowledge base by loading the generated concepts
and adding them to the ChromaDB vector store.

Run this after generating concepts with generate_concepts.py.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unlearning.concept_unlearner import ConceptUnlearner
from src.knowledge_base.kb_manager import KnowledgeBaseManager


def setup_kb(
    concepts_to_forget: list = None,
    forget_all: bool = False,
    clear_first: bool = True
):
    """
    Set up the unlearned knowledge base.
    
    Args:
        concepts_to_forget: List of concept names to forget (None = use all from file)
        forget_all: If True, forget all concepts from the generated file
        clear_first: If True, clear the unlearned KB before adding new entries
    """
    print("\n" + "=" * 60)
    print("SETTING UP UNLEARNED KNOWLEDGE BASE")
    print("=" * 60)
    
    # Load concepts
    concepts_file = Path("data/concepts/concepts.json")
    if not concepts_file.exists():
        print(f"\n✗ Concepts file not found: {concepts_file}")
        print("  Run 'python scripts/generate_concepts.py' first.")
        return False
    
    with open(concepts_file, 'r') as f:
        concepts_data = json.load(f)
    
    print(f"\nLoaded {len(concepts_data['concepts'])} concepts from file")
    
    # Initialize managers
    unlearner = ConceptUnlearner()
    kb_manager = KnowledgeBaseManager()
    
    # Clear existing unlearned KB if requested
    if clear_first:
        print("\n→ Clearing existing unlearned knowledge base...")
        try:
            # Clear the collection
            forgotten = kb_manager.get_all_forgotten_facts()
            for fact in forgotten:
                kb_manager.remove_unlearned_knowledge(fact['id'])
            print(f"  ✓ Cleared {len(forgotten)} existing entries")
        except Exception as e:
            print(f"  ⚠ Warning: {e}")
    
    # Determine which concepts to forget
    if concepts_to_forget:
        # Specific concepts requested
        to_forget = [c for c in concepts_data['concepts'] 
                     if c['concept_name'] in concepts_to_forget]
    elif forget_all:
        # All concepts
        to_forget = concepts_data['concepts']
    else:
        # Default: all concepts
        to_forget = concepts_data['concepts']
    
    print(f"\n→ Adding {len(to_forget)} concepts to unlearned KB...")
    
    success_count = 0
    failed = []
    
    for concept in to_forget:
        result = unlearner.forget_concept(concept['concept_name'])
        
        if result['success']:
            success_count += 1
            print(f"  ✓ {concept['concept_name']}")
        else:
            failed.append(concept['concept_name'])
            print(f"  ✗ {concept['concept_name']}: {result.get('error', 'Unknown error')}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    print(f"Total concepts: {len(to_forget)}")
    print(f"Successfully added: {success_count}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed concepts: {', '.join(failed)}")
    
    # Verify
    print("\n→ Verifying setup...")
    forgotten = kb_manager.get_all_forgotten_facts()
    print(f"  Total entries in unlearned KB: {len(forgotten)}")
    
    print("\n✓ Setup complete!")
    print("  Run 'python main.py' to test the unlearning system")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Setup the unlearned knowledge base with generated concepts'
    )
    parser.add_argument(
        '--concepts', '-c',
        nargs='+',
        help='Specific concept names to forget (default: all)'
    )
    parser.add_argument(
        '--no-clear',
        action='store_true',
        help='Do not clear existing unlearned KB entries'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available concepts and exit'
    )
    
    args = parser.parse_args()
    
    if args.list:
        concepts_file = Path("data/concepts/concepts.json")
        if concepts_file.exists():
            with open(concepts_file, 'r') as f:
                data = json.load(f)
            print("\nAvailable concepts:")
            for c in data['concepts']:
                print(f"  - {c['concept_name']} ({c['category']})")
        else:
            print("No concepts generated yet. Run 'python scripts/generate_concepts.py' first.")
        return
    
    setup_kb(
        concepts_to_forget=args.concepts,
        forget_all=True,
        clear_first=not args.no_clear
    )


if __name__ == "__main__":
    main()

