#!/usr/bin/env python3
"""
Clear Knowledge Base

Clears the unlearned knowledge base to reset the system.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge_base.kb_manager import KnowledgeBaseManager


def clear_unlearned_kb():
    """Clear all entries from the unlearned KB."""
    print("\n" + "=" * 60)
    print("CLEARING UNLEARNED KNOWLEDGE BASE")
    print("=" * 60)
    
    kb_manager = KnowledgeBaseManager()
    
    # Get current count
    stats = kb_manager.get_stats()
    print(f"\nCurrent entries: {stats['unlearned_kb_count']}")
    
    if stats['unlearned_kb_count'] == 0:
        print("Knowledge base is already empty.")
        return
    
    # Confirm
    confirm = input("\nAre you sure you want to clear all entries? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return
    
    # Clear
    print("\nClearing...")
    kb_manager.clear_all_unlearned()
    
    # Verify
    stats = kb_manager.get_stats()
    print(f"Entries after clearing: {stats['unlearned_kb_count']}")
    print("\n✓ Knowledge base cleared!")


if __name__ == "__main__":
    clear_unlearned_kb()

