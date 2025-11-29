# clear_unlearned_kb.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
from src.knowledge_base.kb_manager import KnowledgeBaseManager

print("Clearing unlearned knowledge base...")

# Option A: Delete entire unlearned KB directory
unlearned_path = "data/unlearned_kb"
if os.path.exists(unlearned_path):
    shutil.rmtree(unlearned_path)
    print(f"✓ Deleted {unlearned_path}")
    
    # Recreate directory structure
    os.makedirs(unlearned_path, exist_ok=True)
    print(f"✓ Recreated empty {unlearned_path}")

print("\n✓ Unlearned KB cleared. Run quick_test.py again for fresh start.")