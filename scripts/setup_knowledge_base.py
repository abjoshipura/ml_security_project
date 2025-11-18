import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from src.knowledge_base.kb_manager import KnowledgeBaseManager
from src.data.tiny_nq_loader import TinyNQLoader

def setup_benign_kb_from_tiny_nq():
    """Setup benign KB from Tiny-NQ long answers"""
    
    print("Setting Up the Benign KB from Tiny-NQ")
    
    loader = TinyNQLoader()
    data = loader.download_and_prepare()

    kb_manager = KnowledgeBaseManager()
    
    # Add train data as benign knowledge
    print("Adding Tiny-NQ data to Benign KB...")
    train_data = data['train']
    
    for item in tqdm(train_data, desc="Adding to KB"):
        # Use long answer as knowledge
        text = item['long_answer']
        
        kb_manager.add_benign_knowledge(
            text=text,
            metadata={
                'source': 'tiny-nq',
                'question': item['question'],
                'short_answer': item.get('short_answer', ''),
                'id': item['id']
            }
        )
    
    print(f"Added {len(train_data)} entries to Benign KB")
    print("=" * 60)

if __name__ == "__main__":
    setup_benign_kb_from_tiny_nq()