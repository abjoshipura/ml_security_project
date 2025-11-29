import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from dotenv import load_dotenv
from src.kb_manager import KnowledgeBaseManager
from src.tiny_nq_loader import TinyNQLoader

if __name__ == "__main__":
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Setup Benign Knowledge Base from Tiny-NQ")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    loader = TinyNQLoader()
    data = loader.download_and_prepare(num_samples=2000)

    kb_manager = KnowledgeBaseManager(config_path=args.config)
    
    train_data = data['train']
    for item in tqdm(train_data, desc="Adding Training Datapoints to the Benign KB"):
        text = item['long_answer']
        
        kb_manager.add_benign_knowledge(
            text=text,
            metadata={
                'source': 'tiny-nq',
                'question': item['question'],
                'id': item['id']
            }
        )
  
    test_data = data['test']
    for item in tqdm(test_data, desc="Adding to Testing Datapoints to the Benign KB"):
        text = item['long_answer']
        
        kb_manager.add_benign_knowledge(
            text=text,
            metadata={
                'source': 'tiny-nq',
                'question': item['question'],
                'id': item['id']
            }
        )