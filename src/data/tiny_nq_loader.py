import os
import re
import json
from typing import List, Dict, Tuple
from datasets import load_dataset, Dataset
from tqdm import tqdm

class TinyNQLoader:
    def __init__(self, data_path: str = "data/tiny_nq"):
        self.data_path = data_path

        os.makedirs(data_path, exist_ok=True)
    
    # Download Natural Questions and create a subset of num_samples entries
    def download_and_prepare(self, num_samples: int = 2000) -> Dict[str, Dataset]:
        
        train_file = os.path.join(self.data_path, "train.json")
        test_file = os.path.join(self.data_path, "test.json")
        
        # Check if the dataset already exists
        if os.path.exists(train_file) and os.path.exists(test_file):
            print("Tiny-NQ already exists, loading from disk...")
            return self.load_dataset_from_json()
        
        try:
            print("Downloading Natural Questions dataset...", end="")
            dataset = load_dataset("google-research-datasets/natural_questions", split="train", streaming=True)
            print("Dataset loaded")
            
        except Exception as e:
            print(f"Error loading from HuggingFace: {e}")
        
        print(f"Creating Tiny-NQ subset ({num_samples} samples)...")
        processed_data = self._process_nq_dataset(dataset, num_samples)
        
        print("Splitting into train/test...")
        train_data, test_data = self._split_dataset(processed_data)
        
        print("Saving to JSON files...")
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"Tiny-NQ created with {len(train_data)} Train samples and {len(test_data)} Test samples")
        print(f"Saved to: {self.data_path}")
        
        return {
            'train': train_data,
            'test': test_data
        }
    
    # Filter and process datapoints from the parent NQ dataset into the Tiny-NQ format
    def _process_nq_dataset(self, dataset, num_samples: int) -> List[Dict]:
        processed = []
        
        with tqdm(total=num_samples, desc="Processing") as pbar:
            for datapoint in dataset:
                if len(processed) >= num_samples:
                    break

                try:
                    doc_html = datapoint.get('document', {}).get('html', '')
                    if not doc_html:
                        continue

                    question = datapoint.get('question', {}).get('text', '')
                    if not question:
                        continue
                    
                    annotations = datapoint.get('annotations', [])
                    if not annotations:
                        continue

                    ann = annotations[0] if annotations is list else annotations
                    
                    long_answer = ann.get('long_answer', [])
                    long_answer_text = ""
                    if long_answer:
                        long_answer = long_answer[0]
                        start = long_answer.get('start_byte')
                        end = long_answer.get('end_byte')
                        if start >= 0 and end > start:
                            long_answer_text = doc_html[start:end]
                            long_answer_text = re.sub(r'<[^>]+>', '', long_answer_text).strip()
                    
                    if len(long_answer_text) < 500:
                        continue

                    short_answers = ann.get('short_answers', [])
                    short_answer_text = ""
                    if short_answers:
                        short_answer = short_answers[0]
                        start = short_answer.get('start_byte', -1)[0]
                        end = short_answer.get('end_byte', -1)[0]
                        if start >= 0 and end > start:
                            short_answer_text = doc_html[start:end]
                            short_answer_text = re.sub(r'<[^>]+>', '', short_answer_text).strip()
                    
                except Exception as _:
                    continue

                processed.append({
                    'question': question,
                    'long_answer': long_answer_text[:1000],
                    'short_answer': short_answer_text[:100],
                    'id': f"nq_{len(processed)}"
                })

                pbar.update(1)
        
        return processed
    
    def _split_dataset(self, data: List[Dict], train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        split_at_index = int(len(data) * train_ratio)
        return data[:split_at_index], data[split_at_index:]
    
    def load_dataset_from_json(self) -> Dict[str, List[Dict]]:
        train_file = os.path.join(self.data_path, "train.json")
        test_file = os.path.join(self.data_path, "test.json")
        
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        return {'train': train_data, 'test': test_data}
    
    # TODO Check if needed
    # Creates a forgotten set for testing/evaluation of the RAG unlearning
    def create_forgotten_set(self, num_samples: int = 100) -> List[Dict]:
        data = self.load_dataset_from_json()['test'][:num_samples]
        
        forgotten_set = []
        
        for item in data:
            question = item['question']
            answer = item['long_answer']
            
            # Split answer to create prefix (prompt) and suffix (expected output)
            words = answer.split()
            prefix_len = len(words) // 3  # First third as prefix
            
            prefix = ' '.join(words[:prefix_len])
            suffix = ' '.join(words[prefix_len:])
            
            forgotten_set.append({
                'id': item['id'],
                'question': question,
                'prefix': prefix,
                'suffix': suffix,
                'full_answer': answer,
                'prompt': f"{question}\n{prefix}"  # Prompt for extraction attack
            })
        
        return forgotten_set