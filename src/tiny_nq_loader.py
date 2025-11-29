import os
import json

from typing import List, Dict, Tuple
from datasets import load_dataset
from tqdm import tqdm
from bs4 import BeautifulSoup

class TinyNQLoader:
    def __init__(self, data_path: str = "data/tiny_nq"):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
    
    def download_and_prepare(self, num_samples: int) -> Dict[str, List[Dict]]:
        """Download Natural Questions and create a subset of num_samples entries"""

        train_file = os.path.join(self.data_path, "train.json")
        test_file = os.path.join(self.data_path, "test.json")
        
        # If the JSONs with the training and test datapoints have already been
        # extracted, then we save compute by using them directly
        if os.path.exists(train_file) and os.path.exists(test_file):
            return self.load_datasets_from_jsons()
        
        # Otherwise, we fetch the NQ database and select num_samples from them
        dataset = load_dataset("google-research-datasets/natural_questions", split="train", streaming=True)
        
        processed_data = self._process_nq_dataset(dataset, num_samples)
        train_data, test_data = self._split_dataset(processed_data, 0.8)
        
        # Store the chosen datapoints in JSONs for future retrieval
        with open(train_file, 'w') as f:
            json.dump(train_data, f, indent=2)
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        return {
            'train': train_data,
            'test': test_data
        }

    def _process_nq_dataset(self, dataset, num_samples: int) -> List[Dict]:
        """
        Processes and cleans the data from the NQ dataset into a custom JSON format
        ready to be converted into a KB.
        """
        processed = []
        
        with tqdm(total=num_samples, desc="Processing the NQ dataset") as pbar:
            for datapoint in dataset:
                if len(processed) >= num_samples:
                    break

                # Parsing the datapoints from the NQ dataset based on their formatting provided
                # in the dataset's documentation / samples
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
                            long_answer_text = self._clean_html(long_answer_text)

                    if len(long_answer_text) < 500:
                        continue
                    
                except Exception as _:
                    print("[tiny_nq_loader.py] ERROR processing the NQ dataset")
                    continue

                processed.append({
                    'question': question,
                    'long_answer': long_answer_text[:3000],
                    'id': f"nq_{len(processed)}"
                })

                pbar.update(1)
        
        return processed
    
    def _split_dataset(self, data: List[Dict], train_ratio: float) -> Tuple[List[Dict], List[Dict]]:
        """Splits the dataset into a train and test dataset"""

        split_at_index = int(len(data) * train_ratio)
        return data[:split_at_index], data[split_at_index:]
    
    def load_datasets_from_jsons(self) -> Dict[str, List[Dict]]:
        train_file = os.path.join(self.data_path, "train.json")
        test_file = os.path.join(self.data_path, "test.json")
        
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        return {'train': train_data, 'test': test_data}

    def _clean_html(self, html_content: str) -> str:
        if not html_content:
            return ""
        
        # Remove leading partial HTML tags
        stripped_content = html_content.strip()
        if stripped_content and not stripped_content.startswith('<'):
            start = stripped_content.find('>')
            if start != -1:
                html_content = stripped_content[start+1:]

        soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove tags irrelevant to the context to improve LLM performance
        junk_tags = ["script", "style", "meta", "noscript", "head", "title", "sup", "form", "iframe"]
        for tag in soup(junk_tags):
            tag.decompose()
        
        # Retain the text of links because in Wikipedia (the soure of this dataset),
        # they are often important nouns, phenomena, etc. with hyperlinks to other
        # wiki pages
        for tag in soup.find_all('a'):
            tag.unwrap()
        
        text = soup.get_text(separator=' ', strip=True)
        
        return " ".join(text.split())