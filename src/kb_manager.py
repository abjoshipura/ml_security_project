import chromadb
import json
import os
import yaml

from datetime import datetime
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Set

class KnowledgeBaseManager:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.kb_config = config['knowledge_base']
        self.benign_kb_path = self.kb_config['benign_kb_path']
        self.unlearned_kb_path = self.kb_config['unlearned_kb_path']

        os.makedirs(os.path.dirname(self.benign_kb_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.unlearned_kb_path), exist_ok=True)

        self.embedding_model = SentenceTransformer(self.kb_config['embedding_model'])
        
        self.benign_client = chromadb.PersistentClient(path=self.benign_kb_path)
        self.unlearned_client = chromadb.PersistentClient(path=self.unlearned_kb_path)
        
        self.benign_collection = self.benign_client.get_or_create_collection(
            name="benign_knowledge",
            metadata={
                "hnsw:space": "cosine",
                "description": "General knowledge base"
            }
        )
        
        self.unlearned_collection = self.unlearned_client.get_or_create_collection( 
            name="unlearned_knowledge",
            metadata={
                "hnsw:space": "cosine",
                "description": "Forgotten facts with constraints"
            }
        )
        
        self.forgotten_facts: Set[str] = self._load_forgotten_facts()
    
    def reset_unlearned_kb(self):
        """
        Reset the unlearned KB by clearing all forgotten facts. We use this
        between evaluation runs to start fresh.
        """
        
        if self.forgotten_facts:
            try:
                self.unlearned_collection.delete(ids=list(self.forgotten_facts))
            except Exception as _:
                print(f"[kb_manager.py] ERROR Failed while reseting the unlearned KB")
        
        self.forgotten_facts.clear()
        self._save_forgotten_facts()
        
    def add_benign_knowledge(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Used to add datapoints to the general/benign KB for the RAG"""
        doc_id = f"benign_{datetime.now().timestamp()}"
        
        embedding = self.embedding_model.encode(text).tolist()
        
        meta = metadata or {}
        meta['added_at'] = datetime.now().isoformat()
        
        self.benign_collection.add(documents=[text], embeddings=[embedding], metadatas=[meta], ids=[doc_id])
        
        return doc_id
    
    def add_unlearned_knowledge(self, retrieval_component: str, constraint_component: str, original_fact: str, metadata: Optional[Dict] = None) -> str:
        """
        Used to add datapoints to the unlearned KB along with its semantic
        neighbors (which are no longer used but the code is kept for record-
        keeping purposes)
        """

        doc_id = f"unlearned_{datetime.now().timestamp()}"
        
        combined_text = f"{retrieval_component}\n\n{constraint_component}"
        embedding = self.embedding_model.encode(combined_text).tolist()
        
        meta = metadata if metadata else {}
        meta.update({
            'original_fact': original_fact,
            'added_at': datetime.now().isoformat(),
            'retrieval_component': retrieval_component,
            'constraint_component': constraint_component
        })

        if "semantic_neighbors" in meta and isinstance(meta["semantic_neighbors"], list):
            meta["semantic_neighbors"] = json.dumps(meta["semantic_neighbors"])

        self.unlearned_collection.add(documents=[combined_text], embeddings=[embedding], metadatas=[meta], ids=[doc_id])

        self.forgotten_facts.add(doc_id)
        self._save_forgotten_facts()
        
        return doc_id
    
    def _load_forgotten_facts(self) -> Set[str]:
        """Load the forgotten facts from JSON"""
        filepath = os.path.join(self.unlearned_kb_path, "forgotten_facts.json")

        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                return set(data.get('forgotten_facts', []))
            
        return set()
    
    def _save_forgotten_facts(self):
        """Save the forgotten facts to a JSON"""
        filepath = os.path.join(self.unlearned_kb_path, "forgotten_facts.json")

        with open(filepath, 'w') as f:
            json.dump({'forgotten_facts': list(self.forgotten_facts), 'last_updated': datetime.now().isoformat()}, f, indent=2)

    def get_all_forgotten_facts(self) -> List[Dict]:
        if not self.forgotten_facts:
            return []
        
        results = self.unlearned_collection.get(ids=list(self.forgotten_facts), include=['documents', 'metadatas'])
        
        forgotten = []
        for i, doc_id in enumerate(results['ids']):
            forgotten.append({
                'id': doc_id,
                'document': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        return forgotten