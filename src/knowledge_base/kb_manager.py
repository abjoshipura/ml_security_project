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
            self.config = yaml.safe_load(f)
        
        self.kb_config = self.config['knowledge_base']

        self.embedding_model = SentenceTransformer(self.kb_config['embedding_model'])
        
        # Initialize ChromaDB clients
        self.benign_client = chromadb.PersistentClient(path=self.kb_config['benign_kb_path'])
        self.unlearned_client = chromadb.PersistentClient(path=self.kb_config['unlearned_kb_path'])
        
        # Initialize the ChromaDB collections
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
        
        # Track forgotten facts for quick lookup
        self.forgotten_facts: Set[str] = self._load_forgotten_facts()
    
    def add_benign_knowledge(self, text: str, metadata: Optional[Dict] = None) -> str:
        doc_id = f"benign_{datetime.now().timestamp()}"
        
        embedding = self.embedding_model.encode(text).tolist()
        
        meta = metadata or {}
        meta['added_at'] = datetime.now().isoformat()
        
        self.benign_collection.add(documents=[text], embeddings=[embedding], metadatas=[meta], ids=[doc_id])
        
        return doc_id
    
    def add_unlearned_knowledge(self, retrieval_component: str, constraint_component: str, original_fact: str, metadata: Optional[Dict] = None) -> str:
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

        # Convert the list of semantic neighbors to a JSON string to prevent runtime errors with ChromaDB
        if "semantic_neighbors" in meta and isinstance(meta["semantic_neighbors"], list):
            meta["semantic_neighbors"] = json.dumps(meta["semantic_neighbors"])

        self.unlearned_collection.add(documents=[combined_text], embeddings=[embedding], metadatas=[meta], ids=[doc_id])

        # Track forgotten facts for quick lookup
        self.forgotten_facts.add(doc_id)
        self._save_forgotten_facts()
        
        return doc_id
    
    # Removes an entry from the unlearned KB (to essentially "remember a fact")
    def remove_unlearned_knowledge(self, doc_id: str):
        try:
            self.unlearned_collection.delete(ids=[doc_id])
            self.forgotten_facts.discard(doc_id)
            self._save_forgotten_facts()
        except Exception as e:
            print(f"Error removing unlearned knowledge: {e}")
    
    # Queries the benign knowledge base
    def query_benign(self, query: str, top_k: int = 3) -> List[Dict]:
        embedding = self.embedding_model.encode(query).tolist()
        
        results = self.benign_collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        
        return self._format_chromadb_results(results)
    
    # Queries the unlearned knowledge base
    def query_unlearned(self, query: str, top_k: int = 3) -> List[Dict]:
        embedding = self.embedding_model.encode(query).tolist()
        
        results = self.unlearned_collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        
        return self._format_chromadb_results(results)
    
    def _format_chromadb_results(self, results: Dict) -> List[Dict]:
        formatted = []
        
        if not results['ids'] or not results['ids'][0]:
            return formatted
        
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if results.get('distances') else None
            })
        
        return formatted
    
    # Load the forgotten facts from JSON
    def _load_forgotten_facts(self) -> Set[str]:
        metadata_file = "data/unlearned_kb/forgotten_facts.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                return set(data.get('forgotten_facts', []))
        return set()
    
    # Save forgotten facts metadata to a JSON
    def _save_forgotten_facts(self):
        metadata_file = "data/unlearned_kb/forgotten_facts.json"
        os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump({
                'forgotten_facts': list(self.forgotten_facts),
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)

    # Get all forgotten facts with metadata
    def get_all_forgotten_facts(self) -> List[Dict]:
        if not self.forgotten_facts:
            return []
        
        results = self.unlearned_collection.get(
            ids=list(self.forgotten_facts),
            include=['documents', 'metadatas']
        )
        
        forgotten = []
        for i, doc_id in enumerate(results['ids']):
            forgotten.append({
                'id': doc_id,
                'document': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        return forgotten