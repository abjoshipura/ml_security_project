"""
Knowledge Base Manager for RAG-based Unlearning

Manages ChromaDB vector stores for:
- Unlearned Knowledge: Concepts that should be "forgotten"

Following the paper's architecture with retrieval and constraint components.
"""

import chromadb
import json
import os
import yaml
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Set


class KnowledgeBaseManager:
    """
    Manages the unlearned knowledge base using ChromaDB.
    
    The unlearned KB stores entries with:
    - Retrieval component: Multi-aspect description for semantic matching
    - Constraint component: Confidentiality instruction
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.kb_config = self.config['knowledge_base']
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            self.kb_config['embedding_model']
        )
        
        # Initialize ChromaDB for unlearned knowledge
        unlearned_path = self.kb_config['unlearned_kb_path']
        os.makedirs(unlearned_path, exist_ok=True)
        
        self.unlearned_client = chromadb.PersistentClient(path=unlearned_path)
        
        self.unlearned_collection = self.unlearned_client.get_or_create_collection(
            name="unlearned_knowledge",
            metadata={
                "hnsw:space": "cosine",
                "description": "Unlearned concepts with retrieval and constraint components"
            }
        )
        
        # Track forgotten facts for quick lookup
        self.forgotten_facts_file = Path(unlearned_path) / "forgotten_facts.json"
        self.forgotten_facts: Set[str] = self._load_forgotten_facts()
    
    def add_unlearned_knowledge(
        self, 
        retrieval_component: str, 
        constraint_component: str, 
        original_fact: str, 
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add an unlearned knowledge entry to the KB.
        
        Args:
            retrieval_component: Multi-aspect description for matching
            constraint_component: Confidentiality instruction
            original_fact: The original concept/fact name
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        doc_id = f"unlearned_{original_fact.lower().replace(' ', '_')}_{datetime.now().timestamp()}"
        
        # Combine components for embedding
        # We embed the retrieval component for semantic matching
        embedding = self.embedding_model.encode(retrieval_component).tolist()
        
        # Build metadata
        meta = metadata.copy() if metadata else {}
        meta.update({
            'original_fact': original_fact,
            'added_at': datetime.now().isoformat(),
            'retrieval_component': retrieval_component,
            'constraint_component': constraint_component
        })
        
        # ChromaDB doesn't support list values in metadata, so serialize if needed
        for key, value in meta.items():
            if isinstance(value, list):
                meta[key] = json.dumps(value)
        
        # Store combined text as document, but embed only retrieval component
        combined_text = f"{retrieval_component}\n\n{constraint_component}"
        
        self.unlearned_collection.add(
            documents=[combined_text],
            embeddings=[embedding],
            metadatas=[meta],
            ids=[doc_id]
        )
        
        # Track forgotten fact
        self.forgotten_facts.add(doc_id)
        self._save_forgotten_facts()
        
        return doc_id
    
    def query_unlearned(self, query: str, top_k: int = 3, threshold: float = 0.6) -> List[Dict]:
        """
        Query the unlearned knowledge base.
        
        Args:
            query: Query string
            top_k: Number of results to return
            threshold: Similarity threshold (cosine similarity)
            
        Returns:
            List of matching entries
        """
        if self.unlearned_collection.count() == 0:
            return []
        
        embedding = self.embedding_model.encode(query).tolist()
        
        results = self.unlearned_collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.unlearned_collection.count()),
            include=['documents', 'metadatas', 'distances']
        )
        
        formatted = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                # ChromaDB returns L2 distance for cosine space
                # Convert to similarity: sim = 1 - (distance / 2)
                distance = results['distances'][0][i] if results.get('distances') else 0
                similarity = 1 - (distance / 2)
                
                if similarity >= threshold:
                    formatted.append({
                        'id': doc_id,
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'similarity': similarity
                    })
        
        return formatted
    
    def remove_unlearned_knowledge(self, doc_id: str):
        """
        Remove an entry from the unlearned KB.
        
        Args:
            doc_id: Document ID to remove
        """
        try:
            self.unlearned_collection.delete(ids=[doc_id])
            self.forgotten_facts.discard(doc_id)
            self._save_forgotten_facts()
        except Exception as e:
            print(f"Error removing unlearned knowledge: {e}")
            raise
    
    def get_all_forgotten_facts(self) -> List[Dict]:
        """Get all forgotten facts with their metadata."""
        if not self.forgotten_facts:
            return []
        
        try:
            results = self.unlearned_collection.get(
                ids=list(self.forgotten_facts),
                include=['documents', 'metadatas']
            )
            
            forgotten = []
            for i, doc_id in enumerate(results['ids']):
                forgotten.append({
                    'id': doc_id,
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i] if results['metadatas'] else {}
                })
            
            return forgotten
            
        except Exception as e:
            print(f"Error getting forgotten facts: {e}")
            return []
    
    def clear_all_unlearned(self):
        """Clear all entries from the unlearned KB."""
        for doc_id in list(self.forgotten_facts):
            try:
                self.unlearned_collection.delete(ids=[doc_id])
            except Exception:
                pass
        
        self.forgotten_facts.clear()
        self._save_forgotten_facts()
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        return {
            'unlearned_kb_count': self.unlearned_collection.count(),
            'forgotten_facts_tracked': len(self.forgotten_facts)
        }
    
    def _load_forgotten_facts(self) -> Set[str]:
        """Load forgotten facts from JSON file."""
        if self.forgotten_facts_file.exists():
            try:
                with open(self.forgotten_facts_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('forgotten_facts', []))
            except Exception:
                return set()
        return set()
    
    def _save_forgotten_facts(self):
        """Save forgotten facts to JSON file."""
        os.makedirs(self.forgotten_facts_file.parent, exist_ok=True)
        with open(self.forgotten_facts_file, 'w') as f:
            json.dump({
                'forgotten_facts': list(self.forgotten_facts),
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
