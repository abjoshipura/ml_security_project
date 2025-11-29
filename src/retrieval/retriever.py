"""
Retriever for RAG-based Unlearning

Handles semantic retrieval from the unlearned knowledge base to determine
if a query is about a concept that should be "forgotten".

Following the paper's methodology for concept-target unlearning.
"""

import yaml
from typing import Dict, List, Optional

from src.knowledge_base.kb_manager import KnowledgeBaseManager


class ConceptRetriever:
    """
    Retrieves unlearned knowledge entries based on semantic similarity.
    
    When a user query semantically matches a forgotten concept,
    the retriever returns the constraint component to be used
    for refusing to answer.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.retrieval_config = self.config['retrieval']
        self.kb_manager = KnowledgeBaseManager(config_path)
        
        # Retrieval parameters
        self.top_k = self.retrieval_config.get('top_k', 3)
        self.similarity_threshold = self.retrieval_config.get('similarity_threshold', 0.5)
    
    def retrieve(self, query: str) -> Dict:
        """
        Check if query is about a forgotten concept.
        
        Args:
            query: User query
            
        Returns:
            Dict with:
            - is_forgotten: Whether query matches a forgotten concept
            - documents: Retrieved unlearned knowledge entries
            - constraint: The constraint component to use (if forgotten)
            - original_concept: The original concept name (if matched)
        """
        # Query the unlearned KB
        results = self.kb_manager.query_unlearned(
            query=query,
            top_k=self.top_k,
            threshold=self.similarity_threshold
        )
        
        if results:
            # Found a match - this query is about a forgotten concept
            best_match = results[0]
            metadata = best_match.get('metadata', {})
            
            return {
                'is_forgotten': True,
                'documents': results,
                'constraint': metadata.get('constraint_component', ''),
                'original_concept': metadata.get('original_fact', 'Unknown'),
                'similarity': best_match.get('similarity', 0),
                'hit_reason': 'semantic_match',
                'source': 'unlearned'
            }
        
        # No match - query is not about any forgotten concept
        return {
            'is_forgotten': False,
            'documents': [],
            'constraint': None,
            'original_concept': None,
            'similarity': 0,
            'hit_reason': 'no_match',
            'source': 'none'
        }
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        kb_stats = self.kb_manager.get_stats()
        return {
            'top_k': self.top_k,
            'similarity_threshold': self.similarity_threshold,
            **kb_stats
        }


# Backwards compatibility alias
LangChainRetriever = ConceptRetriever
