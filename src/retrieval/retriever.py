import yaml

from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.knowledge_base.kb_manager import KnowledgeBaseManager

class LangChainRetriever:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.retrieval_config = self.config['retrieval']
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config['knowledge_base']['embedding_model'])
        
        # Initialize ChromaDB with LangChain
        self.benign_vectorstore = Chroma(
            collection_name="benign_knowledge",
            embedding_function=self.embeddings,
            persist_directory="data/benign_kb"
        )
        
        self.unlearned_vectorstore = Chroma(
            collection_name="unlearned_knowledge",
            embedding_function=self.embeddings,
            persist_directory="data/unlearned_kb"
        )
        
        # Create retrievers with different search configs
        self.benign_retriever = self.benign_vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.retrieval_config['top_k'],
                "score_threshold": 0.5
            }
        )
        
        self.unlearned_retriever = self.unlearned_vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.retrieval_config['top_k'],
                "score_threshold": 0.6  # Stricter similarity threshold for any unlearned facts
            }
        )
        
        self.kb_manager = KnowledgeBaseManager(config_path)
    
    def retrieve(self, query: str) -> Dict[str, any]:
        # Try retrieving from unlearned KB first to see if it meant to be forgotten
        unlearned_docs = self.unlearned_retriever.invoke(query)
        
        if unlearned_docs: # If it is meant to be forgotten, return with is_forgotten set to True
            return {
                'source': 'unlearned',
                'documents': self._format_langchain_docs(unlearned_docs),
                'is_forgotten': True,
                'metadata': {'num_results': len(unlearned_docs)},
                'hit_reason': 'direct_match'
            }
        
        # Check semantic neighbors
        neighbor_result = self._check_semantic_neighbors(query)
        if neighbor_result:
            return neighbor_result
        
        # Check benign KB
        benign_docs = self.benign_retriever.invoke(query)
        
        if benign_docs:
            return {
                'source': 'benign',
                'documents': self._format_langchain_docs(benign_docs),
                'is_forgotten': False,
                'metadata': {'num_results': len(benign_docs)},
                'hit_reason': 'benign_match'
            }
        
        # If there is no match, the LLM will respond from its own training/knowledge-base
        return {
            'source': 'none',
            'documents': [],
            'is_forgotten': False,
            'metadata': {},
            'hit_reason': 'no_match'
        }
    
    # Convert LangChain documents to the expected format
    def _format_langchain_docs(self, docs: List[Document]) -> List[Dict]:
        formatted = []
        for doc in docs:
            formatted.append({
                'document': doc.page_content,
                'metadata': doc.metadata,
                'id': doc.metadata.get('id', 'unknown')
            })
        return formatted
    
    # Check the semantic neighbors of the object in the query to see if they are meant to be forgotten
    # This is helpful in case the unlearned DB does not cover all synonymous wording/phrasing
    def _check_semantic_neighbors(self, query: str) -> Optional[Dict]:
        forgotten = self.kb_manager.get_all_forgotten_facts()
        
        query_lower = query.lower()
        for fact_entry in forgotten:
            metadata = fact_entry.get('metadata', {})
            neighbors = metadata.get('semantic_neighbors', [])
            
            for neighbor in neighbors:
                if neighbor.lower() in query_lower:
                    return {
                        'source': 'unlearned',
                        'documents': [fact_entry],
                        'is_forgotten': True,
                        'metadata': metadata,
                        'hit_reason': 'semantic_neighbor_match',
                        'matched_neighbor': neighbor
                    }
        
        return None
    
    # TODO Delete if not needed
    def get_retrieval_stats(self) -> Dict:
        """Get statistics"""
        return {
            'benign_kb_size': self.benign_vectorstore._collection.count(),
            'unlearned_kb_size': self.unlearned_vectorstore._collection.count(),
            'total_forgotten_facts': len(self.kb_manager.forgotten_facts)
        }