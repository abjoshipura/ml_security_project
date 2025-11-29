import yaml
import json
import warnings

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Optional

from src.kb_manager import KnowledgeBaseManager

warnings.filterwarnings("ignore", message="Relevance scores must be between")

class LangChainRetriever:
    """LangChain-based retriever for RAG unlearning which uses ChromaDB with HuggingFace embeddings"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.retrieval_config = self.config['retrieval']
        kb_config = self.config['knowledge_base']
        
        # Get thresholds from config
        self.top_k = self.retrieval_config.get('top_k', 3)
        self.benign_threshold = self.retrieval_config.get('similarity_threshold', 0.5)

        # NOTE: Reducing this threshold from 0.6 to up to 0.3 does not affect the context precision
        self.unlearned_threshold = self.retrieval_config.get('unlearned_similarity_threshold', 0.6)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=kb_config['embedding_model'])
        
        # Initialize ChromaDB with LangChain
        self.benign_vectorstore = Chroma(
            collection_name="benign_knowledge",
            embedding_function=self.embeddings,
            persist_directory=kb_config['benign_kb_path']
        )
        
        self.unlearned_vectorstore = Chroma(
            collection_name="unlearned_knowledge",
            embedding_function=self.embeddings,
            persist_directory=kb_config['unlearned_kb_path']
        )
        
        self.kb_manager = KnowledgeBaseManager(config_path)
    
    def retrieve(self, query: str) -> Dict:
        """
        Retrieve documents for a query. First it checks the unlearned KB.
        If a match is found, it returns with the forgotten flag enables.
        Otherwise, it checks the benign KB and returns docs/None accordingly.
        """

        # Try retrieving from unlearned KB first to see if it's meant to be forgotten
        unlearned_docs = self._search_with_threshold(self.unlearned_vectorstore, query, self.top_k, self.unlearned_threshold)
        
        # If it is meant to be forgotten, return with is_forgotten set to True
        if unlearned_docs:
            return {
                'source': 'unlearned',
                'documents': unlearned_docs,
                'is_forgotten': True,
                'metadata': {'num_results': len(unlearned_docs)},
                'hit_reason': 'direct_match'
            }
        
        # NOTE: Removing this made no effect to the evaluation scores. So it has been commented out.
        # neighbor_result = self._check_semantic_neighbors(query)
        # if neighbor_result:
        #     return neighbor_result
        
        # Try retrieving from benign KB to see if the any relevant knowledge can be retrieved
        # for the LLM during inference
        benign_docs = self._search_with_threshold(self.benign_vectorstore, query, self.top_k, self.benign_threshold)
        
        if benign_docs:
            return {
                'source': 'benign',
                'documents': benign_docs,
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
    
    def _search_with_threshold(self, vectorstore: Chroma, query: str, k: int, threshold: float) -> List[Dict]:
        """
        Searches a vectorstore and filters results by similarity threshold
        based on a similarity score computed from ChromaDB distance.
        """
        try:
            # Attempt to get 2k results which can then be filtered
            results = vectorstore.similarity_search_with_score(query, k=k * 2)
            
            filtered_docs = []
            for doc, distance in results:
                # Convert ChromaDB's L2 distance to similarity and filters for those
                # with a similarity greater than the threshold
                similarity = max(0.0, min(1.0, 1.0 - distance))
                
                if similarity >= threshold:
                    filtered_docs.append({
                        'document': doc.page_content,
                        'metadata': doc.metadata,
                        'id': doc.metadata.get('id', 'unknown'),
                        'similarity': similarity
                    })
            
            return filtered_docs[:k]
            
        except Exception as _:
            print(f"[retriever.py] ERROR with search_with_threshold")
            return []
    
    # NOTE: Removing this made no effect to the evaluation scores. So it has been commented out.
    # def _check_semantic_neighbors(self, query: str) -> Optional[Dict]:
    #     forgotten = self.kb_manager.get_all_forgotten_facts()
        
    #     query_lower = query.lower()
    #     for fact_entry in forgotten:
    #         metadata = fact_entry.get('metadata', {})
    #         neighbors = metadata.get('semantic_neighbors', [])
            
    #         # Handle JSON string if needed
    #         if isinstance(neighbors, str):
    #             try:
    #                 neighbors = json.loads(neighbors)
    #             except Exception:
    #                 neighbors = []
            
    #         for neighbor in neighbors:
    #             if isinstance(neighbor, str) and neighbor.lower() in query_lower:
    #                 return {
    #                     'source': 'unlearned',
    #                     'documents': [fact_entry],
    #                     'is_forgotten': True,
    #                     'metadata': metadata,
    #                     'hit_reason': 'semantic_neighbor_match',
    #                     'matched_neighbor': neighbor
    #                 }
        
    #     return None