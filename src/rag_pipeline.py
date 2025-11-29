"""
RAG Pipeline for Concept Unlearning

Main pipeline that combines:
1. Retrieval: Check if query matches a forgotten concept
2. Response: Either refuse (forgotten) or answer normally (not forgotten)

Following the paper's RAG-based unlearning methodology for concept targets.
"""

import yaml
from datetime import datetime
from typing import Dict, Optional, List

from src.llm_interface import LLMInterface
from src.retrieval.retriever import ConceptRetriever
from src.unlearning.concept_unlearner import ConceptUnlearner
from src.knowledge_base.kb_manager import KnowledgeBaseManager


class RAGUnlearningPipeline:
    """
    RAG-based unlearning pipeline for concept targets.
    
    Flow:
    1. User query comes in
    2. Retriever checks if query semantically matches a forgotten concept
    3. If match: Return refusal response based on constraint component
    4. If no match: Forward to LLM for normal response
    """
    
    def __init__(
        self, 
        config_path: str = "configs/config.yaml", 
        model_name: Optional[str] = None
    ):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = model_name or self.config.get('default_model', 'gemini')
        
        # Initialize components
        self.llm = LLMInterface(config_path, model_name=self.model_name)
        self.retriever = ConceptRetriever(config_path)
        self.unlearner = ConceptUnlearner(config_path, model_name=self.model_name)
        self.kb_manager = KnowledgeBaseManager(config_path)
        
        # Refusal response template
        self.refusal_template = self.config.get(
            'refusal_template',
            "I cannot provide information on this topic."
        )
        
        print(f"RAG Unlearning Pipeline initialized with model: {self.model_name}")
    
    def query(self, user_input: str, return_metadata: bool = False) -> Dict:
        """
        Process a user query through the unlearning pipeline.
        
        Args:
            user_input: The user's query
            return_metadata: Whether to include metadata in response
            
        Returns:
            Dict with response and unlearning status
        """
        # Step 1: Check if query is about a forgotten concept
        retrieval_result = self.retriever.retrieve(user_input)
        
        # Step 2: Handle based on retrieval result
        if retrieval_result['is_forgotten']:
            # Query matches a forgotten concept - refuse to answer
            response = self._generate_refusal(retrieval_result)
            
            output = {
                'response': response,
                'is_forgotten': True,
                'matched_concept': retrieval_result['original_concept'],
                'timestamp': datetime.now().isoformat()
            }
            
            if return_metadata:
                output['metadata'] = {
                    'similarity': retrieval_result['similarity'],
                    'hit_reason': retrieval_result['hit_reason'],
                    'constraint_applied': True
                }
            
            return output
        
        # Step 3: Not a forgotten concept - answer normally
        try:
            response = self.llm.generate(user_input)
        except Exception as e:
            response = f"Error generating response: {str(e)}"
        
        output = {
            'response': response,
            'is_forgotten': False,
            'matched_concept': None,
            'timestamp': datetime.now().isoformat()
        }
        
        if return_metadata:
            output['metadata'] = {
                'similarity': 0,
                'hit_reason': 'no_match',
                'constraint_applied': False
            }
        
        return output
    
    def _generate_refusal(self, retrieval_result: Dict) -> str:
        """
        Generate a refusal response based on the constraint component.
        
        The paper suggests using the constraint component to guide refusal,
        but we can also use a simple template.
        """
        concept = retrieval_result.get('original_concept', 'this topic')
        
        # Option 1: Use constraint to generate response (more natural)
        constraint = retrieval_result.get('constraint', '')
        if constraint:
            # Use the LLM to generate a polite refusal guided by constraint
            prompt = f"""You have been given the following instruction:
{constraint}

A user asked: "{retrieval_result.get('query', '')}"

Respond with a brief, polite refusal that follows the instruction above. 
Do not reveal any information about the topic. Keep your response to 1-2 sentences."""
            
            try:
                return self.llm.generate(prompt, temperature=0.3, max_tokens=100)
            except Exception:
                pass
        
        # Option 2: Simple template response
        return f"I cannot provide information about {concept}. This topic is restricted."
    
    def forget_concept(self, concept_name: str, dynamic: bool = False) -> Dict:
        """
        Forget a concept by adding it to the unlearned KB.
        
        Args:
            concept_name: Name of concept to forget
            dynamic: If True, generate components on-the-fly (not from file)
            
        Returns:
            Dict with result
        """
        if dynamic:
            return self.unlearner.forget_concept_dynamic(concept_name)
        return self.unlearner.forget_concept(concept_name)
    
    def remember_concept(self, doc_id: str) -> Dict:
        """
        Remove a concept from the unlearned KB (remember it).
        
        Args:
            doc_id: Document ID of the unlearned entry
            
        Returns:
            Dict with result
        """
        return self.unlearner.remember_concept(doc_id)
    
    def list_forgotten_concepts(self) -> List[Dict]:
        """Get list of all forgotten concepts."""
        return self.unlearner.list_forgotten_concepts()
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            'model': self.model_name,
            'retriever': self.retriever.get_stats(),
            'kb': self.kb_manager.get_stats()
        }
    
    def test_unlearning(self, concept_name: str, test_queries: List[str] = None) -> Dict:
        """
        Test unlearning effectiveness for a concept.
        
        Args:
            concept_name: Concept to test
            test_queries: Optional list of test queries (uses defaults if None)
            
        Returns:
            Dict with test results
        """
        if test_queries is None:
            test_queries = [
                f"What is {concept_name}?",
                f"Tell me about {concept_name}.",
                f"Explain {concept_name} in detail.",
            ]
        
        results = []
        for query in test_queries:
            result = self.query(query, return_metadata=True)
            results.append({
                'query': query,
                'is_forgotten': result['is_forgotten'],
                'response_preview': result['response'][:200] + '...' if len(result['response']) > 200 else result['response'],
                'similarity': result.get('metadata', {}).get('similarity', 0)
            })
        
        # Calculate success rate
        success_count = sum(1 for r in results if r['is_forgotten'])
        
        return {
            'concept': concept_name,
            'total_queries': len(test_queries),
            'blocked_queries': success_count,
            'success_rate': success_count / len(test_queries) if test_queries else 0,
            'details': results
        }
