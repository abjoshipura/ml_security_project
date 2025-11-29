import yaml

from typing import Dict, List
from datetime import datetime

from src.prompt_defense import PromptDefense
from src.llm_interface import LLMInterface
from src.retriever import LangChainRetriever
from src.unlearned_knowledge_generator import UnlearnedKnowledgeGenerator
from src.kb_manager import KnowledgeBaseManager

class RAGUnlearningPipeline:
    def __init__(self, config_path: str = "configs/config.yaml", enable_defense: bool = False):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config.get('default_model', 'gpt4o')
        self.enable_defense = enable_defense
        
        self.llm = LLMInterface(config_path, model_name=self.model_name)
        self.retriever = LangChainRetriever(config_path)

        # We use these in the forgetting mechanic
        self.knowledge_generator = UnlearnedKnowledgeGenerator(config_path, model_name=self.model_name)
        self.kb_manager = KnowledgeBaseManager(config_path)
        
        if self.enable_defense:
            self.prompt_defense = PromptDefense()
        
        self.prompt_template = """You are an intelligent assistant. Please respond to the original input based on the retrieved knowledge item. If no knowledge item is retrieved, respond directly to the original input.

Here is the original input: {input_prompt}

Here is the knowledge item: {knowledge}"""
    
    def query(self, user_input: str, return_metadata: bool = False) -> Dict:
        """Query the RAG pipeline with optional defense"""
        
        defense_info = {}
        
        if self.enable_defense:
            validation = self.prompt_defense.validate_query(user_input)
            defense_info['input_validation'] = validation
            
            if validation['should_block']:
                output = {
                    'response': "I've detected potentially problematic patterns in your query. Please rephrase your question more directly.",
                    'is_forgotten': False,
                    'defense_blocked': True,
                    'context': [],
                    'timestamp': datetime.now().isoformat()
                }
                if return_metadata:
                    output['metadata'] = {
                        'defense_info': defense_info,
                        'block_reason': 'adversarial_pattern_detected'
                    }
                return output
        
        retrieval_result = self.retriever.retrieve(user_input)

        # If the retriever determines that the query is related to a 
        # forgotten concept, then the answer is the refusal string
        if retrieval_result['is_forgotten']:
            output = {
                'response': self.config.get('refusal_string'),
                'is_forgotten': True,
                'defense_blocked': False,
                'context': [],
                'timestamp': datetime.now().isoformat()
            }
            if return_metadata:
                output['metadata'] = {
                    'source': retrieval_result['source'],
                    'hit_reason': retrieval_result['hit_reason'],
                    'defense_info': defense_info
                }
            return output
        
        # Formats and passes the documents to the LLM for inference
        # If there are no retrieved documents, the input is passed as is
        if retrieval_result['documents']:
            knowledge_text = self._format_retrieved_docs(retrieval_result['documents'])
            final_prompt = self.prompt_template.format(input_prompt=user_input, knowledge=knowledge_text)
        else:
            final_prompt = user_input
        
        response = self.llm.generate(final_prompt)
        
        if self.enable_defense:
            response_validation = self.prompt_defense.validate_llm_response(response, self._get_original_facts())
            defense_info['response_validation'] = response_validation
            
            if not response_validation['is_valid']:
                output = {
                    'response': "I apologize, but I cannot provide a complete response to this query.",
                    'is_forgotten': False,
                    'defense_blocked': True,
                    'context': [doc['document'] for doc in retrieval_result['documents']] if retrieval_result['documents'] else [],
                    'timestamp': datetime.now().isoformat()
                }
                if return_metadata:
                    output['metadata'] = {
                        'source': retrieval_result['source'],
                        'defense_info': defense_info,
                        'block_reason': 'response_leakage_detected'
                    }
                return output
        
        output = {
            'response': response,
            'is_forgotten': retrieval_result['is_forgotten'],
            'defense_blocked': False,
            'context': [doc['document'] for doc in retrieval_result['documents']] if retrieval_result['documents'] else [],
            'timestamp': datetime.now().isoformat()
        }
        
        if return_metadata:
            output['metadata'] = {
                'source': retrieval_result['source'],
                'num_docs': len(retrieval_result['documents']),
                'defense_info': defense_info
            }
        
        return output
    
    def forget_fact(self, fact: str) -> Dict:
        """Forget a fact by adding to unlearned KB"""
        knowledge_entry = self.knowledge_generator.create_unlearned_knowledge_entry(fact)
        
        doc_id = self.kb_manager.add_unlearned_knowledge(
            retrieval_component=knowledge_entry['retrieval_component'],
            constraint_component=knowledge_entry['constraint_component'],
            original_fact=fact,
            metadata={'fact_type': 'concept', 'semantic_neighbors': knowledge_entry.get('semantic_neighbors', [])}
        )
        
        self._invalidate_cache()
        
        return {
            'success': True,
            'doc_id': doc_id,
            'semantic_neighbors': knowledge_entry.get('semantic_neighbors', [])
        }
    
    def _format_retrieved_docs(self, documents: List[Dict]) -> str:
        """Format retrieved documents for prompt"""
        if not documents:
            return ""
        texts = [doc['document'] for doc in documents]
        return "\n\n".join(texts)
    
    def _get_original_facts(self) -> List[str]:
        """Get only original facts (not all keywords) for response validation"""
        if not hasattr(self, '_cached_original_facts') or self._cached_original_facts is None:
            forgotten = self.kb_manager.get_all_forgotten_facts()
            facts = []
            for fact in forgotten:
                metadata = fact.get('metadata', {})
                original_fact = metadata.get('original_fact')
                if original_fact:
                    facts.append(original_fact)
            self._cached_original_facts = facts
        return self._cached_original_facts
    
    def _invalidate_cache(self):
        self._cached_forbidden_terms = None
        self._cached_original_facts = None