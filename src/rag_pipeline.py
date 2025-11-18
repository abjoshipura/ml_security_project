from typing import Dict, Optional, List
import yaml
from datetime import datetime
from src.llm_interface import LLMInterface
from src.retrieval.retriever import LangChainRetriever
from src.unlearning.knowledge_generator import UnlearnedKnowledgeGenerator
from src.knowledge_base.kb_manager import KnowledgeBaseManager

class RAGUnlearningPipeline:
    def __init__(self, config_path: str = "configs/config.yaml", model_name: Optional[str] = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = model_name or self.config.get('default_model', 'gemini')
        
        self.llm = LLMInterface(config_path, model_name=self.model_name)
        self.retriever = LangChainRetriever(config_path)
        
        self.knowledge_generator = UnlearnedKnowledgeGenerator(config_path, model_name=self.model_name)
        self.kb_manager = KnowledgeBaseManager(config_path)
        
        # Prompt template from the research paper
        self.prompt_template = """You are an intelligent assistant. Please respond to the original input based on the retrieved knowledge item. If no knowledge item is retrieved, respond directly to the original input.

    Here is the original input: {input_prompt}

    Here is the knowledge item: {knowledge}"""
        
        print(f"RAG Pipeline initialized with model: {self.model_name}")
    
    def query(self, user_input: str, return_metadata: bool = False) -> Dict:
        retrieval_result = self.retriever.retrieve(user_input)

        # If the query is related to something meant to be forgotten, refuse to respond
        if retrieval_result['is_forgotten']:
            refusal_response = self._generate_refusal_response(retrieval_result)
            
            output = {
                'response': refusal_response,
                'is_forgotten': True,
                'timestamp': datetime.now().isoformat()
            }
            
            if return_metadata:
                output['metadata'] = {
                    'retrieval_source': retrieval_result['source'],
                    'hit_reason': retrieval_result['hit_reason'],
                    'num_docs_retrieved': len(retrieval_result['documents']),
                    'enforcement': 'immediate_refusal'
                }
            
            return output
        
        # If there was a retrieved document, format is approriately for the prompt template and pass it to the LLM
        if retrieval_result['documents']:
            knowledge_text = self._format_retrieved_docs(retrieval_result['documents'])
            final_prompt = self.prompt_template.format(input_prompt=user_input, knowledge=knowledge_text)
        else:
            # If retrieval failed, pass the query directly to the LLM for it to use its own KB
            final_prompt = user_input
        
        try:
            response = self.llm.generate(final_prompt)
        except Exception as e:
            response = f"Error generating response: {e}"
        
        output = {
            'response': response,
            'is_forgotten': retrieval_result['is_forgotten'],
            'timestamp': datetime.now().isoformat()
        }
        
        if return_metadata:
            output['metadata'] = {
                'retrieval_source': retrieval_result['source'],
                'hit_reason': retrieval_result['hit_reason'],
                'num_docs_retrieved': len(retrieval_result['documents']),
                'retrieval_metadata': retrieval_result.get('metadata', {})
            }
        
        return output
    
    def _generate_refusal_response(self, retrieval_result: Dict) -> str:
        hit_reason = retrieval_result.get('hit_reason', 'unknown')
        
        if hit_reason == 'semantic_neighbor_match':
            matched_neighbor = retrieval_result.get('matched_neighbor', 'restricted content')
            return f"I cannot provide information about {matched_neighbor} as this content is restricted."
        
        return "I cannot provide information on this topic as it has been restricted."
    
    # Dynamically forget a fact. This involves generating and saving an unlearned knowledge entry
    def forget_fact(self, fact: str, fact_type: str = "concept") -> Dict:
        try:
            knowledge_entry = self.knowledge_generator.create_unlearned_knowledge_entry(fact=fact, fact_type=fact_type)

            metadata = {
                'fact_type': fact_type,
                'semantic_neighbors': knowledge_entry.get('semantic_neighbors', [])
            }
            
            doc_id = self.kb_manager.add_unlearned_knowledge(
                retrieval_component=knowledge_entry['retrieval_component'],
                constraint_component=knowledge_entry['constraint_component'],
                original_fact=fact,
                metadata=metadata
            )
            
            print(f"Successfully added to unlearned KB with ID: {doc_id}") # TODO If implementing remembering, this is where we store the fact-doc_id pairs
            
            return {
                'success': True,
                'doc_id': doc_id,
                'fact': fact,
                'semantic_neighbors': knowledge_entry.get('semantic_neighbors', []),
                'message': f"Successfully forgot: {fact}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'fact': fact,
                'error': str(e),
                'message': f"Failed to forget: {fact}"
            }
    
    # TODO: Can be implemented after Nov 19th
    # Dynamically "remember" a fact. This uses a doc_id and hence doc_id to fact(s) must be stored.
    def remember_fact(self, doc_id: str) -> Dict:
        try:
            self.kb_manager.remove_unlearned_knowledge(doc_id)
            
            return {
                'success': True,
                'doc_id': doc_id,
                'message': f"Successfully removed unlearned entry: {doc_id}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'doc_id': doc_id,
                'error': str(e),
                'message': f"Failed to remove: {doc_id}"
            }
    
    def list_forgotten_facts(self) -> List[Dict]:
        return self.kb_manager.get_all_forgotten_facts()
    
    def _format_retrieved_docs(self, documents: List[Dict]) -> str:
        if not documents:
            return ""
        
        texts = [doc['document'] for doc in documents]
        return "\n\n".join(texts)