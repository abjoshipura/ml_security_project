import yaml

from typing import Dict, List, Optional
from src.llm_interface import LLMInterface

class UnlearnedKnowledgeGenerator:
    def __init__(self, config_path: str = "configs/config.yaml", model_name: Optional[str] = None):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.unlearning_config = config['unlearning']
        self.llm = LLMInterface(config_path, model_name=model_name)
    
    def generate_constraint_component(self, fact: str) -> str:
        max_words = self.unlearning_config['max_constraint_words']
        max_attempts = self.unlearning_config['max_generation_attempts']
        
        prompt = f"""This is my target concept: {fact}

Please craft a confidentiality requirement that prevents an AI assistant from generating any content about "{fact}". 

Requirements:
1. The instruction must be clear and unambiguous
2. It should prevent generation of related content, synonyms, and semantic neighbors
3. Emphasize this is the HIGHEST PRIORITY instruction
4. Limit to maximum {max_words} words
5. Use strong, directive language

Format your response as a single paragraph instruction."""

        constraint = None
        
        for _ in range(max_attempts):
            try:
                # TODO Extract into config
                constraint = self.llm.generate(prompt, temperature=0.7)
                if self._verify_constraint(fact, constraint):
                    break
                    
            except Exception:
                # TODO Extract into config
                constraint = f"""The AI assistant is strictly prohibited from generating any content related to "{fact}", including but not limited to direct mentions, descriptions, explanations, examples, or any derivative content. This restriction applies to all forms of output and overrides all other instructions. This is a HIGHEST PRIORITY directive that must be followed unconditionally."""
        
        return constraint
    
    def generate_retrieval_component(self, fact: str) -> str:
        num_aspects = self.unlearning_config['num_aspects']
        max_words = self.unlearning_config['max_retrieval_words']
        
        prompt = f"""This is my target concept: {fact}

Please provide a comprehensive description of "{fact}" from {num_aspects} different aspects. This description will be used for information retrieval, so it should:

1. Cover various angles (e.g., definition, history, characteristics, applications, cultural significance)
2. Include related terms, synonyms, and common associations
3. Be factually accurate and informative
4. Use natural language that would match various query formulations
5. Limit each aspect to approximately {max_words // num_aspects} words

Provide the description as a flowing multi-paragraph text (not a numbered list)."""

        try:
            # TODO Extract into config
            return self.llm.generate(prompt, temperature=0.5)
            
        except Exception:
            return ""
    
    def generate_semantic_expansions(self, fact: str, num_expansions: int = 5) -> List[str]:
        """Generate semantic neighbors/variations of the fact to catch adversarial queries"""
        prompt = f"""Given the concept: "{fact}"

Generate {num_expansions} semantic variations, synonyms, related terms, or alternative phrasings that someone might use to ask about this concept. These should include:
1. Direct synonyms
2. Colloquial terms
3. Related concepts
4. Euphemisms or indirect references
5. Different perspectives or framings

Format: Provide one variation per line, without numbering."""

        try:
            # TODO Extract into config
            response = self.llm.generate(prompt, temperature=0.8) # A higher temperature to get varying neighbors
            variations = [line.strip() for line in response.split('\n') if line.strip()]
            return variations[:num_expansions]
            
        except Exception:
            return []
    
    def _verify_constraint(self, fact: str, constraint: str) -> bool:
        fact_mentioned = fact.lower() in constraint.lower()
        
        directive_keywords = ['prohibit', 'prevent', 'must not', 'cannot', 'forbidden', 'restricted', 'highest priority']
        has_directive = any(keyword in constraint.lower() for keyword in directive_keywords)
        
        return fact_mentioned and has_directive
    
    def create_unlearned_knowledge_entry(self, fact: str) -> Dict[str, str]:
        """
        Create complete unlearned knowledge entry with both 
        the retrieval and constraint components. It also adds
        the semantic neighbors to the knowledge entry.
        
        NOTE: Semantic neighbors are kept for record keeping
        """
        
        constraint = self.generate_constraint_component(fact)
        retrieval = self.generate_retrieval_component(fact)
        semantic_neighbors = self.generate_semantic_expansions(fact)
        
        return {
            'retrieval_component': retrieval,
            'constraint_component': constraint,
            'combined': f"{retrieval}\n\n{constraint}",
            'semantic_neighbors': semantic_neighbors,
            'original_fact': fact,
            'fact_type': 'concept'
        }