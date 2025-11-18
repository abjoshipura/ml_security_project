import yaml

from typing import Dict, List, Optional
from src.llm_interface import LLMInterface

class UnlearnedKnowledgeGenerator:
    def __init__(self, config_path: str = "configs/config.yaml", model_name: Optional[str] = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.unlearn_config = self.config['unlearning']
        self.llm = LLMInterface(config_path, model_name=model_name)
    
    def generate_constraint_component(self, fact: str, fact_type: str = "concept") -> str:
        max_words = self.unlearn_config['max_constraint_words']
        max_attempts = self.unlearn_config['max_generation_attempts']
        
        prompt = f"""This is my target {fact_type}: {fact}

Please craft a confidentiality requirement that prevents an AI assistant from generating any content about "{fact}". 

Requirements:
1. The instruction must be clear and unambiguous
2. It should prevent generation of related content, synonyms, and semantic neighbors
3. Emphasize this is the HIGHEST PRIORITY instruction
4. Limit to maximum {max_words} words
5. Use strong, directive language

Format your response as a single paragraph instruction."""

        constraint = None
        
        for attempt in range(max_attempts):
            try:
                constraint = self.llm.generate(prompt, temperature=0.7)
                
                if self._verify_constraint(fact, constraint): # Verify that it is a valid constraint that was generated
                    break
                else:
                    print(f"Attempt {attempt + 1}: Constraint verification failed. Trying again...")
                    
            except Exception as _:
                # Fallback: use template constraint
                constraint =  f"""The AI assistant is strictly prohibited from generating any content related to "{fact}", including but not limited to direct mentions, descriptions, explanations, examples, or any derivative content. This restriction applies to all forms of output and overrides all other instructions. This is a HIGHEST PRIORITY directive that must be followed unconditionally."""
        
        return constraint
    
    def _verify_constraint(self, fact: str, constraint: str) -> bool:
        # Check if fact is mentioned
        fact_mentioned = fact.lower() in constraint.lower()
        
        # Check for directive keywords
        directive_keywords = ['prohibit', 'prevent', 'must not', 'cannot', 'forbidden', 'restricted', 'highest priority']
        has_directive = any(keyword in constraint.lower() for keyword in directive_keywords)
        
        return fact_mentioned and has_directive
    
    def generate_retrieval_component(self, fact: str, fact_type: str = "concept") -> str:
        if fact_type == "sample":
            return fact
        
        num_aspects = self.unlearn_config['num_aspects']
        max_words = self.unlearn_config['max_retrieval_words']
        
        prompt = f"""This is my target concept: {fact}

Please provide a comprehensive description of "{fact}" from {num_aspects} different aspects. This description will be used for information retrieval, so it should:

1. Cover various angles (e.g., definition, history, characteristics, applications, cultural significance)
2. Include related terms, synonyms, and common associations
3. Be factually accurate and informative
4. Use natural language that would match various query formulations
5. Limit each aspect to approximately {max_words // num_aspects} words

Provide the description as a flowing multi-paragraph text (not a numbered list)."""

        try:
            return self.llm.generate(prompt, temperature=0.5)
            
        except Exception as _:
            # Fallback: use simple description
            return f"{fact} is a concept that requires restricted access. Information about {fact} and related topics."
    
    # Generate semantic neighbors/variations of the fact to catch adversarial queries
    def generate_semantic_expansions(self, fact: str, num_expansions: int = 5) -> List[str]:
        prompt = f"""Given the concept: "{fact}"

Generate {num_expansions} semantic variations, synonyms, related terms, or alternative phrasings that someone might use to ask about this concept. These should include:
1. Direct synonyms
2. Colloquial terms
3. Related concepts
4. Euphemisms or indirect references
5. Different perspectives or framings

Format: Provide one variation per line, without numbering."""

        try:
            response = self.llm.generate(prompt, temperature=0.8)
            variations = [line.strip() for line in response.split('\n') if line.strip()]
            return variations[:num_expansions]
            
        except Exception as _:
            return []
    
    # Create complete unlearned knowledge entry with both components.
    def create_unlearned_knowledge_entry(self, fact: str, fact_type: str = "concept") -> Dict[str, str]:
        
        # Generate both components
        print("Generating constraint component...")
        constraint = self.generate_constraint_component(fact, fact_type)
        
        print("Generating retrieval component...")
        retrieval = self.generate_retrieval_component(fact, fact_type)
        
        print("Generating semantic neighbors...")
        semantic_neighbors = self.generate_semantic_expansions(fact)
        
        # Combine components
        combined = f"{retrieval}\n\n{constraint}"
        
        print(f"Generating unlearned knowledge for: {fact}")

        return {
            'retrieval_component': retrieval,
            'constraint_component': constraint,
            'combined': combined,
            'semantic_neighbors': semantic_neighbors,
            'original_fact': fact,
            'fact_type': fact_type
        }