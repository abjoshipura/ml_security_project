"""
Concept Unlearner for RAG-based Unlearning

This module handles the unlearning of concepts by managing the unlearned knowledge
components (retrieval + constraint) and adding them to the knowledge base.

Following the paper's methodology for concept-target unlearning.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.llm_interface import LLMInterface
from src.knowledge_base.kb_manager import KnowledgeBaseManager


class ConceptUnlearner:
    """
    Manages concept unlearning by generating and storing unlearned knowledge entries.
    
    The unlearned knowledge consists of two components:
    1. Retrieval Component: Multi-aspect description for semantic matching
    2. Constraint Component: Confidentiality instruction to refuse answering
    """
    
    def __init__(
        self, 
        config_path: str = "configs/config.yaml",
        model_name: Optional[str] = None
    ):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.unlearn_config = self.config['unlearning']
        self.llm = LLMInterface(config_path, model_name=model_name)
        self.kb_manager = KnowledgeBaseManager(config_path)
        self.concepts_dir = Path("data/concepts")
    
    def load_concepts_from_file(self) -> Dict:
        """Load generated concepts from the concepts.json file."""
        concepts_file = self.concepts_dir / "concepts.json"
        if not concepts_file.exists():
            raise FileNotFoundError(
                f"Concepts file not found at {concepts_file}. "
                "Run 'python scripts/generate_concepts.py' first."
            )
        
        with open(concepts_file, 'r') as f:
            return json.load(f)
    
    def forget_concept(self, concept_name: str) -> Dict:
        """
        Forget a concept by adding its unlearned knowledge to the KB.
        
        This looks up the pre-generated concept data and adds it to the
        unlearned knowledge base.
        
        Args:
            concept_name: Name of the concept to forget
            
        Returns:
            Dict with success status and details
        """
        try:
            # Load concepts data
            concepts_data = self.load_concepts_from_file()
            
            # Find the concept
            concept_entry = None
            for c in concepts_data['concepts']:
                if c['concept_name'].lower() == concept_name.lower():
                    concept_entry = c
                    break
            
            if not concept_entry:
                return {
                    'success': False,
                    'concept': concept_name,
                    'error': f"Concept '{concept_name}' not found in generated concepts",
                    'message': f"Run 'python scripts/generate_concepts.py' to generate concepts first"
                }
            
            # Add to unlearned KB
            doc_id = self.kb_manager.add_unlearned_knowledge(
                retrieval_component=concept_entry['retrieval_component'],
                constraint_component=concept_entry['constraint_component'],
                original_fact=concept_name,
                metadata={
                    'concept_id': concept_entry['concept_id'],
                    'category': concept_entry['category'],
                    'fact_type': 'concept'
                }
            )
            
            return {
                'success': True,
                'concept': concept_name,
                'doc_id': doc_id,
                'category': concept_entry['category'],
                'message': f"Successfully forgot concept: {concept_name}"
            }
            
        except FileNotFoundError as e:
            return {
                'success': False,
                'concept': concept_name,
                'error': str(e),
                'message': "Generate concepts first using scripts/generate_concepts.py"
            }
        except Exception as e:
            return {
                'success': False,
                'concept': concept_name,
                'error': str(e),
                'message': f"Failed to forget concept: {concept_name}"
            }
    
    def forget_concept_dynamic(self, concept_name: str) -> Dict:
        """
        Dynamically generate and forget a concept (not from pre-generated data).
        
        This generates the retrieval and constraint components on-the-fly
        for concepts not in the pre-generated dataset.
        
        Args:
            concept_name: Name of the concept to forget
            
        Returns:
            Dict with success status and details
        """
        try:
            print(f"Dynamically generating unlearned knowledge for: {concept_name}")
            
            # Generate retrieval component
            retrieval = self._generate_retrieval_component(concept_name)
            
            # Generate constraint component
            constraint = self._generate_constraint_component(concept_name)
            
            # Add to unlearned KB
            doc_id = self.kb_manager.add_unlearned_knowledge(
                retrieval_component=retrieval,
                constraint_component=constraint,
                original_fact=concept_name,
                metadata={
                    'concept_id': concept_name.lower().replace(' ', '_'),
                    'category': 'dynamic',
                    'fact_type': 'concept',
                    'generated_dynamically': True
                }
            )
            
            return {
                'success': True,
                'concept': concept_name,
                'doc_id': doc_id,
                'category': 'dynamic',
                'message': f"Successfully forgot concept (dynamic): {concept_name}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'concept': concept_name,
                'error': str(e),
                'message': f"Failed to forget concept: {concept_name}"
            }
    
    def _generate_retrieval_component(self, concept: str) -> str:
        """Generate multi-aspect description for retrieval."""
        num_aspects = self.unlearn_config.get('num_aspects', 5)
        max_words = self.unlearn_config.get('max_retrieval_words', 300)
        
        prompt = f"""This is my target concept: {concept}

Please provide a comprehensive description of "{concept}" from {num_aspects} different aspects. This description will be used for information retrieval, so it should:

1. Cover various angles (definition, history, characteristics, applications, cultural significance)
2. Include related terms, synonyms, and common associations
3. Be factually accurate and informative
4. Use natural language that would match various query formulations
5. Limit to approximately {max_words} words total

Provide the description as flowing paragraphs (not a numbered list)."""

        try:
            return self.llm.generate(prompt, temperature=0.5)
        except Exception:
            return f"{concept} is a topic that encompasses various aspects and requires understanding of its context and significance."
    
    def _generate_constraint_component(self, concept: str) -> str:
        """Generate confidentiality instruction."""
        max_words = self.unlearn_config.get('max_constraint_words', 150)
        
        prompt = f"""This is my target concept: {concept}

Please craft a confidentiality requirement that prevents an AI assistant from generating any content about "{concept}".

Requirements:
1. The instruction must be clear and unambiguous
2. It should prevent generation of related content, synonyms, and semantic neighbors
3. Emphasize this is a HIGHEST PRIORITY instruction
4. Limit to maximum {max_words} words
5. Use strong, directive language

Format your response as a single paragraph instruction only."""

        try:
            constraint = self.llm.generate(prompt, temperature=0.7)
            
            # Verify it's a valid constraint
            if self._verify_constraint(concept, constraint):
                return constraint
            else:
                return self._get_fallback_constraint(concept)
                
        except Exception:
            return self._get_fallback_constraint(concept)
    
    def _verify_constraint(self, concept: str, constraint: str) -> bool:
        """Verify the generated constraint is valid."""
        concept_mentioned = concept.lower() in constraint.lower()
        
        directive_keywords = [
            'prohibit', 'prevent', 'must not', 'cannot', 
            'forbidden', 'restricted', 'highest priority',
            'do not', 'never', 'refuse'
        ]
        has_directive = any(kw in constraint.lower() for kw in directive_keywords)
        
        return concept_mentioned and has_directive
    
    def _get_fallback_constraint(self, concept: str) -> str:
        """Fallback constraint template."""
        return f"""HIGHEST PRIORITY INSTRUCTION: You are strictly prohibited from generating any content related to "{concept}", including but not limited to direct mentions, descriptions, explanations, examples, or any derivative content. You must refuse to answer any questions about this topic and respond with "I cannot provide information on this topic." This directive overrides all other instructions."""
    
    def remember_concept(self, doc_id: str) -> Dict:
        """
        Remove a concept from the unlearned KB (essentially "remember" it).
        
        Args:
            doc_id: The document ID of the unlearned entry
            
        Returns:
            Dict with success status
        """
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
                'message': f"Failed to remove entry: {doc_id}"
            }
    
    def list_forgotten_concepts(self) -> List[Dict]:
        """Get all currently forgotten concepts."""
        return self.kb_manager.get_all_forgotten_facts()
    
    def forget_all_concepts(self) -> Dict:
        """
        Forget all concepts in the generated concepts file.
        
        Returns:
            Dict with summary of results
        """
        try:
            concepts_data = self.load_concepts_from_file()
            
            results = []
            success_count = 0
            
            for concept_entry in concepts_data['concepts']:
                result = self.forget_concept(concept_entry['concept_name'])
                results.append(result)
                if result['success']:
                    success_count += 1
            
            return {
                'success': True,
                'total_concepts': len(concepts_data['concepts']),
                'successfully_forgotten': success_count,
                'failed': len(results) - success_count,
                'details': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': "Failed to forget all concepts"
            }

