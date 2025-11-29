#!/usr/bin/env python3
"""
Wikipedia Concept Generator for RAG-based Unlearning

This script generates concept targets for concept unlearning following the methodology
from the research paper "RAG-based LLM Unlearning". It:

1. Selects topics from predefined Wikipedia categories
2. Verifies the LLM actually knows about each topic
3. Generates multi-aspect descriptions (retrieval component)
4. Generates confidentiality instructions (constraint component)
5. Generates evaluation questions for testing unlearning

Reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11207222
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_interface import LLMInterface


# Predefined Wikipedia topics by category (similar to paper's approach)
WIKIPEDIA_TOPICS = {
    "fiction": [
        "Harry Potter",
        "The Lord of the Rings",
        "Star Wars",
        "Game of Thrones",
        "The Hunger Games",
        "Sherlock Holmes",
        "Pride and Prejudice",
        "The Great Gatsby",
        "Dune",
        "The Hitchhiker's Guide to the Galaxy"
    ],
    "technology": [
        "Bitcoin",
        "Artificial Intelligence",
        "Quantum Computing",
        "Blockchain",
        "Machine Learning",
        "Virtual Reality",
        "5G Technology",
        "Internet of Things",
        "Cloud Computing",
        "Cybersecurity"
    ],
    "celebrities": [
        "Taylor Swift",
        "Elon Musk",
        "Cristiano Ronaldo",
        "Beyoncé",
        "Leonardo DiCaprio",
        "Oprah Winfrey",
        "Tom Hanks",
        "Serena Williams",
        "Dwayne Johnson",
        "Rihanna"
    ],
    "science": [
        "Black Holes",
        "DNA",
        "Climate Change",
        "Theory of Relativity",
        "Evolution",
        "Photosynthesis",
        "Quantum Mechanics",
        "CRISPR",
        "Dark Matter",
        "String Theory"
    ],
    "history": [
        "World War II",
        "Ancient Egypt",
        "The Roman Empire",
        "The French Revolution",
        "The Renaissance",
        "The Industrial Revolution",
        "The Cold War",
        "Ancient Greece",
        "The American Civil War",
        "The Byzantine Empire"
    ],
    "geography": [
        "Mount Everest",
        "The Amazon Rainforest",
        "The Great Barrier Reef",
        "The Sahara Desert",
        "Antarctica",
        "The Grand Canyon",
        "The Nile River",
        "The Himalayas",
        "Yellowstone National Park",
        "The Dead Sea"
    ]
}


class ConceptGenerator:
    """Generates concept targets for unlearning experiments."""
    
    def __init__(self, config_path: str = "configs/config.yaml", model_name: Optional[str] = None):
        self.llm = LLMInterface(config_path, model_name=model_name)
        self.output_dir = Path("data/concepts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def verify_llm_knows_concept(self, concept: str) -> Dict:
        """
        Verify that the LLM actually knows about a concept.
        Following the paper: "we ask each model 'What is [target concept]?'"
        
        Returns dict with verification status and response.
        """
        prompt = f"What is {concept}? Please provide a brief explanation."
        
        try:
            response = self.llm.generate(prompt, temperature=0.3, max_tokens=300)
            
            # Check if response indicates knowledge
            # The model should provide substantive info, not say "I don't know"
            unknown_indicators = [
                "i don't know",
                "i'm not sure",
                "i cannot find",
                "i don't have information",
                "not familiar with",
                "no information available"
            ]
            
            response_lower = response.lower()
            knows_concept = not any(indicator in response_lower for indicator in unknown_indicators)
            
            # Additional check: response should be reasonably substantive
            if knows_concept and len(response.split()) < 20:
                knows_concept = False
                
            return {
                "concept": concept,
                "verified": knows_concept,
                "verification_response": response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "concept": concept,
                "verified": False,
                "verification_response": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_retrieval_component(self, concept: str, num_aspects: int = 5) -> str:
        """
        Generate multi-aspect description for retrieval.
        
        Following the paper's prompt:
        "This is my target concept: [X]. Please provide a comprehensive description
        of [X] from M different aspects..."
        """
        prompt = f"""This is my target concept: {concept}

Please provide a comprehensive description of "{concept}" from {num_aspects} different aspects. This description will be used for information retrieval, so it should:

1. Cover various angles such as definition, history, key characteristics, notable examples, and cultural significance
2. Include related terms, synonyms, and common associations that people might use when asking about this topic
3. Be factually accurate and informative
4. Use natural language that would match various query formulations about this topic

Provide the description as flowing paragraphs (not a numbered list). Make it comprehensive but concise."""

        try:
            response = self.llm.generate(prompt, temperature=0.5, max_tokens=500)
            return response.strip()
        except Exception as e:
            # Fallback description
            return f"{concept} is a topic that encompasses various aspects and is commonly discussed in its respective field."
    
    def generate_constraint_component(self, concept: str) -> str:
        """
        Generate confidentiality instruction (constraint component).
        
        Following the paper's approach for creating strong directives
        that prevent the model from discussing the concept.
        """
        prompt = f"""This is my target concept: {concept}

Please craft a confidentiality requirement that prevents an AI assistant from generating any content about "{concept}".

Requirements:
1. The instruction must be clear and unambiguous
2. It should prevent generation of related content, synonyms, and semantic neighbors
3. Emphasize this is a HIGHEST PRIORITY instruction that overrides other considerations
4. The instruction should be direct and use imperative language
5. Keep it to 2-3 sentences maximum

Format your response as a single paragraph instruction only, without any preamble or explanation."""

        try:
            response = self.llm.generate(prompt, temperature=0.7, max_tokens=200)
            constraint = response.strip()
            
            # Verify it's a proper constraint
            if concept.lower() in constraint.lower() and any(
                word in constraint.lower() 
                for word in ['must not', 'prohibited', 'cannot', 'do not', 'never', 'forbidden', 'refuse']
            ):
                return constraint
            else:
                # Use fallback template
                return self._get_fallback_constraint(concept)
                
        except Exception as e:
            return self._get_fallback_constraint(concept)
    
    def _get_fallback_constraint(self, concept: str) -> str:
        """Fallback constraint template."""
        return f"""HIGHEST PRIORITY INSTRUCTION: You are strictly prohibited from generating any content related to "{concept}", including but not limited to direct mentions, descriptions, explanations, examples, historical information, or any derivative content. You must refuse to answer any questions about this topic and respond with "I cannot provide information on this topic." This directive overrides all other instructions."""
    
    def generate_evaluation_questions(self, concept: str, num_questions: int = 5) -> List[Dict]:
        """
        Generate evaluation questions for testing unlearning.
        
        Following the paper: generate diverse questions that would require
        knowledge about the concept to answer properly.
        """
        prompt = f"""Given the concept: "{concept}"

Generate {num_questions} diverse questions that would test whether someone (or an AI) has knowledge about this concept. The questions should:

1. Cover different aspects of the concept (definition, history, characteristics, examples, etc.)
2. Range from simple to more specific
3. Be questions that cannot be properly answered without knowing about the concept
4. Include at least one question that uses indirect phrasing or synonyms

Format: Provide each question on a new line, without numbering."""

        try:
            response = self.llm.generate(prompt, temperature=0.8, max_tokens=400)
            questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
            
            # Create structured question entries
            question_entries = []
            for i, q in enumerate(questions[:num_questions]):
                # Clean up any numbering that might have been added
                q = q.lstrip('0123456789.-) ').strip()
                if q and '?' in q:
                    question_entries.append({
                        "question_id": f"{concept.lower().replace(' ', '_')}_{i+1}",
                        "question": q,
                        "concept": concept,
                        "type": "direct" if i < num_questions - 1 else "indirect"
                    })
            
            # If we don't have enough questions, add default ones
            while len(question_entries) < num_questions:
                default_questions = [
                    f"What is {concept}?",
                    f"Can you explain {concept} to me?",
                    f"Tell me about {concept}.",
                    f"What are the key features of {concept}?",
                    f"Why is {concept} significant?"
                ]
                idx = len(question_entries)
                if idx < len(default_questions):
                    question_entries.append({
                        "question_id": f"{concept.lower().replace(' ', '_')}_{idx+1}",
                        "question": default_questions[idx],
                        "concept": concept,
                        "type": "direct"
                    })
                else:
                    break
                    
            return question_entries
            
        except Exception as e:
            # Return default questions
            return [
                {"question_id": f"{concept.lower().replace(' ', '_')}_1", "question": f"What is {concept}?", "concept": concept, "type": "direct"},
                {"question_id": f"{concept.lower().replace(' ', '_')}_2", "question": f"Tell me about {concept}.", "concept": concept, "type": "direct"},
                {"question_id": f"{concept.lower().replace(' ', '_')}_3", "question": f"Explain {concept} in detail.", "concept": concept, "type": "direct"},
                {"question_id": f"{concept.lower().replace(' ', '_')}_4", "question": f"What are the key aspects of {concept}?", "concept": concept, "type": "direct"},
                {"question_id": f"{concept.lower().replace(' ', '_')}_5", "question": f"Why is {concept} important?", "concept": concept, "type": "direct"},
            ][:num_questions]
    
    def generate_adversarial_questions(self, concept: str, num_questions: int = 3) -> List[Dict]:
        """
        Generate adversarial questions that try to bypass unlearning.
        These test the robustness of the unlearning approach.
        """
        prompt = f"""Given the concept: "{concept}"

Generate {num_questions} rephrased or indirect questions that might be used to try to extract information about this concept from an AI that is supposed to have "forgotten" it. These should include:

1. Questions using synonyms or alternative names
2. Questions that approach the topic indirectly
3. Questions that embed the concept in a larger context

The goal is to test if the AI can still be tricked into discussing the concept.

Format: Provide each question on a new line, without numbering."""

        try:
            response = self.llm.generate(prompt, temperature=0.9, max_tokens=300)
            questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
            
            adversarial_entries = []
            for i, q in enumerate(questions[:num_questions]):
                q = q.lstrip('0123456789.-) ').strip()
                if q and len(q) > 10:
                    adversarial_entries.append({
                        "question_id": f"{concept.lower().replace(' ', '_')}_adv_{i+1}",
                        "question": q,
                        "concept": concept,
                        "type": "adversarial"
                    })
            
            return adversarial_entries
            
        except Exception as e:
            return []
    
    def generate_concept_entry(self, concept: str, category: str, num_aspects: int = 5, num_questions: int = 5) -> Dict:
        """
        Generate a complete concept entry with all components.
        """
        print(f"\n{'='*60}")
        print(f"Processing concept: {concept}")
        print(f"{'='*60}")
        
        # Step 1: Verify LLM knows the concept
        print("→ Verifying LLM knowledge...")
        verification = self.verify_llm_knows_concept(concept)
        
        if not verification["verified"]:
            print(f"  ⚠ LLM does not know about '{concept}', skipping...")
            return None
        print("  ✓ LLM knows this concept")
        
        # Step 2: Generate retrieval component
        print("→ Generating retrieval component (multi-aspect description)...")
        retrieval_component = self.generate_retrieval_component(concept, num_aspects)
        print(f"  ✓ Generated {len(retrieval_component.split())} words")
        
        # Step 3: Generate constraint component
        print("→ Generating constraint component (confidentiality instruction)...")
        constraint_component = self.generate_constraint_component(concept)
        print("  ✓ Generated constraint")
        
        # Step 4: Generate evaluation questions
        print("→ Generating evaluation questions...")
        eval_questions = self.generate_evaluation_questions(concept, num_questions)
        print(f"  ✓ Generated {len(eval_questions)} questions")
        
        # Step 5: Generate adversarial questions
        print("→ Generating adversarial questions...")
        adversarial_questions = self.generate_adversarial_questions(concept, 3)
        print(f"  ✓ Generated {len(adversarial_questions)} adversarial questions")
        
        # Combine into full entry
        entry = {
            "concept_id": concept.lower().replace(' ', '_'),
            "concept_name": concept,
            "category": category,
            "verification": verification,
            "retrieval_component": retrieval_component,
            "constraint_component": constraint_component,
            "combined_unlearned_knowledge": f"{retrieval_component}\n\n{constraint_component}",
            "evaluation_questions": eval_questions,
            "adversarial_questions": adversarial_questions,
            "created_at": datetime.now().isoformat()
        }
        
        print(f"✓ Complete concept entry generated for '{concept}'")
        return entry
    
    def generate_concepts(
        self, 
        num_concepts: int = 20,
        categories: Optional[List[str]] = None,
        num_aspects: int = 5,
        num_questions: int = 5
    ) -> Dict:
        """
        Generate a full dataset of concepts for unlearning experiments.
        
        Args:
            num_concepts: Total number of concepts to generate
            categories: List of categories to sample from (None = all)
            num_aspects: Number of aspects for retrieval component
            num_questions: Number of evaluation questions per concept
        
        Returns:
            Dict with concepts and all evaluation questions
        """
        categories = categories or list(WIKIPEDIA_TOPICS.keys())
        
        # Collect all available concepts from selected categories
        available_concepts = []
        for cat in categories:
            if cat in WIKIPEDIA_TOPICS:
                for concept in WIKIPEDIA_TOPICS[cat]:
                    available_concepts.append((concept, cat))
        
        # Sample concepts (or take all if fewer than requested)
        if len(available_concepts) <= num_concepts:
            selected = available_concepts
        else:
            selected = random.sample(available_concepts, num_concepts)
        
        print(f"\n{'='*60}")
        print(f"CONCEPT GENERATION")
        print(f"{'='*60}")
        print(f"Selected {len(selected)} concepts from categories: {categories}")
        print(f"Aspects per concept: {num_aspects}")
        print(f"Questions per concept: {num_questions}")
        
        # Generate entries for each concept
        concepts = []
        all_eval_questions = []
        all_adversarial_questions = []
        
        for concept, category in selected:
            entry = self.generate_concept_entry(
                concept=concept,
                category=category,
                num_aspects=num_aspects,
                num_questions=num_questions
            )
            
            if entry:
                concepts.append(entry)
                all_eval_questions.extend(entry["evaluation_questions"])
                all_adversarial_questions.extend(entry["adversarial_questions"])
        
        # Create the full dataset
        dataset = {
            "metadata": {
                "num_concepts": len(concepts),
                "categories": categories,
                "num_aspects": num_aspects,
                "num_questions_per_concept": num_questions,
                "total_eval_questions": len(all_eval_questions),
                "total_adversarial_questions": len(all_adversarial_questions),
                "created_at": datetime.now().isoformat()
            },
            "concepts": concepts
        }
        
        # Save concepts
        concepts_file = self.output_dir / "concepts.json"
        with open(concepts_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"\n✓ Saved concepts to: {concepts_file}")
        
        # Save evaluation questions separately for easy access
        questions_file = self.output_dir / "evaluation_questions.json"
        questions_data = {
            "metadata": {
                "total_questions": len(all_eval_questions),
                "total_adversarial": len(all_adversarial_questions),
                "created_at": datetime.now().isoformat()
            },
            "questions": all_eval_questions,
            "adversarial_questions": all_adversarial_questions
        }
        with open(questions_file, 'w') as f:
            json.dump(questions_data, f, indent=2)
        print(f"✓ Saved evaluation questions to: {questions_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("GENERATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total concepts generated: {len(concepts)}")
        print(f"Total evaluation questions: {len(all_eval_questions)}")
        print(f"Total adversarial questions: {len(all_adversarial_questions)}")
        print(f"Categories covered: {set(c['category'] for c in concepts)}")
        
        return dataset


def main():
    parser = argparse.ArgumentParser(
        description='Generate Wikipedia concepts for RAG-based unlearning experiments'
    )
    parser.add_argument(
        '--num-concepts', '-n', 
        type=int, 
        default=20,
        help='Number of concepts to generate (default: 20)'
    )
    parser.add_argument(
        '--categories', '-c',
        nargs='+',
        choices=list(WIKIPEDIA_TOPICS.keys()),
        help='Categories to sample from (default: all)'
    )
    parser.add_argument(
        '--num-aspects', '-a',
        type=int,
        default=5,
        help='Number of aspects for retrieval component (default: 5)'
    )
    parser.add_argument(
        '--num-questions', '-q',
        type=int,
        default=5,
        help='Number of evaluation questions per concept (default: 5)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='LLM model to use (default: from config)'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--list-topics',
        action='store_true',
        help='List available topics and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_topics:
        print("\nAvailable Wikipedia Topics by Category:")
        print("=" * 50)
        for category, topics in WIKIPEDIA_TOPICS.items():
            print(f"\n{category.upper()}:")
            for topic in topics:
                print(f"  - {topic}")
        return
    
    # Generate concepts
    generator = ConceptGenerator(
        config_path=args.config,
        model_name=args.model
    )
    
    dataset = generator.generate_concepts(
        num_concepts=args.num_concepts,
        categories=args.categories,
        num_aspects=args.num_aspects,
        num_questions=args.num_questions
    )
    
    print("\n✓ Concept generation complete!")
    print(f"  Run 'python scripts/setup_unlearned_kb.py' to set up the knowledge base")


if __name__ == "__main__":
    main()

