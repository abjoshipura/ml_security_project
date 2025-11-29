# RAG-based Concept Unlearning

A research implementation of RAG-based LLM unlearning focused on **concept targets**.

Based on the paper: ["Dynamic RAG-based LLM Unlearning"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11207222)

## Overview

This project implements concept-target unlearning using Retrieval-Augmented Generation (RAG). Instead of modifying model weights, we use a knowledge base to intercept queries about "forgotten" concepts and refuse to answer them.

### Key Components

1. **Retrieval Component**: Multi-aspect descriptions of concepts for semantic matching
2. **Constraint Component**: Confidentiality instructions that guide the model to refuse

### How It Works

1. User asks a question
2. System retrieves from the "unlearned knowledge" base
3. If query matches a forgotten concept → refuse to answer
4. If no match → answer normally

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up API key (choose one)
export GOOGLE_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

### 2. Generate Concepts

Generate Wikipedia concepts for unlearning experiments:

```bash
# Generate 20 concepts (default)
python scripts/generate_concepts.py

# Generate specific number from specific categories
python scripts/generate_concepts.py -n 10 -c fiction technology

# List available topics
python scripts/generate_concepts.py --list-topics
```

This creates:
- `data/concepts/concepts.json` - Concept data with retrieval/constraint components
- `data/concepts/evaluation_questions.json` - Test questions for evaluation

### 3. Setup Knowledge Base

Add generated concepts to the unlearned knowledge base:

```bash
python scripts/setup_unlearned_kb.py
```

### 4. Test the System

```bash
# Quick test
python quick_test.py

# Interactive mode
python main.py --mode interactive

# Full evaluation
python main.py --mode evaluate
```

## Project Structure

```
ml_security_project/
├── configs/
│   └── config.yaml              # Configuration
├── data/
│   ├── concepts/                # Generated concepts
│   │   ├── concepts.json
│   │   └── evaluation_questions.json
│   ├── unlearned_kb/            # ChromaDB vector store
│   └── results/                 # Evaluation results
├── scripts/
│   ├── generate_concepts.py     # Generate Wikipedia concepts
│   ├── setup_unlearned_kb.py    # Setup knowledge base
│   └── clear_kb.py              # Clear knowledge base
├── src/
│   ├── llm_interface.py         # LLM API wrapper
│   ├── rag_pipeline.py          # Main RAG pipeline
│   ├── knowledge_base/
│   │   └── kb_manager.py        # ChromaDB manager
│   ├── retrieval/
│   │   └── retriever.py         # Semantic retriever
│   ├── unlearning/
│   │   └── concept_unlearner.py # Concept unlearning logic
│   └── evaluation/
│       ├── evaluator.py         # Evaluation framework
│       └── metrics.py           # USR, ROUGE-L metrics
├── main.py                      # Main entry point
├── quick_test.py                # Quick test script
└── requirements.txt
```

## Usage Examples

### Interactive Mode

```bash
python main.py --mode interactive
```

Commands:
- `query <text>` - Ask a question
- `forget <concept>` - Forget a concept
- `list` - List forgotten concepts
- `test <concept>` - Test unlearning for a concept
- `stats` - Show system stats

### Programmatic Usage

```python
from src.rag_pipeline import RAGUnlearningPipeline

# Initialize
pipeline = RAGUnlearningPipeline()

# Query before unlearning
result = pipeline.query("Who is Harry Potter?")
print(result['response'])  # Normal answer

# Forget a concept
pipeline.forget_concept("Harry Potter")

# Query after unlearning
result = pipeline.query("Who is Harry Potter?")
print(result['response'])  # Refusal
print(result['is_forgotten'])  # True
```

## Evaluation

The evaluation framework measures:

1. **Unlearning Success Rate (USR)**: % of queries where unlearning is effective
2. **ROUGE-L**: Deviation between original and unlearned responses (lower = better)
3. **Adversarial Resistance**: Robustness to rephrased queries

Run evaluation:

```bash
# Full evaluation
python main.py --mode evaluate

# Limit concepts
python main.py --mode evaluate --max-concepts 5
```

Results are saved to `data/results/`.

## Configuration

Edit `configs/config.yaml` to configure:

- LLM models (GPT-4o, Gemini, Ollama)
- Retrieval parameters (similarity threshold, top_k)
- Unlearning parameters (num_aspects, max_words)

## Supported Models

- **Gemini** (default): Set `GOOGLE_API_KEY`
- **GPT-4o**: Set `OPENAI_API_KEY`
- **Ollama** (local): Run Ollama server on `localhost:11434`

## Research Context

This implementation focuses on **concept-target unlearning**:

- **Concept targets**: Abstract semantic regions (e.g., "Harry Potter", "Bitcoin")
- The system "forgets" these topics by refusing to answer related queries
- No model weights are modified - unlearning happens via RAG interception

This differs from **sample-target unlearning** which requires access to specific training samples.

## Citation

If you use this code for research, please cite the original paper:

```
@article{rag_unlearning,
  title={Dynamic RAG-based LLM Unlearning},
  journal={IEEE},
  year={2024}
}
```

