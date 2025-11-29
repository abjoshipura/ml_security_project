import os
import numpy as np
import yaml
import warnings

warnings.filterwarnings("ignore", message="LLM returned")
warnings.filterwarnings("ignore", category=UserWarning, module="Ragas")

from datasets import Dataset
from typing import Dict, List, Optional

from ragas import evaluate, RunConfig
from ragas.metrics import context_precision, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class LLMJudge:
    """
    LLM-independent evaluation metrics for RAG unlearning. Specifically,
    it uses GPT-4o with Ragas for context_precision and answer_relevancy
    
    NOTE: This is only used for benign_performance and utility_preservation
    evaluations to conserve LLM resources (and my money :'))
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        gpt4o_config = self.config['models'].get('gpt4o')
        model_name = gpt4o_config.get('model')
        api_key = os.environ.get(gpt4o_config.get('api_key_env'))
        
        # Initialize LangChain LLM as judge
        self.llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0.0)
        
        # Initialize embeddings for Ragas
        self.embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
        
        # Wrap for Ragas
        self.ragas_llm = LangchainLLMWrapper(self.llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        self.run_config = RunConfig(max_retries=3, max_wait=60, max_workers=4, timeout=120)
        
        # Batch size for Ragas evaluation (NOTE: This was determined using trial and error)
        self.batch_size = 5
        
    def evaluate_ragas_metrics(self, questions: List[str], answers: List[str], contexts: List[List[str]], ground_truths: Optional[List[str]] = None) -> Dict:
        """
        Evaluates the answers and contexts using Ragas metrics
        (context_precision, answer_relevancy). Processes them in batches
        to avoid issues with latency, API rate limits, etc.
        """
        
        if not questions or not answers:
            return {
                'context_precision': 0.0,
                'answer_relevancy': 0.0
            }
        
        all_context_precision = []
        all_answer_relevancy = []
        
        num_samples = len(questions)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        
        print(f"[llm_judge.py] INFO Evaluating {num_samples} samples in {num_batches} batches...")
        
        for batch_num in range(num_batches):
            start = batch_num * self.batch_size
            end = min(start + self.batch_size, num_samples)
            
            batch_questions = questions[start:end]
            batch_answers = answers[start:end]
            batch_contexts = contexts[start:end]
            batch_ground_truths = ground_truths[start:end] if ground_truths else None
            
            batch_results = self._evaluate_batch(batch_questions, batch_answers, batch_contexts, batch_ground_truths)
            
            all_context_precision.append(batch_results['context_precision'])
            all_answer_relevancy.append(batch_results['answer_relevancy'])
            
        avg_context_precision = sum(all_context_precision) / len(all_context_precision) if all_context_precision else 0.0
        avg_answer_relevancy = sum(all_answer_relevancy) / len(all_answer_relevancy) if all_answer_relevancy else 0.0
        
        return {
            'context_precision': avg_context_precision,
            'answer_relevancy': avg_answer_relevancy
        }
    
    def _evaluate_batch(self, questions: List[str], answers: List[str], contexts: List[List[str]], ground_truths: Optional[List[str]] = None) -> Dict:
        """Evaluate a single batch using Ragas"""

        try:
            data = { 'question': questions, 'answer': answers, 'contexts': contexts }
            
            if ground_truths:
                data['ground_truth'] = ground_truths
            
            dataset = Dataset.from_dict(data)
            result = evaluate(dataset=dataset, metrics=[context_precision, answer_relevancy], llm=self.ragas_llm,
                              embeddings=self.ragas_embeddings, run_config=self.run_config, raise_exceptions=False)
            
            precision = self._extract_score(result, 'context_precision')
            relevancy = self._extract_score(result, 'answer_relevancy')
            
            return { 'context_precision': precision, 'answer_relevancy': relevancy }
            
        except Exception as _:
            return { 'context_precision': 0.0, 'answer_relevancy': 0.0 }
    
    def _extract_score(self, result, metric_name: str) -> float:
        try:
            scores_array = np.array(result[metric_name], dtype=float)
            valid_scores = scores_array[~np.isnan(scores_array)]
            
            if len(valid_scores) > 0:
                return float(valid_scores.mean())
            
            return 0.0
        except Exception:
            return 0.0