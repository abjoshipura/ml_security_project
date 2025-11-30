import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.rag_pipeline import RAGUnlearningPipeline

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the RAG on Your Query (may not use the Tiny NQ dataset)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--enable-defense", action='store_true', help="Enable defense mechanisms within the RAG")
    parser.add_argument("--query", type=str, help="Query to ask the RAG system")
    parser.add_argument("--forget", type=str, help="Fact/concept to forget")
    parser.add_argument("--reset-kbs", action='store_true', help="Reset knowledge bases")
    parser.add_argument("--verbose", action='store_true', help="Print verbose output")
    
    args = parser.parse_args()

    if not args.reset_kbs and not args.forget and not args.query:
        parser.error("You must provide either --query, --forget, or --reset-kbs")

    pipeline = RAGUnlearningPipeline(config_path=args.config, enable_defense=args.enable_defense)
    
    if args.reset_kbs:
        print("Resetting KBs...")
        pipeline.kb_manager.reset_unlearned_kb()
        print("Done!")

    if args.forget:
        print(f"Forgetting the fact/concept: {args.forget}")
        result = pipeline.forget_fact(args.forget)
        if result.get('success'):
            print("Fact/Concept Forgotten!")
        else:
            print("Failed at Forgetting Fact/Concept")

    if args.query:
        results = pipeline.query(args.query)
        
        if args.verbose:
            print(results)
        else:
            print("\n\nRAG Response:")
            print(results['response'])