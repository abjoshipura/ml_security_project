import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from src.evaluator import Evaluator

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run RAG Unlearning Evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--num-benign", type=int, default=25)
    parser.add_argument("--num-facts-to-forget", type=int, default=25)
    parser.add_argument("--num-injection-attacks", type=int, default=30)
    parser.add_argument("--num-jailbreak-attacks", type=int, default=30, help="Jailbreak attacks")
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--quick", action='store_true', help="Quick test mode")
    
    args = parser.parse_args()
    
    if args.quick:
        args.num_benign = 5
        args.num_facts_to_forget = 5
        args.num_injection_attacks = 10
        args.num_jailbreak_attacks = 10

    # TODO: Remove
    # if args.quick:
    #     args.num_benign = 150
    #     args.num_forgotten = 100
    #     args.num_pi_attacks = 200
    #     args.num_jb_attacks = 200
    
    evaluator = Evaluator(config_path=args.config, output_dir=args.output_dir)
    
    print("[run_evaluation.py] INFO Starting the full evaluation...")
    
    results = evaluator.run_full_evaluation(
        num_benign_facts=args.num_benign,
        num_facts_to_forget=args.num_facts_to_forget,
        num_injection_attacks=args.num_injection_attacks,
        num_jailbreak_attacks=args.num_jailbreak_attacks
    )

    print("[run_evaluation.py] SUCCESS The evaluation results (and corresponding queries) are available under dir: evaluation_results")