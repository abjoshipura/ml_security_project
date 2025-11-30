#!/bin/bash

echo "Welcome to Group 5's Final Project: RAG-based Unlearning"
echo "Let's get your environment set up!"
echo

echo "Step 0: Checking for .env file..."
if [ ! -f .env ]; then
    echo "Seems like you have not set up the .env file yet. Copying the template for you..."
    cp template.env .env
    echo "Done! Please do the following:"
    echo "   1. Fill the .env with your own OpenAI and Google API Keys"
    echo "   2. Re-run this setup file"
    exit 1
fi
echo "File found"
echo

read -p "Enter path to the config file of your choice (e.g., configs/config.yaml): " CONFIG_PATH
echo "Done!"
echo
echo "================================================================================="
echo "IMPORTANT: Please use the same config path/file for any script you run henceforth"
echo "================================================================================="
echo

if [ ! -d .venv ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment"
        exit 1
    fi
    echo "Virtual environment created successfully."
else
    echo "[1/4] Virtual environment already exists... skipping creation."
fi
echo

echo "[2/4] Activating virtual environment..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment"
    exit 1
fi
echo "Virtual environment activated."
echo

echo "[3/4] Installing the required dependencies... [This may take some time]"
pip install -qq -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies"
    exit 1
fi
echo "Dependencies installed successfully."
echo

if [ ! -d data ]; then
    echo "[4/4] Setting up knowledge base... [This may take some time]"
    python scripts/setup_knowledge_base.py --config "$CONFIG_PATH"
    if [ $? -ne 0 ]; then
        echo "Failed to setup knowledge base"
        exit 1
    fi
else
    echo "[4/4] Knowledge base has already been setup... skipping setup."
fi
echo "Done!"
echo

echo "========================================"
echo "Setup Complete!"
echo
echo "Suggested Next Steps:"
echo "---------------------"
echo "- For a test of the RAG itself run:"
echo "  python scripts/test_system.py \"Your query here\""
echo "  python scripts/test_system.py \"Your query here\" --config"
echo "  python scripts/test_system.py \"Your query here\" --verbose"
echo "  python scripts/test_system.py \"Your query here\" --enable-defense"
echo
echo "- To forget a fact/concept:"
echo "  python scripts/query_rag.py --forget \"fact or concept\""
echo
echo "- To reset knowledge bases:"
echo "  python scripts/query_rag.py --reset-kbs"
echo
echo "- For a quick evaluation run:"
echo "  python scripts/run_evaluation.py --quick"
echo
echo "- For a comprehensive evaluation with custom parameters:"
echo "  python src/run_evaluation.py --config configs/config.yaml --num-benign 50 --num-facts-to-forget 50 --num-injection-attacks 60 --num-jailbreak-attacks 60"
echo
echo "  Here are the options better laid out for evaluation:"
echo "  --config                 Path to config file (default: configs/config.yaml)"
echo "  --num-benign             Number of benign queries (default: 25)"
echo "  --num-facts-to-forget    Number of facts to forget (default: 25)"
echo "  --num-injection-attacks  Number of injection attacks (default: 30)"
echo "  --num-jailbreak-attacks  Number of jailbreak attacks (default: 30)"
echo "  --output-dir             Output directory (default: evaluation_results)"
echo "  --quick                  Run quick evaluation with defaults"
echo
echo
echo "IMPORTANT: The cost of the quick evaluation is about \$0.33 to \$0.66. The cost with default params is about \$2."