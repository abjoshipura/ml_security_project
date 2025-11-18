"""
Fine-tune Llama-2-7b-chat on Tiny-NQ dataset.
Uses LoRA for efficient fine-tuning.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import yaml
from src.data.tiny_nq_loader import TinyNQLoader

class LlamaFineTuner:
    """Fine-tune Llama-2-7b-chat on Tiny-NQ"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']['llama2']
        self.ft_config = self.config['fine_tuning']
        
        self.output_dir = "models/llama2-tiny-nq-finetuned"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def prepare_data(self) -> Dataset:
        """Load and prepare Tiny-NQ data"""
        
        print("Loading Tiny-NQ dataset...")
        loader = TinyNQLoader()
        data = loader.download_and_prepare()
        
        train_data = data['train']
        
        # Format for training
        formatted_data = []
        for item in train_data:
            # Format: Question + Answer
            text = f"<s>[INST] {item['question']} [/INST] {item['long_answer']}</s>"
            formatted_data.append({'text': text})
        
        return Dataset.from_list(formatted_data)
    
    def load_model_and_tokenizer(self):
        """Load Llama-2 model with LoRA"""
        
        print("Loading Llama-2-7b-chat...")
        
        model_path = self.model_config['model']
        hf_token = os.getenv(self.model_config['hf_token_env'])
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=hf_token,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model in 8-bit
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            token=hf_token,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.ft_config['lora_r'],
            lora_alpha=self.ft_config['lora_alpha'],
            target_modules=["q_proj", "v_proj"],
            lora_dropout=self.ft_config['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        print("✓ Model loaded with LoRA adapters")
        self.model.print_trainable_parameters()
    
    def tokenize_function(self, examples):
        """Tokenize data"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.ft_config['max_length'],
            padding="max_length"
        )
    
    def fine_tune(self):
        """Run fine-tuning"""
        
        # Prepare data
        dataset = self.prepare_data()
        print(f"Dataset size: {len(dataset)}")
        
        # Tokenize
        print("Tokenizing...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Load model
        self.load_model_and_tokenizer()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.ft_config['num_epochs'],
            per_device_train_batch_size=self.ft_config['batch_size'],
            gradient_accumulation_steps=4,
            learning_rate=self.ft_config['learning_rate'],
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            warmup_steps=100,
            report_to="none"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        # Train
        print("\n" + "=" * 60)
        print("STARTING FINE-TUNING")
        print("=" * 60)
        
        trainer.train()
        
        # Save
        print("\nSaving fine-tuned model...")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"\n✓ Fine-tuning complete!")
        print(f"Model saved to: {self.output_dir}")

def main():
    finetuner = LlamaFineTuner()
    finetuner.fine_tune()

if __name__ == "__main__":
    main()