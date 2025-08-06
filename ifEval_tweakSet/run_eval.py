#!/usr/bin/env python3
"""
Combined IFEval Script for Binary and Likert Evaluation
Supports multiple models and configurable parameters via command-line arguments.
"""

import argparse
import json
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from huggingface_hub import login
from typing import List, Dict, Tuple, Optional, Any

# Import utility functions
from utils import (
    softmax,
    get_likert_token_ids,
    get_binary_token_ids,
    valid_ratings,
    extract_score_and_logits,
    extract_character_and_logits,
    logits_calc,
    create_likert_prompt,
    get_pairwise_prompts
)


class IFEvalEvaluator:
    def __init__(self, model_name: str, access_token: str, cache_dir: str = None, 
                 quantization: str = "4bit", device_map: str = "auto"):
        """
        Initialize the IFEval evaluator with model and configuration.
        
        Args:
            model_name: HuggingFace model name
            access_token: HuggingFace access token
            cache_dir: Directory for model cache
            quantization: Quantization type ("4bit", "8bit", or "none")
            device_map: Device mapping strategy
        """
        self.model_name = model_name
        self.access_token = access_token
        self.cache_dir = cache_dir
        self.device_map = device_map
        
        # Login to HuggingFace
        login(token=access_token)
        
        # Setup quantization config
        if quantization == "4bit":
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "8bit":
            self.bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            self.bnb_config = None
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        
        print(f"Initialized IFEval evaluator with model: {model_name}")
    
    def generate_likert_ratings(self, prompt: str, response: str) -> Tuple[Optional[str], Optional[List[float]], Optional[float]]:
        """Generate Likert scale ratings for a response."""
        likert_prompt = create_likert_prompt(prompt, response)
        
        input_ids = self.tokenizer.encode(likert_prompt, return_tensors="pt").cuda()
        generation_output = self.model.generate(
            input_ids,
            max_new_tokens=32,
            do_sample=False,
            temperature=1.0,
            output_scores=True,
            return_dict_in_generate=True
        )
        
        scores, logit_numbers = extract_score_and_logits(generation_output, input_ids, self.tokenizer)
        
        if scores is not None and logit_numbers:
            logit_numbers_list = [float(value) for value in logit_numbers.values()]
            return str(scores), logit_numbers_list, logits_calc(logit_numbers_list)
        else:
            return None, None, None
    
    def get_binary_logits(self, prompt: str, pairs: List[str]) -> Tuple[List[str], List[List[float]]]:
        """Get binary logits for pairwise comparison."""
        agg_logits = []
        text_responses = []
        
        # Swap order to mitigate position bias
        for chosen_position in [0, 1]:
            response_A = pairs[chosen_position]
            response_B = pairs[1 - chosen_position]
            user_prompt = get_pairwise_prompts(prompt, response_A, response_B)
            
            input_ids = self.tokenizer.encode(user_prompt, return_tensors="pt").cuda()
            generation_output = self.model.generate(
                input_ids,
                max_new_tokens=32,
                do_sample=False,
                temperature=1.0,
                output_scores=True,
                return_dict_in_generate=True
            )
            
            response = self.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
            char_inside_brackets, logit_A, logit_B = extract_character_and_logits(generation_output, input_ids, self.tokenizer)
            text_responses.append(char_inside_brackets)
            
            bin_logits = [logit_A, logit_B]
            agg_logits.append([bin_logits[chosen_position], bin_logits[1-chosen_position]])
        
        return text_responses, agg_logits
    
    def evaluate_likert(self, data_file: str, comparison_type: str) -> Dict[str, Any]:
        """Run Likert scale evaluation on the dataset."""
        print(f"\n=== Starting Likert evaluation with model: {self.model_name} ===")
        print(f"Comparison type: {comparison_type}")
        
        with open(data_file, 'r') as f:
            ifeval_mod = json.load(f)
        
        # Define comparison mappings
        comparison_mappings = {
            "severity": ["severity_1", "severity_2", "severity_3"],
            "distractor": ["distractor_1", "distractor_2", "distractor_3"],
            "assert": ["assert_response"],
            "verbose": ["verbose_output"],
            "sycophantic": ["sycophantic_output"]
        }
        
        if comparison_type not in comparison_mappings:
            print(f"Error: Unknown comparison type '{comparison_type}'. Available types: {list(comparison_mappings.keys())}")
            return {}
        
        comparison_fields = comparison_mappings[comparison_type]
        counts = {field: 0 for field in comparison_fields}
        total_valid = 0
        
        for i, data in enumerate(tqdm(ifeval_mod, desc=f"Processing IFEval (Likert - {comparison_type})")):
            prompt = data["prompt"]
            original_response = data["orig_response"]
            
            # Generate rating for original response
            base_rating, base_logits, base_mean = self.generate_likert_ratings(prompt, original_response)
            
            if not valid_ratings(base_rating):
                continue
            
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt: {prompt[:100]}...")
            print(f"Original Response Rating: {base_rating} (Mean: {base_mean:.3f})")
            
            # Generate ratings for comparison responses
            comparison_ratings = {}
            for field in comparison_fields:
                if field in data:
                    response = data[field]
                    rating, logits, mean_score = self.generate_likert_ratings(prompt, response)
                    comparison_ratings[field] = {
                        "rating": rating,
                        "mean_score": mean_score,
                        "logits": logits
                    }
                    
                    if valid_ratings(rating):
                        print(f"{field} Rating: {rating} (Mean: {mean_score:.3f})")
                        
                        # Count if original is better
                        if int(base_rating) > int(rating):
                            counts[field] += 1
                    else:
                        print(f"{field} Rating: Invalid")
                else:
                    print(f"Warning: Field '{field}' not found in data")
            
            # Check if all ratings are valid
            all_valid = all(valid_ratings(comparison_ratings[field]["rating"]) 
                           for field in comparison_fields if field in data)
            if all_valid:
                total_valid += 1
        
        # Print summary
        print(f"\n=== Likert Evaluation Results ===")
        print(f"Model: {self.model_name}")
        print(f"Comparison Type: {comparison_type}")
        print(f"Total Valid Comparisons: {total_valid}")
        
        for field in comparison_fields:
            if total_valid > 0:
                percentage = (counts[field] / total_valid) * 100
                print(f"{field}: {counts[field]}/{total_valid} ({percentage:.1f}%) - Original preferred")
            else:
                print(f"{field}: No valid comparisons")
        
        return {
            "model": self.model_name,
            "evaluation_type": "likert",
            "comparison_type": comparison_type,
            "counts": counts,
            "total_valid": total_valid
        }
    
    def evaluate_binary(self, data_file: str, comparison_type: str) -> Dict[str, Any]:
        """Run binary pairwise evaluation on the dataset."""
        print(f"\n=== Starting binary evaluation with model: {self.model_name} ===")
        print(f"Comparison type: {comparison_type}")
        
        with open(data_file, 'r') as f:
            ifeval_mod = json.load(f)
        
        # Define comparison mappings
        comparison_mappings = {
            "severity": ["severity_1", "severity_2", "severity_3"],
            "distractor": ["distractor_1", "distractor_2", "distractor_3"],
            "assert": ["assert_response"],
            "verbose": ["verbose_output"],
            "sycophantic": ["sycophantic_output"]
        }
        
        if comparison_type not in comparison_mappings:
            print(f"Error: Unknown comparison type '{comparison_type}'. Available types: {list(comparison_mappings.keys())}")
            return {}
        
        comparison_fields = comparison_mappings[comparison_type]
        counts = {field: 0 for field in comparison_fields}
        discards = {field: 0 for field in comparison_fields}
        total_samples = len(ifeval_mod)
        
        for i, data in enumerate(tqdm(ifeval_mod, desc=f"Processing IFEval (Binary - {comparison_type})")):
            prompt = data["prompt"]
            base_response = data["orig_response"]
            
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt: {prompt[:100]}...")
            
            # Evaluate each comparison field
            for field in comparison_fields:
                if field in data:
                    comparison_response = data[field]
                    
                    # Get binary logits
                    text_responses, logits = self.get_binary_logits(prompt, [base_response, comparison_response])
                    
                    # Check for valid results
                    if any(char is None for char in text_responses) or any(any(entry is None for entry in sublist) for sublist in logits):
                        discards[field] += 1
                        print(f"{field}: Invalid response - discarded")
                        continue
                    
                    # Calculate aggregated logits
                    agg_logit = [x + y for x, y in zip(logits[0], logits[1])]
                    
                    # Determine preference
                    if agg_logit[0] < agg_logit[1]:
                        counts[field] += 1
                        preference = "Original preferred"
                    else:
                        preference = "Comparison preferred"
                    
                    print(f"{field}: {text_responses} | Logits: {agg_logit} | {preference}")
                else:
                    print(f"Warning: Field '{field}' not found in data")
        
        # Print summary
        print(f"\n=== Binary Evaluation Results ===")
        print(f"Model: {self.model_name}")
        print(f"Comparison Type: {comparison_type}")
        print(f"Total Samples: {total_samples}")
        
        for field in comparison_fields:
            valid_samples = total_samples - discards[field]
            if valid_samples > 0:
                percentage = (counts[field] / valid_samples) * 100
                print(f"{field}: {counts[field]}/{valid_samples} ({percentage:.1f}%) - Original preferred | Discards: {discards[field]}")
            else:
                print(f"{field}: No valid samples | Discards: {discards[field]}")
        
        return {
            "model": self.model_name,
            "evaluation_type": "binary",
            "comparison_type": comparison_type,
            "counts": counts,
            "discards": discards,
            "total_samples": total_samples
        }


def main():
    parser = argparse.ArgumentParser(description="Combined IFEval Script for Binary and Likert Evaluation")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                       help="HuggingFace model name")
    parser.add_argument("--access_token", type=str, required=True,
                       help="HuggingFace access token")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Directory for model cache")
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit", "none"], default="4bit",
                       help="Quantization type")
    parser.add_argument("--device_map", type=str, default="auto",
                       help="Device mapping strategy")
    
    # Evaluation configuration
    parser.add_argument("--eval_type", type=str, choices=["binary", "likert", "both"], required=True,
                       help="Evaluation type: binary, likert, or both")
    parser.add_argument("--comparison_type", type=str, 
                       choices=["severity", "distractor", "assert", "verbose", "sycophantic"], required=True,
                       help="Type of comparison to evaluate")
    parser.add_argument("--data_file", type=str, 
                       default="data_gpt/binary_resolution.json",
                       help="Path to input data file")
    
    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=32,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Generation temperature")
    parser.add_argument("--do_sample", action="store_true", default=False,
                       help="Enable sampling during generation")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = IFEvalEvaluator(
        model_name=args.model,
        access_token=args.access_token,
        cache_dir=args.cache_dir,
        quantization=args.quantization,
        device_map=args.device_map
    )
    
    # Run evaluation based on type
    if args.eval_type in ["binary", "both"]:
        binary_results = evaluator.evaluate_binary(args.data_file, args.comparison_type)
        print(f"\nBinary evaluation completed for {args.comparison_type}")
    
    if args.eval_type in ["likert", "both"]:
        likert_results = evaluator.evaluate_likert(args.data_file, args.comparison_type)
        print(f"\nLikert evaluation completed for {args.comparison_type}")
    
    print("\n=== All evaluations completed successfully! ===")


if __name__ == "__main__":
    main() 