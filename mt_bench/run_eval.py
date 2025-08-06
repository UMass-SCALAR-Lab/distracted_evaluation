import json
import re
from typing import Dict, List, Tuple, Optional, Any
import argparse
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

class PreferenceFlipEvaluator:
    def __init__(
        self,
        model_name: str,
        distractor_data: Dict[str, Any],
        cache_dir: Optional[str] = None,
        output_dir: str = "./results"
    ):
        self.model_name = model_name
        self.distractor_data = distractor_data
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        
        # Initialize model and tokenizer
        self._initialize_model()
        
        # Load benchmark dataset
        self._load_benchmark_dataset()
    
    def _initialize_model(self) -> None:
        """Initialize the model with quantization configuration."""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.model.eval()
        
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    def _load_benchmark_dataset(self) -> None:
        """Load and prepare the benchmark dataset."""
        mt_bench_ = load_dataset("lmsys/mt_bench_human_judgments", split="human")
        self.mt_bench = mt_bench_.filter(lambda x: x['turn'] == 1)
    
    def _get_binary_token_ids(self) -> Tuple[int, int]:
        """Get token IDs for 'A' and 'B'."""
        token_id_A = self.tokenizer.encode('A', add_special_tokens=False)[0]
        token_id_B = self.tokenizer.encode('B', add_special_tokens=False)[0]
        return token_id_A, token_id_B
    
    def _get_score_token_ids(self) -> List[int]:
        """Get token IDs for scores 1-7."""
        return [self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 8)]
    
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def _logits_to_score(self, logits: np.ndarray) -> float:
        """Convert logits to a mean score (1-7)."""
        prob_dist = self._softmax(logits)
        return np.sum(prob_dist * np.arange(1, 8))
    
    def _get_absolute_prompt(self, conv: str) -> str:
        """Generate prompt for absolute scoring."""
        return f"""
        Please act as an impartial judge and evaluate the quality of a response provided by an
        AI assistant to the user question displayed below. Score the response on a scale of 1-7.
        Your evaluation should consider helpfulness, relevance, accuracy, depth, and detail.
        Be objective and don't let length influence you. Only output: Score:<1-7>
        <|Start of Conversation|>
        ### User:
        {conv}
        <|End of Conversation|>
        """
    
    def _get_pairwise_prompt(self, convA: str, convB: str) -> str:
        """Generate prompt for pairwise comparison."""
        return f"""
        Evaluate quality of responses from assistants A and B below.
        Choose which better follows instructions and answers the question.
        Consider helpfulness, relevance, accuracy, depth, and detail.
        Avoid position bias and length influence. Be objective.
        Only output: [[A]] if A is better, [[B]] if B is better, [[C]] for tie.
        <|Assistant A's Conversation|>
        ### User:
        {convA}
        <|End of Assistant A|>
        <|Assistant B's Conversation|>
        ### User:
        {convB}
        <|End of Assistant B|>
        """
    
    def _extract_from_generation(
        self,
        generation_output: Any,
        input_ids: torch.Tensor,
        pattern: str,
        token_values: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
        """Extract generated value and corresponding logits."""
        try:
            generated_text = self.tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
            match = re.search(pattern, generated_text)
            
            if not match:
                return None, None
                
            extracted_value = match.group(1)
            generated_tokens = self.tokenizer(generated_text, return_tensors="pt")["input_ids"][0]
            input_len = input_ids.shape[1]
            
            # Find the token step where the value was generated
            token_step = None
            for i in range(input_len, len(generated_tokens)):
                decoded_token = self.tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
                if extracted_value in decoded_token:
                    token_step = i - input_len
                    break
            
            if token_step is None or token_step >= len(generation_output.scores):
                return None, None
                
            logits = generation_output.scores[token_step][0]
            
            if token_values:
                logits_dict = {k: logits[v].item() for k, v in token_values.items()}
                return extracted_value, logits_dict
                
            return extracted_value, None
            
        except Exception:
            return None, None
    
    def _generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 16
    ) -> Any:
        """Generate model response with scores."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        return self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    def _get_absolute_rating(self, response: str) -> Tuple[Optional[int], Optional[Dict[int, float]], Optional[float]]:
        """Get absolute rating for a response."""
        score_pattern = r'Score:\s*(\d+)'
        score_tokens = {str(i): self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(1, 8)}
        
        generation = self._generate_response(self._get_absolute_prompt(response))
        score, logits = self._extract_from_generation(generation, generation.sequences, score_pattern, score_tokens)
        
        if score and logits:
            logit_values = [float(v) for v in logits.values()]
            return int(score), logits, self._logits_to_score(np.array(logit_values))
        return None, None, None
    
    def _get_pairwise_comparison(self, response_a: str, response_b: str) -> Tuple[Optional[str], Optional[List[List[float]]]]:
        """Get pairwise comparison between two responses."""
        token_a, token_b = self._get_binary_token_ids()
        pattern = r'\[\[(?:Assistant )?(.)\]\]'
        token_values = {'A': token_a, 'B': token_b}
        
        agg_logits = []
        text_responses = []
        
        for chosen_position in [0, 1]:
            a, b = (response_a, response_b) if chosen_position == 0 else (response_b, response_a)
            generation = self._generate_response(self._get_pairwise_prompt(a, b))
            char, logits = self._extract_from_generation(generation, generation.sequences, pattern, token_values)
            
            text_responses.append(char)
            if logits:
                agg_logits.append([logits['A'], logits['B']] if chosen_position == 0 else [logits['B'], logits['A']])
        
        return text_responses, agg_logits if agg_logits else None
    
    def _prepare_conversation(self, conv: List[Dict[str, str]]) -> str:
        """Format conversation for evaluation."""
        return f"""User:
                {conv[0]["content"]}
                ### Assistant A:
                {conv[1]["content"]}
                ### User:
                {conv[2]["content"]}
                ### Assistant A:
                {conv[3]["content"]}"""
    
    def evaluate_pairwise(self) -> Dict[str, Any]:
        """Evaluate pairwise preference flips."""
        results = []
        flips = discards = 0
        
        for data in tqdm(self.mt_bench, desc="Processing (Pairwise)"):
            model_a, model_b = data["model_a"], data["model_b"]
            response_a = self._prepare_conversation(data["conversation_a"])
            response_b = self._prepare_conversation(data["conversation_b"])
            
            # Initial comparison
            base_responses, base_logits = self._get_pairwise_comparison(response_a, response_b)
            if not base_responses or not base_logits:
                discards += 1
                continue
                
            base_agg = [x + y for x, y in zip(base_logits[0], base_logits[1])]
            initial_winner = model_a if base_agg[0] > base_agg[1] else model_b
            low_ranked_model = model_b if initial_winner == model_a else model_a
            
            # Get modified response
            modified_response = self.distractor_data.get(data['question_id'], {}).get(low_ranked_model)
            if not modified_response:
                continue
                
            # Modified comparison
            mod_responses, mod_logits = self._get_pairwise_comparison(
                modified_response if low_ranked_model == model_a else response_a,
                modified_response if low_ranked_model == model_b else response_b
            )
            
            if not mod_responses or not mod_logits:
                discards += 1
                continue
                
            mod_agg = [x + y for x, y in zip(mod_logits[0], mod_logits[1])]
            modified_winner = model_a if mod_agg[0] > mod_agg[1] else model_b
            
            # Check for flip
            if 'C' not in ''.join(mod_responses) and initial_winner != modified_winner:
                flips += 1
                
            results.append({
                "question_id": data["question_id"],
                "initial_winner": initial_winner,
                "modified_winner": modified_winner,
                "flip": initial_winner != modified_winner
            })
        
        return {
            "results": results,
            "stats": {"flips": flips, "discards": discards},
            "config": {
                "model": self.model_name,
                "eval_type": "pairwise"
            }
        }
    
    def evaluate_absolute(self) -> Dict[str, Any]:
        """Evaluate absolute preference flips."""
        results = []
        flips = flips_strict = discards = 0
        
        for data in tqdm(self.mt_bench, desc="Processing (Absolute)"):
            model_a, model_b = data["model_a"], data["model_b"]
            response_a = self._prepare_conversation(data["conversation_a"])
            response_b = self._prepare_conversation(data["conversation_b"])
            
            # Initial ratings
            rating1, _, score1 = self._get_absolute_rating(response_a)
            rating2, _, score2 = self._get_absolute_rating(response_b)
            
            if None in (rating1, rating2, score1, score2):
                discards += 1
                continue
                
            initial_winner = model_a if score1 > score2 else model_b
            low_ranked_model = model_b if initial_winner == model_a else model_a
            
            # Get modified response
            modified_response = self.distractor_data.get(data['question_id'], {}).get(low_ranked_model)
            if not modified_response:
                continue
                
            # Modified ratings
            mod_rating1, _, mod_score1 = self._get_absolute_rating(
                modified_response if low_ranked_model == model_a else response_a
            )
            mod_rating2, _, mod_score2 = self._get_absolute_rating(
                modified_response if low_ranked_model == model_b else response_b
            )
            
            if None in (mod_rating1, mod_rating2, mod_score1, mod_score2):
                discards += 1
                continue
                
            modified_winner = model_a if mod_score1 > mod_score2 else model_b
            
            # Check for flips
            if initial_winner != modified_winner:
                if abs(mod_score1 - mod_score2) >= 0.5:
                    flips_strict += 1
                if abs(mod_score1 - mod_score2) >= 0.1:
                    flips += 1
                    
            results.append({
                "question_id": data["question_id"],
                "initial_winner": initial_winner,
                "modified_winner": modified_winner,
                "flip": initial_winner != modified_winner
            })
        
        return {
            "results": results,
            "stats": {"flips": flips, "flips_strict": flips_strict, "discards": discards},
            "config": {
                "model": self.model_name,
                "eval_type": "absolute"
            }
        }
    
    def save_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save evaluation results to file."""
        with open(f"{self.output_dir}/{filename}", 'w') as f:
            json.dump(results, f, indent=2)

def load_distractor_data(distractor_file: str) -> Dict[str, Any]:
    """Load distractor data from file."""
    with open(distractor_file, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Evaluate preference flips in LLM judgments")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 'Qwen/Qwen2.5-3B-Instruct')")
    parser.add_argument("--distractor_file", type=str, required=True, help="Path to distractor data JSON file")
    parser.add_argument("--cache_dir", type=str, help="Cache directory for models")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--eval_type", type=str, default="both", choices=["pairwise", "absolute", "both"], help="Evaluation type")
    
    args = parser.parse_args()
    
    # Load distractor data
    distractor_data = load_distractor_data(args.distractor_file)
    
    evaluator = PreferenceFlipEvaluator(
        model_name=args.model,
        distractor_data=distractor_data,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir
    )
    
    if args.eval_type in ["pairwise", "both"]:
        pairwise_results = evaluator.evaluate_pairwise()
        evaluator.save_results(pairwise_results, "pairwise_results.json")
        print(f"Pairwise results - Flips: {pairwise_results['stats']['flips']}, Discards: {pairwise_results['stats']['discards']}")
    
    if args.eval_type in ["absolute", "both"]:
        absolute_results = evaluator.evaluate_absolute()
        evaluator.save_results(absolute_results, "absolute_results.json")
        print(f"Absolute results - Flips: {absolute_results['stats']['flips']}, Strict flips: {absolute_results['stats']['flips_strict']}, Discards: {absolute_results['stats']['discards']}")

if __name__ == "__main__":
    main()