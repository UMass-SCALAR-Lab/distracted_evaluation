#!/usr/bin/env python3
"""
Utility functions for IFEval evaluation scripts.
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute softmax of logits for numerical stability.
    
    Args:
        logits: Input logits array
        
    Returns:
        Softmax probabilities
    """
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def get_likert_token_ids(tokenizer) -> List[int]:
    """
    Get token IDs for Likert scale scores (1-7).
    
    Args:
        tokenizer: HuggingFace tokenizer
        
    Returns:
        List of token IDs for scores 1-7
    """
    scores = ['1', '2', '3', '4', '5', '6', '7']
    score_ids = []
    for score in scores:
        token_id = tokenizer.encode(score, add_special_tokens=False)
        assert len(token_id) == 1
        score_ids.append(token_id[0])
    return score_ids


def get_binary_token_ids(tokenizer) -> List[int]:
    """
    Get token IDs for binary choices (A, B).
    
    Args:
        tokenizer: HuggingFace tokenizer
        
    Returns:
        List of token IDs for 'A' and 'B'
    """
    token_id_A = tokenizer.encode('A', add_special_tokens=False)
    token_id_B = tokenizer.encode('B', add_special_tokens=False)
    
    assert len(token_id_A) == 1 and len(token_id_B) == 1
    return [token_id_A[0], token_id_B[0]]


def valid_ratings(rating: str) -> bool:
    """
    Validate if rating is in valid range.
    
    Args:
        rating: Rating string to validate
        
    Returns:
        True if rating is valid (1-7), False otherwise
    """
    valid_ratings = ['1', '2', '3', '4', '5', '6', '7']
    return rating in valid_ratings


def extract_score_and_logits(generation_output, input_ids, tokenizer) -> Tuple[Optional[int], Optional[Dict]]:
    """
    Extract score and logits from generation output for Likert evaluation.
    
    Args:
        generation_output: Model generation output
        input_ids: Input token IDs
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Tuple of (score, logits_dict) or (None, None) if extraction fails
    """
    try:
        generated_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        
        score_pattern = r'Score:\s*(\d+)'
        score_match = re.search(score_pattern, generated_text)
        
        if not score_match:
            print("No 'Score: [X]' pattern found in the generated text.")
            return None, None
        
        score = int(score_match.group(1))
        
        generated_tokens = tokenizer(generated_text, return_tensors="pt")["input_ids"][0]
        input_tokens_length = input_ids.shape[1]
        
        token_step = None
        for i in range(input_tokens_length, len(generated_tokens)):
            decoded_token = tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
            if str(score) in decoded_token:
                token_step = i - input_tokens_length
                break
        
        if token_step is None or token_step >= len(generation_output.scores):
            print("Could not find the generation step for the score.")
            return None, None
        
        logits = generation_output.scores[token_step]
        number_tokens = {num: tokenizer.encode(str(num), add_special_tokens=False)[0] for num in range(1, 8)}
        logits_numbers = {num: logits[0, token].item() for num, token in number_tokens.items()}
        
        return score, logits_numbers
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None


def extract_character_and_logits(generation_output, input_ids, tokenizer) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Extract character choice and logits from generation output for binary evaluation.
    
    Args:
        generation_output: Model generation output
        input_ids: Input token IDs
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Tuple of (character, logit_A, logit_B) or (None, None, None) if extraction fails
    """
    try:
        generated_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
        
        matches = list(re.finditer(r'\[\[(?:Assistant )?(.)\]\]', generated_text))
        if not matches:
            print("No [[*]] or [[Assistant *]] pattern found in the generated text.")
            return None, None, None
        
        last_match = matches[-1]
        char_inside_brackets = last_match.group(1)
        
        generated_tokens = tokenizer(generated_text, return_tensors="pt")["input_ids"][0]
        input_tokens_length = input_ids.shape[1]
        
        token_step = None
        for i in range(input_tokens_length, len(generated_tokens)):
            decoded_token = tokenizer.decode(generated_tokens[i], skip_special_tokens=True)
            if char_inside_brackets in decoded_token:
                token_step = i - input_tokens_length
                break
        
        if token_step is None or token_step >= len(generation_output.scores):
            print("Could not find the generation step for the character inside brackets.")
            return None, None, None
        
        logits = generation_output.scores[token_step]
        token_A = tokenizer.encode('A', add_special_tokens=False)[0]
        token_B = tokenizer.encode('B', add_special_tokens=False)[0]
        
        logit_A = logits[0, token_A].item()
        logit_B = logits[0, token_B].item()
        
        return char_inside_brackets, logit_A, logit_B
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None


def logits_calc(logits: List[float]) -> float:
    """
    Calculate mean score from logits using softmax.
    
    Args:
        logits: List of logit values
        
    Returns:
        Mean score calculated from softmax probabilities
    """
    prob_distribution = softmax(np.array(logits))
    scores = np.arange(1, 8)
    mean_score = np.sum(prob_distribution * scores)
    return mean_score


def create_likert_prompt(prompt: str, response: str) -> str:
    """
    Create prompt for Likert scale evaluation.
    
    Args:
        prompt: Instruction prompt
        response: Response to evaluate
        
    Returns:
        Formatted Likert evaluation prompt
    """
    return f"""You are given an instruction and a response generated by a language model corresponding to the instruction. The instruction consists of an open ended task marked 'Question' and text format requirements marked 'Format'. Analyze the response for adherence to every text format requirement one by one. Score the response on a scale of 1-7 (1-worst, 7-best) and reduce score for every text formatting requirement not followed.\n#Instruction:{prompt} \n Response: {response}
    \nOnly output the score for the response. Use the following output format: Score:<1-7>"""


def get_pairwise_prompts(prompt: str, responseA: str, responseB: str) -> str:
    """
    Create prompt for pairwise binary evaluation.
    
    Args:
        prompt: Instruction prompt
        responseA: First response to compare
        responseB: Second response to compare
        
    Returns:
        Formatted pairwise evaluation prompt
    """
    return f"""You are given an instruction and two responses 'A' and 'B' generated by a language model. The instruction consists of an open ended task marked 'Question' and text format requirements marked 'Format'. Analyze each response for adherence to every text formatting instruction one by one. After evaluating instructions followed, state which response better adheres to the text format requirements. Output 'A' is response A is better, 'B' if response B is better or 'C' if both responses are equally effective to indicate Tie'. Avoid any position biases and ensure that the
        order in which the responses were presented does not influence your decision. Do not allow
        the length of the responses to influence your evaluation. Do not favor certain names of
        the assistants. Be as objective as possible. Do not give an explanation. Only output your
        final answer as a preference between the two assistants by strictly following this output format: [[A]] if assistant A is better, [[B]]
        if assistant B is better, and [[C]] for a tie. \n#Instruction:{prompt} \n Response A: {responseA} \n Response B: {responseB}""" 