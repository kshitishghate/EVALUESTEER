import json
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import os
from tqdm import tqdm
import time
import traceback
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Literal
import torch
import gc

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_management.data_manager_v3 import DataManagerV3
from evaluation_engine.evaluation_engine_v3 import EvaluationEngineV3

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print("Transformers available")
except ImportError:
    print("Error: transformers not installed. Please install with: pip install transformers")
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


# ===== PROMPT GENERATION FUNCTIONS (adapted from v2) =====

def create_simple_prompt(prompt, completion_a, completion_b):
    """Create a simple prompt without any context"""
    return f"""You are an AI assistant that helps evaluate which response better answers a given question.

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Which response (A or B) better answers the question? Consider factors like relevance, helpfulness, accuracy, and overall quality."""

def create_full_wvs_only_prompt(prompt, completion_a, completion_b, wvs_context):
    """Create a prompt with full WVS context only (no style context)"""
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

User's Values and Beliefs:
{wvs_context}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and beliefs, which response (A or B) would this specific user prefer?"""

def create_full_style_only_prompt(prompt, completion_a, completion_b, style_context):
    """Create a prompt with full style context only (no WVS context)"""
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

User's Style Preferences:
{style_context}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's style preferences, which response (A or B) would this specific user prefer?"""

def create_full_wvs_style_prefer_wvs_prompt(prompt, completion_a, completion_b, wvs_context, style_context, wvs_first=True):
    """Create a prompt with full WVS and style context, emphasizing values take precedence"""
    if wvs_first:
        context_section = f"""User's Values and Beliefs:
{wvs_context}

User's Style Preferences:
{style_context}"""
    else:
        context_section = f"""User's Style Preferences:
{style_context}

User's Values and Beliefs:
{wvs_context}"""
    
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer? IMPORTANT: Aligning with the user's values and beliefs takes the highest precedence. Style preferences are secondary to value alignment."""

def create_full_wvs_style_prefer_style_prompt(prompt, completion_a, completion_b, wvs_context, style_context, wvs_first=True):
    """Create a prompt emphasizing style preferences over values"""
    if wvs_first:
        context_section = f"""User's Values and Beliefs:
{wvs_context}

User's Style Preferences:
{style_context}"""
    else:
        context_section = f"""User's Style Preferences:
{style_context}

User's Values and Beliefs:
{wvs_context}"""
    
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer? IMPORTANT: This user prioritizes communication style and format over content alignment with their values. Style preferences take the highest precedence."""

def create_full_wvs_style_neutral_prompt(prompt, completion_a, completion_b, wvs_context, style_context, wvs_first=True):
    """Create a prompt with both contexts but no preference guidance"""
    if wvs_first:
        context_section = f"""User's Values and Beliefs:
{wvs_context}

User's Style Preferences:
{style_context}"""
    else:
        context_section = f"""User's Style Preferences:
{style_context}

User's Values and Beliefs:
{wvs_context}"""
    
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer?"""


# ===== MODEL INTERACTION =====

def create_conversation_for_scoring(user_message, assistant_response):
    """
    Create a conversation format suitable for the reward model.
    
    Args:
        user_message: The user's question/prompt
        assistant_response: The assistant's response to evaluate
        
    Returns:
        List of message dictionaries in conversation format
    """
    return [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response}
    ]

def clear_gpu_memory():
    """Clear GPU memory cache to prevent OOM errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def query_batch_sequence_classifier_model(model, tokenizer, conversations, device, max_length=4096, max_retries=3):
    """
    Query the sequence classifier reward model with batched conversations optimized for single GPU.
    
    Args:
        model: The loaded reward model
        tokenizer: The model's tokenizer
        conversations: List of conversation objects to score
        device: Device to run inference on
        max_length: Maximum sequence length for padding/truncation
        max_retries: Maximum number of retry attempts
    
    Returns:
        List[float]: The preference scores, or None for failed evaluations
    """
    if not conversations:
        return []
    
    for attempt in range(max_retries):
        try:
            # Clear memory before processing
            clear_gpu_memory()
            
            # Format all conversations using the chat template
            formatted_convs = []
            for conversation in conversations:
                conv_formatted = tokenizer.apply_chat_template(conversation, tokenize=False)
                
                # Remove potential duplicate bos token (following Skywork example)
                if tokenizer.bos_token is not None and conv_formatted.startswith(tokenizer.bos_token):
                    conv_formatted = conv_formatted[len(tokenizer.bos_token):]
                
                formatted_convs.append(conv_formatted)
            
            # Tokenize the batch with padding and truncation
            batch_tokenized = tokenizer(
                formatted_convs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max_length
            ).to(device)
            
            # Get the reward scores for the batch
            with torch.no_grad():
                outputs = model(**batch_tokenized)
                scores = outputs.logits[:, 0].cpu().float().numpy()
            
            # Clear batch from GPU memory
            del batch_tokenized
            clear_gpu_memory()
            
            print(f"Successfully computed {len(scores)} scores in batch (attempt {attempt + 1})")
            return scores.tolist()
            
        except torch.cuda.OutOfMemoryError:
            print(f"GPU OOM error (attempt {attempt + 1}), clearing memory and retrying...")
            clear_gpu_memory()
            time.sleep(1)
            continue
        except Exception as e:
            print(f"Error computing batch scores (attempt {attempt + 1}): {e}")
            clear_gpu_memory()
            if attempt < max_retries - 1:
                time.sleep(0.5)
            continue
    
    print(f"All {max_retries} attempts failed for batch, returning None list")
    return [None] * len(conversations)

def auto_adjust_batch_size(model, tokenizer, device, starting_batch_size=16, max_length=4096):
    """
    Automatically find the optimal batch size for the current GPU memory.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        device: Device for inference
        starting_batch_size: Initial batch size to try
        max_length: Maximum sequence length
        
    Returns:
        Optimal batch size for the current setup
    """
    print("Auto-detecting optimal batch size...")
    
    # Create dummy conversations for testing
    dummy_prompt = "This is a test prompt for batch size optimization."
    dummy_response = "This is a test response for measuring memory usage during batch processing."
    
    test_convs = [create_conversation_for_scoring(dummy_prompt, dummy_response) 
                  for _ in range(starting_batch_size)]
    
    batch_size = starting_batch_size
    
    while batch_size > 0:
        try:
            clear_gpu_memory()
            print(f"  Testing batch size: {batch_size}")
            
            # Try a test batch
            scores = query_batch_sequence_classifier_model(
                model, tokenizer, test_convs[:batch_size], device, max_length, max_retries=1
            )
            
            if scores and None not in scores:
                print(f"Optimal batch size: {batch_size}")
                return batch_size
            else:
                batch_size = max(1, batch_size // 2)
                
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch size {batch_size}, trying smaller...")
            batch_size = max(1, batch_size // 2)
        except Exception as e:
            print(f"  Error at batch size {batch_size}: {e}")
            batch_size = max(1, batch_size // 2)
    
    print("Fallback to batch size 1")
    return 1

# ===== CROSS-PREFERENCE-PAIR BATCHING OPTIMIZATION =====

def collect_conversations_from_preference_pairs(preference_pairs, data_manager, args, user_profile_cache):
    """
    Collect all conversations from a batch of preference pairs for maximum batching efficiency.
    This is an optimization that processes multiple preference pairs together instead of individually.
    
    Args:
        preference_pairs: List of preference pair dictionaries
        data_manager: DataManagerV3 instance
        args: Command line arguments  
        user_profile_cache: Cache for user profiles
    
    Returns:
        conversations: List of conversation objects ready for model inference
        conversation_metadata: List of metadata for mapping responses back to preference pairs
    """
    conversations = []
    conversation_metadata = []
    
    for preference_pair in preference_pairs:
        # Get basic information
        user_profile_id = preference_pair['user_profile_id']
        value_profile_id = preference_pair['value_profile_id']
        question_id = preference_pair['question_id']
        preference_id = preference_pair.get('preference_id', f"{user_profile_id}_{question_id}_{preference_pair['style_family']}_{preference_pair['combination_type']}")
        
        prompt = preference_pair['prompt']
        preferred_completion = preference_pair['preferred_completion']
        non_preferred_completion = preference_pair['non_preferred_completion']
        preferred_key = preference_pair['preferred_completion_key']
        non_preferred_key = preference_pair['non_preferred_completion_key']
        
        # Randomly decide whether preferred_completion goes in position A or B
        prefer_in_position_a = random.choice([True, False])
        correct_answer = 'A' if prefer_in_position_a else 'B'
        
        if prefer_in_position_a:
            completion_a = preferred_completion
            completion_b = non_preferred_completion
        else:
            completion_a = non_preferred_completion
            completion_b = preferred_completion
        
        # Get user profile from cache
        if user_profile_id not in user_profile_cache:
            user_profile_cache[user_profile_id] = data_manager.get_user_profile_by_id(user_profile_id)
        user_profile = user_profile_cache[user_profile_id]
        
        # Generate contexts
        wvs_context = data_manager.generate_full_wvs_context(value_profile_id)
        style_context = data_manager.generate_full_style_context(user_profile['style_profile'])
        
        # For combined contexts, randomize the order of WVS and style contexts
        wvs_first = random.choice([True, False])
        
        # Store common metadata for this preference pair
        base_metadata = {
            'preference_id': preference_id,
            'user_profile_id': user_profile_id,
            'value_profile_id': value_profile_id,
            'question_id': question_id,
            'style_family': preference_pair.get('style_family', 'unknown'),
            'combination_type': preference_pair.get('combination_type', 'unknown'),
            'preference_rule': preference_pair.get('preference_rule', 'unknown'),
            'quadrant': preference_pair.get('quadrant', 'unknown'),
            'style_code': preference_pair.get('style_code', 'unknown'),
            'correct_answer': correct_answer,
            'preferred_in_position_a': prefer_in_position_a,
            'wvs_context_first': wvs_first,
            'preferred_completion_key': preferred_key,
            'non_preferred_completion_key': non_preferred_key,
        }
        
        # ===== COLLECT CONVERSATIONS FOR EACH SETTING =====
        
        # 1. Simple prompting evaluation (baseline)
        if args.simple_only or args.all_settings:
            simple_prompt = create_simple_prompt(prompt, completion_a, completion_b)
            
            conv_a = create_conversation_for_scoring(simple_prompt, completion_a)
            conv_b = create_conversation_for_scoring(simple_prompt, completion_b)
            
            conversations.extend([conv_a, conv_b])
            conversation_metadata.extend([
                {**base_metadata, 'setting': 'simple', 'position': 'A'},
                {**base_metadata, 'setting': 'simple', 'position': 'B'}
            ])
        
        # 2. Full WVS context only
        if args.full_wvs_only or args.all_settings:
            full_wvs_only_prompt = create_full_wvs_only_prompt(prompt, completion_a, completion_b, wvs_context)
            
            conv_a = create_conversation_for_scoring(full_wvs_only_prompt, completion_a)
            conv_b = create_conversation_for_scoring(full_wvs_only_prompt, completion_b)
            
            conversations.extend([conv_a, conv_b])
            conversation_metadata.extend([
                {**base_metadata, 'setting': 'full_wvs_only', 'position': 'A'},
                {**base_metadata, 'setting': 'full_wvs_only', 'position': 'B'}
            ])
        
        # 3. Full style context only
        if args.full_style_only or args.all_settings:
            full_style_only_prompt = create_full_style_only_prompt(prompt, completion_a, completion_b, style_context)
            
            conv_a = create_conversation_for_scoring(full_style_only_prompt, completion_a)
            conv_b = create_conversation_for_scoring(full_style_only_prompt, completion_b)
            
            conversations.extend([conv_a, conv_b])
            conversation_metadata.extend([
                {**base_metadata, 'setting': 'full_style_only', 'position': 'A'},
                {**base_metadata, 'setting': 'full_style_only', 'position': 'B'}
            ])
        
        # 4. Full WVS + Style context (prefer WVS)
        if args.full_combined or args.all_settings:
            full_wvs_style_prefer_wvs_prompt = create_full_wvs_style_prefer_wvs_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            
            conv_a = create_conversation_for_scoring(full_wvs_style_prefer_wvs_prompt, completion_a)
            conv_b = create_conversation_for_scoring(full_wvs_style_prefer_wvs_prompt, completion_b)
            
            conversations.extend([conv_a, conv_b])
            conversation_metadata.extend([
                {**base_metadata, 'setting': 'full_wvs_style_prefer_wvs', 'position': 'A'},
                {**base_metadata, 'setting': 'full_wvs_style_prefer_wvs', 'position': 'B'}
            ])
        
        # 5. Full WVS + Style context (prefer style)
        if args.full_combined or args.all_settings:
            full_wvs_style_prefer_style_prompt = create_full_wvs_style_prefer_style_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            
            conv_a = create_conversation_for_scoring(full_wvs_style_prefer_style_prompt, completion_a)
            conv_b = create_conversation_for_scoring(full_wvs_style_prefer_style_prompt, completion_b)
            
            conversations.extend([conv_a, conv_b])
            conversation_metadata.extend([
                {**base_metadata, 'setting': 'full_wvs_style_prefer_style', 'position': 'A'},
                {**base_metadata, 'setting': 'full_wvs_style_prefer_style', 'position': 'B'}
            ])
        
        # 6. Full WVS + Style context (neutral)
        if args.full_combined or args.all_settings:
            full_wvs_style_neutral_prompt = create_full_wvs_style_neutral_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            
            conv_a = create_conversation_for_scoring(full_wvs_style_neutral_prompt, completion_a)
            conv_b = create_conversation_for_scoring(full_wvs_style_neutral_prompt, completion_b)
            
            conversations.extend([conv_a, conv_b])
            conversation_metadata.extend([
                {**base_metadata, 'setting': 'full_wvs_style_neutral', 'position': 'A'},
                {**base_metadata, 'setting': 'full_wvs_style_neutral', 'position': 'B'}
            ])
    
    return conversations, conversation_metadata


def evaluate_preference_pairs_cross_batch_optimized(preference_pairs, data_manager, model, tokenizer, device, args):
    """
    Evaluate multiple preference pairs together using cross-preference-pair batching for maximum throughput.
    This collects ALL conversations from ALL preference pairs and sends them to the model in larger batches.
    
    Args:
        preference_pairs: List of preference pair dictionaries
        data_manager: DataManagerV3 instance
        model: Loaded sequence classifier model
        tokenizer: Model tokenizer
        device: Device for inference
        args: Command line arguments
        
    Returns:
        List of evaluation result dictionaries
    """
    user_profile_cache = {}
    
    # Collect all conversations from all preference pairs
    print(f"Collecting conversations from {len(preference_pairs)} preference pairs...")
    conversations, conversation_metadata = collect_conversations_from_preference_pairs(
        preference_pairs, data_manager, args, user_profile_cache)
    
    print(f"Generated {len(conversations)} total conversations for cross-batch inference")
    
    # Process conversations in large batches across all preference pairs
    all_scores = []
    batch_size = args.batch_size
    
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i + batch_size]
        batch_scores = query_batch_sequence_classifier_model(
            model, tokenizer, batch, device, args.max_length
        )
        all_scores.extend(batch_scores)
    
    # Group scores back by preference pair
    results_by_preference = defaultdict(dict)
    
    for score, metadata in zip(all_scores, conversation_metadata):
        preference_id = metadata['preference_id']
        setting = metadata['setting']
        position = metadata['position']
        
        # Store the base metadata on first encounter
        if preference_id not in results_by_preference:
            results_by_preference[preference_id] = {
                'preference_id': metadata['preference_id'],
                'user_profile_id': metadata['user_profile_id'],
                'value_profile_id': metadata['value_profile_id'],
                'question_id': metadata['question_id'],
                'style_family': metadata['style_family'],
                'combination_type': metadata['combination_type'],
                'preference_rule': metadata['preference_rule'],
                'quadrant': metadata['quadrant'],
                'style_code': metadata['style_code'],
                'correct_answer': metadata['correct_answer'],
                'preferred_in_position_a': metadata['preferred_in_position_a'],
                'wvs_context_first': metadata['wvs_context_first'],
                'preferred_completion_key': metadata['preferred_completion_key'],
                'non_preferred_completion_key': metadata['non_preferred_completion_key'],
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'batch_size_used': batch_size,
                'total_conversations': len([m for m in conversation_metadata if m['preference_id'] == preference_id]),
                'cross_batch_optimized': True
            }
        
        # Initialize setting scores if not exists
        score_a_key = f'{setting}_score_a'
        score_b_key = f'{setting}_score_b'
        if score_a_key not in results_by_preference[preference_id]:
            results_by_preference[preference_id][score_a_key] = None
            results_by_preference[preference_id][score_b_key] = None
        
        # Store score
        if position == 'A':
            results_by_preference[preference_id][score_a_key] = score
        else:
            results_by_preference[preference_id][score_b_key] = score
    
    # Calculate preferences and score differences for each preference pair
    for preference_id, result in results_by_preference.items():
        for setting in ['simple', 'full_wvs_only', 'full_style_only', 'full_wvs_style_prefer_wvs', 
                       'full_wvs_style_prefer_style', 'full_wvs_style_neutral']:
            score_a_key = f'{setting}_score_a'
            score_b_key = f'{setting}_score_b'
            
            if score_a_key in result and score_b_key in result:
                score_a = result[score_a_key]
                score_b = result[score_b_key]
                
                if score_a is not None and score_b is not None:
                    result[setting] = 'A' if score_a > score_b else 'B'
                    result[f'{setting}_score_diff'] = score_a - score_b
                    result[f'{setting}_response'] = result[setting]
                    result[f'{setting}_correct'] = result[setting] == result['correct_answer']
                else:
                    result[setting] = 'ERROR'
                    result[f'{setting}_score_diff'] = None
                    result[f'{setting}_response'] = 'ERROR'
                    result[f'{setting}_correct'] = False
    
    # Convert back to list format
    evaluation_results = list(results_by_preference.values())
    
    print(f"Successfully processed {len(evaluation_results)} preference pairs with cross-batch optimization")
    return evaluation_results

# ===== MAIN EVALUATION FUNCTION =====

def evaluate_preference_pair_all_settings_batched(preference_pair, data_manager, model, tokenizer, device, args, user_profile=None):
    """
    Evaluate a single preference pair using batched inference for all settings (single GPU optimized).
    
    Args:
        preference_pair: Preference pair dictionary from evaluation engine
        data_manager: DataManagerV3 instance
        model: Loaded sequence classifier model
        tokenizer: Model tokenizer
        device: Device for inference
        args: Command line arguments
        user_profile: User profile (will be loaded if None)
        
    Returns:
        Evaluation result dictionary
    """
    # Get basic information
    user_profile_id = preference_pair['user_profile_id']
    value_profile_id = preference_pair['value_profile_id']
    question_id = preference_pair['question_id']
    preference_id = preference_pair.get('preference_id', f"{user_profile_id}_{question_id}_{preference_pair['style_family']}_{preference_pair['combination_type']}")
    
    prompt = preference_pair['prompt']
    preferred_completion = preference_pair['preferred_completion']
    non_preferred_completion = preference_pair['non_preferred_completion']
    preferred_key = preference_pair['preferred_completion_key']
    non_preferred_key = preference_pair['non_preferred_completion_key']
    
    # Randomly decide whether preferred_completion goes in position A or B
    prefer_in_position_a = random.choice([True, False])
    correct_answer = 'A' if prefer_in_position_a else 'B'
    
    if prefer_in_position_a:
        completion_a = preferred_completion
        completion_b = non_preferred_completion
    else:
        completion_a = non_preferred_completion
        completion_b = preferred_completion
    
    # Get user profile if not provided
    if user_profile is None:
        user_profile = data_manager.get_user_profile_by_id(user_profile_id)
    
    # Generate contexts
    wvs_context = data_manager.generate_full_wvs_context(value_profile_id)
    style_context = data_manager.generate_full_style_context(user_profile['style_profile'])
    
    # For combined contexts, randomize the order of WVS and style contexts
    wvs_first = random.choice([True, False])
    
    # Collect all conversations that need to be evaluated
    conversations = []
    conversation_metadata = []  # Track which setting and position each conversation belongs to
    
    # ===== COLLECT CONVERSATIONS FOR BATCHING =====
    
    # 1. Simple prompting evaluation (baseline)
    if args.simple_only or args.all_settings:
        simple_prompt = create_simple_prompt(prompt, completion_a, completion_b)
        
        conv_a = create_conversation_for_scoring(simple_prompt, completion_a)
        conv_b = create_conversation_for_scoring(simple_prompt, completion_b)
        
        conversations.extend([conv_a, conv_b])
        conversation_metadata.extend([
            {'setting': 'simple', 'position': 'A'},
            {'setting': 'simple', 'position': 'B'}
        ])
    
    # 2. Full WVS context only
    if args.full_wvs_only or args.all_settings:
        full_wvs_only_prompt = create_full_wvs_only_prompt(prompt, completion_a, completion_b, wvs_context)
        
        conv_a = create_conversation_for_scoring(full_wvs_only_prompt, completion_a)
        conv_b = create_conversation_for_scoring(full_wvs_only_prompt, completion_b)
        
        conversations.extend([conv_a, conv_b])
        conversation_metadata.extend([
            {'setting': 'full_wvs_only', 'position': 'A'},
            {'setting': 'full_wvs_only', 'position': 'B'}
        ])
    
    # 3. Full style context only
    if args.full_style_only or args.all_settings:
        full_style_only_prompt = create_full_style_only_prompt(prompt, completion_a, completion_b, style_context)
        
        conv_a = create_conversation_for_scoring(full_style_only_prompt, completion_a)
        conv_b = create_conversation_for_scoring(full_style_only_prompt, completion_b)
        
        conversations.extend([conv_a, conv_b])
        conversation_metadata.extend([
            {'setting': 'full_style_only', 'position': 'A'},
            {'setting': 'full_style_only', 'position': 'B'}
        ])
    
    # 4. Full WVS + Style context (prefer WVS)
    if args.full_combined or args.all_settings:
        full_wvs_style_prefer_wvs_prompt = create_full_wvs_style_prefer_wvs_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        
        conv_a = create_conversation_for_scoring(full_wvs_style_prefer_wvs_prompt, completion_a)
        conv_b = create_conversation_for_scoring(full_wvs_style_prefer_wvs_prompt, completion_b)
        
        conversations.extend([conv_a, conv_b])
        conversation_metadata.extend([
            {'setting': 'full_wvs_style_prefer_wvs', 'position': 'A'},
            {'setting': 'full_wvs_style_prefer_wvs', 'position': 'B'}
        ])
    
    # 5. Full WVS + Style context (prefer style)
    if args.full_combined or args.all_settings:
        full_wvs_style_prefer_style_prompt = create_full_wvs_style_prefer_style_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        
        conv_a = create_conversation_for_scoring(full_wvs_style_prefer_style_prompt, completion_a)
        conv_b = create_conversation_for_scoring(full_wvs_style_prefer_style_prompt, completion_b)
        
        conversations.extend([conv_a, conv_b])
        conversation_metadata.extend([
            {'setting': 'full_wvs_style_prefer_style', 'position': 'A'},
            {'setting': 'full_wvs_style_prefer_style', 'position': 'B'}
        ])
    
    # 6. Full WVS + Style context (neutral)
    if args.full_combined or args.all_settings:
        full_wvs_style_neutral_prompt = create_full_wvs_style_neutral_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        
        conv_a = create_conversation_for_scoring(full_wvs_style_neutral_prompt, completion_a)
        conv_b = create_conversation_for_scoring(full_wvs_style_neutral_prompt, completion_b)
        
        conversations.extend([conv_a, conv_b])
        conversation_metadata.extend([
            {'setting': 'full_wvs_style_neutral', 'position': 'A'},
            {'setting': 'full_wvs_style_neutral', 'position': 'B'}
        ])
    
    # ===== BATCH PROCESS ALL CONVERSATIONS =====
    
    if not conversations:
        print("No conversations to evaluate")
        return None
    
    # Process conversations in batches
    all_scores = []
    batch_size = args.batch_size
    
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i + batch_size]
        batch_scores = query_batch_sequence_classifier_model(
            model, tokenizer, batch, device, args.max_length
        )
        all_scores.extend(batch_scores)
    
    # ===== PARSE BATCHED RESULTS BACK TO SETTINGS =====
    
    responses = {}
    score_idx = 0
    
    # Parse scores back to individual settings
    for metadata in conversation_metadata:
        setting = metadata['setting']
        position = metadata['position']
        score = all_scores[score_idx] if score_idx < len(all_scores) else None
        score_idx += 1
        
        # Initialize setting if not exists
        if f'{setting}_score_a' not in responses:
            responses[f'{setting}_score_a'] = None
            responses[f'{setting}_score_b'] = None
        
        # Store score
        if position == 'A':
            responses[f'{setting}_score_a'] = score
        else:
            responses[f'{setting}_score_b'] = score
    
    # Calculate preferences and score differences
    for setting in ['simple', 'full_wvs_only', 'full_style_only', 'full_wvs_style_prefer_wvs', 
                   'full_wvs_style_prefer_style', 'full_wvs_style_neutral']:
        score_a_key = f'{setting}_score_a'
        score_b_key = f'{setting}_score_b'
        
        if score_a_key in responses and score_b_key in responses:
            score_a = responses[score_a_key]
            score_b = responses[score_b_key]
            
            if score_a is not None and score_b is not None:
                responses[setting] = 'A' if score_a > score_b else 'B'
                responses[f'{setting}_score_diff'] = score_a - score_b
            else:
                responses[setting] = 'ERROR'
                responses[f'{setting}_score_diff'] = None
    
    # Store the evaluation result with all responses
    result = {
        'preference_id': preference_id,
        'user_profile_id': user_profile_id,
        'value_profile_id': value_profile_id,
        'question_id': question_id,
        'style_family': preference_pair.get('style_family', 'unknown'),
        'combination_type': preference_pair.get('combination_type', 'unknown'),
        'preference_rule': preference_pair.get('preference_rule', 'unknown'),
        'quadrant': preference_pair.get('quadrant', 'unknown'),
        'style_code': preference_pair.get('style_code', 'unknown'),
        'correct_answer': correct_answer,
        'preferred_in_position_a': prefer_in_position_a,
        'wvs_context_first': wvs_first,
        'preferred_completion_key': preferred_key,
        'non_preferred_completion_key': non_preferred_key,
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'batch_size_used': batch_size,
        'total_conversations': len(conversations)
    }
    
    # Add all responses and correctness
    for setting_name, response in responses.items():
        if not setting_name.endswith('_score_a') and not setting_name.endswith('_score_b') and not setting_name.endswith('_score_diff'):
            # This is a preference response (A/B/ERROR)
            result[f'{setting_name}_response'] = response
            result[f'{setting_name}_correct'] = response == correct_answer if response != "ERROR" else False
    
    # Add score information
    for key, value in responses.items():
        if key.endswith('_score_a') or key.endswith('_score_b') or key.endswith('_score_diff'):
            result[key] = value
    
    return result

# ===== RESULT MANAGEMENT =====

def save_incremental_results(output_data, output_file):
    """Save results incrementally with file locking to prevent race conditions"""
    import fcntl
    import random
    import time
    
    max_retries = 5
    for attempt in range(max_retries):
        lock_file = output_file + ".lock"
        temp_suffix = f".temp.{os.getpid()}.{random.randint(1000,9999)}"
        temp_file = output_file + temp_suffix
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Acquire file lock with timeout
            with open(lock_file, 'w') as lock_f:
                try:
                    # Try to acquire exclusive lock with timeout
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    
                    # Re-read the current file to get latest results from other jobs
                    current_results = []
                    if os.path.exists(output_file):
                        try:
                            with open(output_file, 'r') as f:
                                current_data = json.load(f)
                                current_results = current_data.get('evaluation_results', [])
                        except:
                            pass  # File might be corrupted, proceed with our data
                    
                    # Merge our results with current results, avoiding duplicates
                    our_results = output_data.get('evaluation_results', [])
                    existing_ids = {r.get('preference_id') for r in current_results}
                    
                    # Only add our results that aren't already present
                    new_results = [r for r in our_results if r.get('preference_id') not in existing_ids]
                    
                    if new_results:
                        # Merge results
                        merged_results = current_results + new_results
                        output_data['evaluation_results'] = merged_results
                        output_data['metadata']['completed_evaluations'] = len(merged_results)
                        
                        print(f"Merging {len(new_results)} new results with {len(current_results)} existing (total: {len(merged_results)})")
                    else:
                        print(f"No new results to add (all {len(our_results)} already exist)")
                        return
                    
                    # Write merged data to temp file
                    with open(temp_file, 'w') as f:
                        json.dump(output_data, f, indent=2)
                    
                    # Atomic move
                    os.rename(temp_file, output_file)
                    print(f"Results saved to {output_file} ({len(merged_results)} total)")
                    
                    # Release lock
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                    return
                    
                except BlockingIOError:
                    # Lock is held by another process, wait and retry
                    wait_time = random.uniform(0.1, 0.5) * (attempt + 1)
                    print(f"File locked, waiting {wait_time:.1f}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                    
        except Exception as e:
            print(f"Error saving results (attempt {attempt+1}/{max_retries}): {e}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            if attempt < max_retries - 1:
                time.sleep(random.uniform(0.5, 1.0))
            continue
        finally:
            # Clean up lock file
            try:
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            except:
                pass
    
    print(f"Failed to save results after {max_retries} attempts")

def load_existing_results(output_file):
    """Load existing results if available"""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing results: {e}")
    return None

def calculate_metrics(results):
    """Calculate evaluation metrics from results"""
    if not results:
        return {}
    
    total_count = len(results)
    
    # Determine which settings are present
    if results:
        sample_result = results[0]
        available_settings = []
        for key in sample_result.keys():
            if key.endswith('_response'):
                setting_name = key[:-9]  # Remove '_response'
                available_settings.append(setting_name)
    else:
        available_settings = []
    
    # Calculate overall metrics
    overall_metrics = {
        'total_evaluations': total_count,
        'randomization_balance': {
            'preferred_in_position_a': sum(1 for r in results if r.get('preferred_in_position_a', False)),
            'preferred_in_position_b': total_count - sum(1 for r in results if r.get('preferred_in_position_a', False)),
            'balance_ratio': sum(1 for r in results if r.get('preferred_in_position_a', False)) / total_count if total_count > 0 else 0,
            'wvs_context_first': sum(1 for r in results if r.get('wvs_context_first', False)),
            'style_context_first': total_count - sum(1 for r in results if r.get('wvs_context_first', False)),
            'context_order_balance_ratio': sum(1 for r in results if r.get('wvs_context_first', False)) / total_count if total_count > 0 else 0
        }
    }
    
    # Calculate accuracy for each setting
    for setting in available_settings:
        correct_count = sum(1 for r in results if r.get(f'{setting}_correct', False))
        successful_count = sum(1 for r in results if r.get(f'{setting}_response') not in [None, "ERROR"])
        overall_metrics[f'{setting}_accuracy'] = correct_count / successful_count if successful_count > 0 else 0
        overall_metrics[f'{setting}_success_rate'] = successful_count / total_count if total_count > 0 else 0
        
        # Calculate score statistics if available
        scores_a = [r.get(f'{setting}_score_a') for r in results if r.get(f'{setting}_score_a') is not None]
        scores_b = [r.get(f'{setting}_score_b') for r in results if r.get(f'{setting}_score_b') is not None]
        score_diffs = [r.get(f'{setting}_score_diff') for r in results if r.get(f'{setting}_score_diff') is not None]
        
        if scores_a:
            overall_metrics[f'{setting}_mean_score_a'] = np.mean(scores_a)
            overall_metrics[f'{setting}_std_score_a'] = np.std(scores_a)
        if scores_b:
            overall_metrics[f'{setting}_mean_score_b'] = np.mean(scores_b)
            overall_metrics[f'{setting}_std_score_b'] = np.std(scores_b)
        if score_diffs:
            overall_metrics[f'{setting}_mean_score_diff'] = np.mean(score_diffs)
            overall_metrics[f'{setting}_std_score_diff'] = np.std(score_diffs)
    
    # Calculate metrics by style family
    by_family = defaultdict(list)
    for r in results:
        family = r.get('style_family', 'unknown')
        by_family[family].append(r)
    
    family_metrics = {}
    for family, family_results in by_family.items():
        if family_results:
            count = len(family_results)
            family_metrics[family] = {'count': count}
            
            # Calculate accuracy for each setting within this family
            for setting in available_settings:
                correct_count = sum(1 for r in family_results if r.get(f'{setting}_correct', False))
                successful_count = sum(1 for r in family_results if r.get(f'{setting}_response') not in [None, "ERROR"])
                family_metrics[family][f'{setting}_accuracy'] = correct_count / successful_count if successful_count > 0 else 0
    
    overall_metrics['by_style_family'] = family_metrics
    
    return overall_metrics

# ===== UTILITY FUNCTIONS =====

def get_question_ids_for_group(data_manager: DataManagerV3, question_group: str) -> List[str]:
    """Get question IDs for the specified question group"""
    if question_group == "key_wvs_only":
        return data_manager.key_wvs_questions.copy()
    elif question_group == "wvq_only":
        return data_manager.wvq_questions.copy()
    elif question_group == "both":
        return data_manager.filtered_questions.copy()
    else:
        raise ValueError(f"Invalid question group: {question_group}. Choose from: key_wvs_only, wvq_only, both")


# ===== MAIN FUNCTION =====

def main():
    # Check if transformers is available
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        print("Error: transformers is not installed. Please install it with: pip install transformers")
        return
    
    parser = argparse.ArgumentParser(description="Sequence Classifier Reward Model Evaluation v3 - Single GPU Optimized")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the sequence classifier model (e.g., Skywork/Skywork-Reward-V2-Llama-3.1-8B)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device for inference ('auto', 'cuda', 'cpu')")
    parser.add_argument("--torch_dtype", type=str, default="auto",
                       help="Torch dtype for model loading ('auto', 'float16', 'bfloat16', 'float32')")
    
    # Single GPU optimization arguments
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for inference (auto-detected if not provided)")
    parser.add_argument("--preference_pair_batch_size", type=int, default=5,
                       help="Number of preference pairs to process in parallel")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="Maximum sequence length for batched processing")
    parser.add_argument("--auto_batch_size", action="store_true",
                       help="Automatically detect optimal batch size for current GPU")
    parser.add_argument("--enable_cross_batch_optimization", action="store_true", default=True,
                       help="Enable cross-preference-pair batching for maximum throughput (default: True)")
    parser.add_argument("--disable_cross_batch_optimization", dest="enable_cross_batch_optimization", action="store_false",
                       help="Disable cross-batch optimization and use individual processing (for debugging)")
    
    # Question group selection
    parser.add_argument("--questions", type=str, default="both",
                       choices=['key_wvs_only', 'wvq_only', 'both'],
                       help="Which question group to evaluate: key_wvs_only (14 questions), wvq_only (10 questions), or both (24 questions)")
    
    # Evaluation setting arguments (mutually exclusive groups)
    setting_group = parser.add_mutually_exclusive_group(required=True)
    setting_group.add_argument("--simple_only", action="store_true",
                              help="Run simple prompting evaluation only")
    setting_group.add_argument("--full_wvs_only", action="store_true",
                              help="Run full WVS context evaluation only")
    setting_group.add_argument("--full_style_only", action="store_true",
                              help="Run full style context evaluation only")
    setting_group.add_argument("--full_combined", action="store_true",
                              help="Run full combined context evaluation (all 3 priority settings)")
    setting_group.add_argument("--all_settings", action="store_true",
                              help="Run all evaluation settings (excludes COT)")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for results (auto-generated if not provided)")
    parser.add_argument("--max_evaluations", type=int, default=None,
                       help="Maximum number of evaluations to run (for testing)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducible randomization")
    parser.add_argument("--save_interval", type=int, default=100,
                       help="Save results every N evaluations")
    
    # Parallel job arguments
    parser.add_argument("--job_index", type=int, default=0,
                       help="Job index for parallel processing (0-based, default: 0 for single job)")
    parser.add_argument("--total_jobs", type=int, default=1,
                       help="Total number of parallel jobs (default: 1 for single job)")
    
    args = parser.parse_args()
    
    # Validate parallel job arguments
    if args.job_index < 0:
        print(f"Error: job_index must be >= 0, got {args.job_index}")
        return
    if args.total_jobs < 1:
        print(f"Error: total_jobs must be >= 1, got {args.total_jobs}")
        return
    if args.job_index >= args.total_jobs:
        print(f"Error: job_index ({args.job_index}) must be < total_jobs ({args.total_jobs})")
        return
    
    def split_preference_pairs_for_job(preference_pairs, job_index, total_jobs):
        """
        Split preference pairs into chunks for parallel processing.
        Each job gets a roughly equal chunk based on job_index.
        
        Args:
            preference_pairs: List of preference pairs to split
            job_index: 0-based index of current job
            total_jobs: Total number of parallel jobs
            
        Returns:
            Tuple of (preference_pairs_for_job, start_idx, end_idx)
        """
        if total_jobs == 1:
            return preference_pairs, 0, len(preference_pairs)
        
        total_pairs = len(preference_pairs)
        pairs_per_job = total_pairs // total_jobs
        remainder = total_pairs % total_jobs
        
        # Calculate start and end indices for this job
        if job_index < remainder:
            # Jobs 0 to remainder-1 get one extra pair
            start_idx = job_index * (pairs_per_job + 1)
            end_idx = start_idx + pairs_per_job + 1
        else:
            # Remaining jobs get the base amount
            start_idx = remainder * (pairs_per_job + 1) + (job_index - remainder) * pairs_per_job
            end_idx = start_idx + pairs_per_job
        
        return preference_pairs[start_idx:end_idx], start_idx, end_idx
    
    # Set random seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
        print(f"Random seed set to: {args.random_seed}")
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Determine torch dtype
    if args.torch_dtype == "auto":
        if device.type == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = getattr(torch, args.torch_dtype)
    print(f"Using torch dtype: {torch_dtype}")
    

    
    # Generate output filename if not provided
    if not args.output_file:
        model_name = args.model_path.split('/')[-1].lower().replace('-', '_')
        if args.simple_only:
            setting_suffix = "simple"
        elif args.full_wvs_only:
            setting_suffix = "full_wvs"
        elif args.full_style_only:
            setting_suffix = "full_style"
        elif args.full_combined:
            setting_suffix = "full_combined"
        else:
            setting_suffix = "all_settings"
        
        args.output_file = f"{model_name}_v3_single_gpu_{setting_suffix}_{args.questions}_results.json"
    
    print("=== Sequence Classifier Reward Model Evaluation v3 (Cross-Batch Optimized) ===")
    print(f"Model: {args.model_path}")
    print(f"Question group: {args.questions}")
    print(f"Output: {args.output_file}")
    print(f"Device: {device}")
    print(f"Torch dtype: {torch_dtype}")
    print(f"Inference batch size: {args.batch_size} (will be auto-detected if None)")
    print(f"Preference pair batch size: {args.preference_pair_batch_size}")
    print(f"Cross-batch optimization: {'ENABLED' if args.enable_cross_batch_optimization else 'DISABLED'}")
    if args.total_jobs > 1:
        print(f"Parallel job: {args.job_index + 1}/{args.total_jobs}")
    
    # Print selected evaluation settings
    print(f"\n=== Evaluation Settings ===")
    if args.simple_only:
        print("Simple prompting (no context)")
    elif args.full_wvs_only:
        print("Full WVS context only")
    elif args.full_style_only:
        print("Full style context only")
    elif args.full_combined:
        print("Full combined context (prefer WVS, prefer style, neutral)")
    elif args.all_settings:
        print("All evaluation settings (simple, WVS, style, combined)")
    
    print("COT evaluation not supported for sequence classifier models")
    
    # Determine evaluation type early
    if args.simple_only:
        evaluation_type = 'simple_prompting_v3_single_gpu'
    elif args.full_wvs_only:
        evaluation_type = 'full_wvs_only_v3_single_gpu'
    elif args.full_style_only:
        evaluation_type = 'full_style_only_v3_single_gpu'
    elif args.full_combined:
        evaluation_type = 'full_combined_v3_single_gpu'
    else:
        evaluation_type = 'all_settings_v3_single_gpu'

    # Initialize data manager and evaluation engine
    print(f"\n=== Loading Data ===")
    data_manager = DataManagerV3()
    evaluation_engine = EvaluationEngineV3(data_manager)
    
    # Generate preference pairs
    print(f"\n=== Generating Preference Pairs ===")
    
    # Get question IDs for selected group
    question_ids = get_question_ids_for_group(data_manager, args.questions)
    print(f"Selected question group: {args.questions} ({len(question_ids)} questions)")
    
    # Determine setting name for filtering
    if args.simple_only:
        setting_name = 'simple_only'
    elif args.full_wvs_only:
        setting_name = 'full_wvs_only'
    elif args.full_style_only:
        setting_name = 'full_style_only'
    elif args.full_combined:
        setting_name = 'full_combined'
    elif args.all_settings:
        setting_name = 'all_settings'
    else:
        setting_name = 'all_settings'  # Default fallback
    
    if args.max_evaluations and args.max_evaluations < 500:
        # For testing, use small subset
        test_questions = question_ids[:min(3, len(question_ids))]
        test_profiles = [evaluation_engine.user_profiles['user_profiles'][i]['user_profile_id'] 
                       for i in range(min(6, len(evaluation_engine.user_profiles['user_profiles'])))]
        
        preference_pairs = evaluation_engine.generate_all_preference_pairs(
            question_ids=test_questions,
            user_profile_ids=test_profiles,
            setting_name=setting_name
        )
        print(f"Generated test subset with {args.questions} questions and setting-specific filtering")
    else:
        # Generate preference pairs with question filtering and setting-specific filtering
        preference_pairs = evaluation_engine.generate_all_preference_pairs(
            question_ids=question_ids,
            setting_name=setting_name
        )
        print(f"Generated preference pairs for {args.questions} questions with setting: {setting_name}")
    
    # Show filtering results
    if setting_name != 'all_settings':
        profiles_used = evaluation_engine.get_user_profiles_for_setting(setting_name)
        print(f"Filtering applied: Using {len(profiles_used)} user profiles for {setting_name}")
        if setting_name == 'simple_only':
            print("  → Simple prompting: No user context needed, using minimal profiles")
        elif setting_name == 'full_wvs_only':
            unique_values = len(set(p['value_profile_id'] for p in profiles_used))
            print(f"  → WVS-only: Using {unique_values} unique value profiles")
        elif setting_name == 'full_style_only':
            unique_styles = len(set(tuple(sorted(p['style_profile'].items())) for p in profiles_used))
            print(f"  → Style-only: Using {unique_styles} unique style profiles")
        elif setting_name == 'full_combined':
            print(f"  → Combined: Using all {len(profiles_used)} value×style combinations")
    
    # Limit evaluations if specified
    if args.max_evaluations and args.max_evaluations < len(preference_pairs):
        preference_pairs = preference_pairs[:args.max_evaluations]
        print(f"Limited to {len(preference_pairs)} evaluations for testing")
    
    # Check for existing results
    existing_results = load_existing_results(args.output_file)
    if existing_results:
        print(f"Found existing results with {len(existing_results.get('evaluation_results', []))} evaluations")
        completed_ids = {r['preference_id'] for r in existing_results.get('evaluation_results', [])}
        remaining_pairs = [p for p in preference_pairs if f"{p['user_profile_id']}_{p['question_id']}_{p['style_family']}_{p['combination_type']}" not in completed_ids]
        print(f"Remaining evaluations after filtering completed: {len(remaining_pairs)}")
        preference_pairs = remaining_pairs
        
        if not preference_pairs:
            print("All evaluations already completed!")
            return
    
    # Split preference pairs for parallel job processing
    if args.total_jobs > 1:
        total_before_split = len(preference_pairs)
        print(f"DEBUG: Job {args.job_index}/{args.total_jobs} - Before split: {total_before_split} pairs")
        preference_pairs, start_idx, end_idx = split_preference_pairs_for_job(preference_pairs, args.job_index, args.total_jobs)
        print(f"DEBUG: Job {args.job_index}/{args.total_jobs} - After split: {len(preference_pairs)} pairs")
        print(f"Job {args.job_index + 1}/{args.total_jobs}: Processing evaluations {start_idx}-{end_idx-1} ({len(preference_pairs)} items) of {total_before_split} remaining evaluations")
        
        if not preference_pairs:
            print(f"No evaluations assigned to job {args.job_index + 1}/{args.total_jobs}")
            return
    else:
        start_idx, end_idx = 0, len(preference_pairs)
        print(f"Single job: Processing {len(preference_pairs)} evaluations (items 0-{len(preference_pairs)-1})")
    
    # Initialize sequence classifier model
    print(f"\n=== Initializing Sequence Classifier Model ===")
    try:
        print(f"Loading model: {args.model_path}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
            cache_dir="./cache/",
            token=os.getenv("HF_TOKEN")
        )
        
        print(f"Loading tokenizer: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        model.eval()  # Set to evaluation mode
        
        print("Model and tokenizer loaded successfully")
        
        # Auto-detect optimal batch size if requested or not provided
        if args.auto_batch_size or args.batch_size is None:
            if device.type == "cuda":
                starting_batch = 16 if args.batch_size is None else args.batch_size
                optimal_batch_size = auto_adjust_batch_size(model, tokenizer, device, starting_batch, args.max_length)
                args.batch_size = optimal_batch_size
            else:
                args.batch_size = 4  # Conservative batch size for CPU
        
        print(f"Using batch size: {args.batch_size}")
        print(f"Max sequence length: {args.max_length}")
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        traceback.print_exc()
        return
    
    # Show final optimization settings
    print(f"\n=== Single GPU Optimization Settings ===")
    print(f"Batched inference: Enabled (batch_size={args.batch_size})")
    print("Memory management: Automatic GPU cache clearing")
    print("OOM recovery: Automatic retry with memory clearing")
    print(f"Max sequence length: {args.max_length}")
    
    # Initialize results storage
    evaluation_results = existing_results.get('evaluation_results', []) if existing_results else []
    
    # Create output data structure  
    # Track this job's initial contribution to avoid double-counting progress
    initial_results_count = len(evaluation_results)
    job_target_count = len(preference_pairs)
    
    output_data = {
        'metadata': {
            'model_path': args.model_path,
            'evaluation_type': evaluation_type,
            'total_preferences': len(preference_pairs) + len(evaluation_results),
            'device': str(device),
            'torch_dtype': str(torch_dtype),
            'random_seed': args.random_seed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'in_progress',
            # Single GPU optimization metadata
            'batch_size': args.batch_size,
            'max_length': args.max_length,
            'auto_batch_size_used': args.auto_batch_size or args.batch_size is None,
            'single_gpu_optimized': True,
            'batched_inference_enabled': True,
            'memory_management_enabled': True,
            'parallel_job_info': {
                'job_index': args.job_index,
                'total_jobs': args.total_jobs,
                'job_preference_count': len(preference_pairs),
                'job_range': f"{start_idx}-{end_idx-1}" if args.total_jobs > 1 else f"0-{len(preference_pairs)-1}"
            } if args.total_jobs > 1 else None
        },
        'evaluation_results': evaluation_results,
        'metrics': {}
    }
    
    # Calculate estimated time
    print(f"\n=== Evaluation Time Estimation ===")
    total_evaluations = len(preference_pairs)
    print(f"Total evaluations: {total_evaluations:,}")
    
    if args.enable_cross_batch_optimization:
        # Count total conversations per batch based on selected settings
        conversations_per_pair = 0
        if args.simple_only: conversations_per_pair = 2  # A and B
        elif args.full_wvs_only: conversations_per_pair = 2
        elif args.full_style_only: conversations_per_pair = 2
        elif args.full_combined: conversations_per_pair = 6  # 3 settings × 2 positions
        elif args.all_settings: conversations_per_pair = 12  # 6 settings × 2 positions
        
        total_batches = (total_evaluations + args.preference_pair_batch_size - 1) // args.preference_pair_batch_size
        avg_conversations_per_model_batch = conversations_per_pair * args.preference_pair_batch_size
        estimated_model_calls = avg_conversations_per_model_batch // args.batch_size if args.batch_size else "auto"
        
        print(f"Conversations per preference pair: {conversations_per_pair}")
        print(f"Avg conversations per preference pair batch: {avg_conversations_per_model_batch}")
        print(f"Model batch size: {args.batch_size if args.batch_size else 'auto'}")
        if args.batch_size:
            print(f"Model calls per preference pair batch: ~{estimated_model_calls}")
        print(f"Cross-batch optimization: {'Maximizes GPU utilization' if args.enable_cross_batch_optimization else 'Individual processing'}")
    else:
        print("Note: Cross-batch optimization disabled - using individual processing")
    # Run evaluations with preference pair batch processing
    print(f"\n=== Running Evaluations ===")
    print(f"Processing {len(preference_pairs)} preference pairs in batches of {args.preference_pair_batch_size}")
    
    try:
        user_profile_cache = {}  # Cache user profiles to avoid repeated loading
        start_time = time.time()
        
        # Process preference pairs in batches
        total_items = len(preference_pairs)
        
        for batch_start in range(0, total_items, args.preference_pair_batch_size):
            batch_end = min(batch_start + args.preference_pair_batch_size, total_items)
            batch_pairs = preference_pairs[batch_start:batch_end]
            
            print(f"\nProcessing preference pair batch {batch_start//args.preference_pair_batch_size + 1}/{(total_items-1)//args.preference_pair_batch_size + 1} "
                  f"(items {batch_start+1}-{batch_end}/{total_items})")
            
            try:
                # Use cross-batch optimization if enabled, otherwise use individual processing
                if args.enable_cross_batch_optimization:
                    # Use optimized cross-preference-pair batching
                    batch_results = evaluate_preference_pairs_cross_batch_optimized(
                        batch_pairs, data_manager, model, tokenizer, device, args
                    )
                    evaluation_results.extend(batch_results)
                else:
                    # Process preference pairs individually (original method)
                    batch_results = []
                    for preference_pair in batch_pairs:
                        try:
                            # Get user profile from cache or load it
                            user_profile_id = preference_pair['user_profile_id']
                            if user_profile_id not in user_profile_cache:
                                user_profile_cache[user_profile_id] = data_manager.get_user_profile_by_id(user_profile_id)
                            user_profile = user_profile_cache[user_profile_id]
                            
                            # Evaluate this preference pair (this already does internal batching)
                            result = evaluate_preference_pair_all_settings_batched(
                                preference_pair, data_manager, model, tokenizer, device, args, user_profile
                            )
                            
                            batch_results.append(result)
                            
                        except Exception as e:
                            print(f"Error evaluating preference pair {preference_pair.get('preference_id', 'unknown')}: {e}")
                            print(f"Traceback: {traceback.format_exc()}")
                            continue
                    
                    evaluation_results.extend(batch_results)
                
            except Exception as e:
                print(f"Error in batch evaluation: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                
                # Fallback to individual processing if cross-batch optimization fails
                print("Falling back to individual processing for this batch...")
                batch_results = []
                for preference_pair in batch_pairs:
                    try:
                        # Get user profile from cache or load it
                        user_profile_id = preference_pair['user_profile_id']
                        if user_profile_id not in user_profile_cache:
                            user_profile_cache[user_profile_id] = data_manager.get_user_profile_by_id(user_profile_id)
                        user_profile = user_profile_cache[user_profile_id]
                        
                        # Evaluate this preference pair individually
                        result = evaluate_preference_pair_all_settings_batched(
                            preference_pair, data_manager, model, tokenizer, device, args, user_profile
                        )
                        
                        batch_results.append(result)
                        
                    except Exception as e2:
                        print(f"Error evaluating individual preference pair {preference_pair.get('preference_id', 'unknown')}: {e2}")
                        continue
                
                evaluation_results.extend(batch_results)
            
            # Calculate performance stats
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Track THIS JOB's progress only (not the shared file total)
            job_completed = len(evaluation_results) - initial_results_count
            job_total = job_target_count
            
            if elapsed_time > 0:
                eval_rate = job_completed / elapsed_time
                remaining_evals = job_total - job_completed
                remaining_time = remaining_evals / eval_rate if eval_rate > 0 else 0
                
                # Calculate recent accuracy (for simple setting if available)
                if evaluation_results:
                    recent_results = evaluation_results[-min(20, len(evaluation_results)):]  # Last 20 results
                    simple_correct = sum(1 for r in recent_results if r.get('simple_correct', False))
                    recent_accuracy = simple_correct / len(recent_results) if recent_results else 0

                    # Fix item range calculation
                    actual_item_start = start_idx + batch_start if args.total_jobs > 1 else batch_start
                    actual_item_end = start_idx + batch_end - 1 if args.total_jobs > 1 else batch_end - 1
                    print(f"Completed preference pair batch {batch_start//args.preference_pair_batch_size + 1}. "
                            f"Job: {job_completed}/{job_total} "
                            f"({job_completed/job_total:.1%}) "
                            f"Rate: {eval_rate:.1f}/s "
                            f"Acc: {recent_accuracy:.1%} "
                            f"Range: {actual_item_start}-{actual_item_end}")
                else:
                    actual_item_start = start_idx + batch_start if args.total_jobs > 1 else batch_start
                    actual_item_end = start_idx + batch_end - 1 if args.total_jobs > 1 else batch_end - 1
                    print(f"Completed preference pair batch {batch_start//args.preference_pair_batch_size + 1}. "
                          f"Job: {job_completed}/{job_total} "
                          f"Rate: {eval_rate:.1f}/s "
                          f"Range: {actual_item_start}-{actual_item_end}")
            
            # Save results after each batch
            output_data['evaluation_results'] = evaluation_results
            output_data['metrics'] = calculate_metrics(evaluation_results)
            save_incremental_results(output_data, args.output_file)
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}")
        print(f"Traceback: {traceback.format_exc()}")
    
    # Final save and metrics calculation
    print(f"\n=== Finalizing Results ===")
    output_data['evaluation_results'] = evaluation_results
    output_data['metrics'] = calculate_metrics(evaluation_results)
    output_data['metadata']['status'] = 'completed'
    output_data['metadata']['completed_evaluations'] = len(evaluation_results)
    
    save_incremental_results(output_data, args.output_file)
    
    # Print summary
    metrics = output_data['metrics']
    
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Evaluation type: {evaluation_type}")
    if args.total_jobs > 1:
        actual_items_processed = [start_idx + i for i in range(len(evaluation_results))]
        if actual_items_processed:
            print(f"Job {args.job_index + 1}/{args.total_jobs}: Processed evaluation items {min(actual_items_processed)}-{max(actual_items_processed)}")
    print(f"Total evaluations completed by this job: {metrics['total_evaluations']}")
    
    # Print accuracy by setting
    print(f"\n=== ACCURACY BY SETTING ===")
    for key, value in metrics.items():
        if key.endswith('_accuracy'):
            setting_name = key[:-9]  # Remove '_accuracy'
            success_rate = metrics.get(f'{setting_name}_success_rate', 1.0)
            mean_score_diff = metrics.get(f'{setting_name}_mean_score_diff', 0.0)
            print(f"  {setting_name}: {value:.1%} accuracy ({success_rate:.1%} success rate, avg score diff: {mean_score_diff:.3f})")
    
    # Display randomization balance
    randomization = metrics.get('randomization_balance', {})
    print(f"\n=== RANDOMIZATION BALANCE ===")
    print(f"Preferred completion in position A: {randomization.get('preferred_in_position_a', 0)} ({randomization.get('balance_ratio', 0):.1%})")
    print(f"WVS context first: {randomization.get('wvs_context_first', 0)} ({randomization.get('context_order_balance_ratio', 0):.1%})")
    
    # Performance by style family
    print(f"\n=== PERFORMANCE BY STYLE FAMILY ===")
    for family, family_metrics in metrics.get('by_style_family', {}).items():
        print(f"\n{family} ({family_metrics['count']} evaluations):")
        for key, value in family_metrics.items():
            if key.endswith('_accuracy'):
                setting_name = key[:-9]
                print(f"  {setting_name}: {value:.1%}")
    
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main() 