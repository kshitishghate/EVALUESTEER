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
sys.path.append(str(Path(__file__).parent.parent))

from data_management.data_manager_v3 import DataManagerV3
from evaluation_engine.evaluation_engine_v3 import EvaluationEngineV3

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print("Transformers available")
except ImportError:
    print("Error: transformers not installed. Please install with: pip install transformers")
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


class AblationDataManager(DataManagerV3):
    """Extended data manager for WVS ablation studies"""
    
    def __init__(self, base_path: str = None):
        super().__init__(base_path)
        
        # Filter key_wvs_questions to only include those available in synthetic data
        self.available_key_wvs_questions = self._get_available_key_wvs_questions()
        
    def _get_available_key_wvs_questions(self) -> List[str]:
        """Get only the key WVS questions that are actually available in synthetic data"""
        synthetic_data = self.load_synthetic_data()
        available_question_ids = {item['question_id'] for item in synthetic_data}
        
        available_key_wvs = []
        missing_key_wvs = []
        
        for q_id in self.key_wvs_questions:
            if q_id in available_question_ids:
                available_key_wvs.append(q_id)
            else:
                missing_key_wvs.append(q_id)
        
        if missing_key_wvs:
            print(f"Note: Excluding {len(missing_key_wvs)} key WVS questions not in synthetic data: {missing_key_wvs}")
        
        print(f"Available key WVS questions for ablation: {len(available_key_wvs)} questions")
        return available_key_wvs
        
    def generate_ablation_wvs_context(self, value_profile_id: int, question_id: str, 
                                    num_statements: int = 4, random_seed: int = None) -> str:
        """
        Generate ablated WVS context with limited statements from core WVS questions only.
        
        Args:
            value_profile_id: WVS user ID
            question_id: Current question ID to ensure relevant statement is included
            num_statements: Total number of statements to include (default: 4)
            random_seed: Random seed for reproducible statement selection (NOTE: only used for this specific context generation)
            
        Returns:
            Formatted context string with limited WVS statements from core WVS questions
        """
        # Create a local random state for this context generation only
        # This preserves the global random state for other randomization
        local_random = random.Random(random_seed) if random_seed is not None else random
            
        # Get the value profile
        value_profile = self.get_value_profile_by_id(value_profile_id)
        
        # Load WVS and metadata for context generation
        wvs_df = self.load_wvs_human_data()
        metadata_df = self.load_question_metadata()
        
        # Get user's row in WVS data
        user_rows = wvs_df[wvs_df['D_INTERVIEW'] == value_profile_id]
        if len(user_rows) == 0:
            return f"No WVS context available for user {value_profile_id}."
        
        user_row = user_rows.iloc[0]
        
        # Get statements from available core WVS questions only
        core_wvs_statements = []
        relevant_statement = None
        
        for q_col in self.available_key_wvs_questions:
            user_response = user_row.get(q_col)
            
            # Skip if the user didn't answer this question
            if pd.isna(user_response):
                continue
            
            # Skip if the response is not numeric
            try:
                user_response = int(user_response)
            except (ValueError, TypeError):
                continue
            
            # Skip invalid values like -99
            if user_response < 0:
                continue
                
            # Find the metadata for this question
            q_metadata_rows = metadata_df[metadata_df['question_id'] == q_col]
            
            if len(q_metadata_rows) == 0:
                continue
                
            q_metadata = q_metadata_rows.iloc[0]
            
            # Get the answer_ids_to_grouped_answer_ids mapping
            answer_mapping = q_metadata['answer_ids_to_grouped_answer_ids']
            
            # Find which group this user's response belongs to
            group_id = None
            for aid, gid in answer_mapping.items():
                if int(aid) == user_response:
                    group_id = gid
                    break
            
            if group_id is not None:
                # Get the statement that corresponds to this group
                converted_statements = q_metadata['converted_statements']
                if group_id < len(converted_statements):
                    statement = converted_statements[group_id]
                    
                    # Check if this is the relevant statement for the current question
                    if q_col == question_id:
                        relevant_statement = statement
                    else:
                        core_wvs_statements.append(statement)
        
        # Select statements for ablation context
        selected_statements = []
        
        # 1. Always include the relevant statement if available
        if relevant_statement:
            selected_statements.append(f"- {relevant_statement}")
            remaining_slots = num_statements - 1
        else:
            # No relevant statement found (expected for WVQ questions)
            remaining_slots = num_statements
        
        # 2. Randomly select additional statements from core WVS questions only  
        if core_wvs_statements and remaining_slots > 0:
            # Ensure we don't select more than available
            n_to_select = min(remaining_slots, len(core_wvs_statements))
            selected_additional = local_random.sample(core_wvs_statements, n_to_select)
            
            for stmt in selected_additional:
                selected_statements.append(f"- {stmt}")
        
        if not selected_statements:
            return f"No core WVS statements available for user {value_profile_id} (ablation)."
        
        # Randomize order to avoid position bias
        if len(selected_statements) > 1:
            local_random.shuffle(selected_statements)
        
        context = (f"Based on this user's World Values Survey responses, the following statements "
                  f"describe their core values and beliefs (ablated to {len(selected_statements)} key statements from core WVS questions):\n\n" + 
                  "\n".join(selected_statements))
        
        return context


class AblationPreferenceFilter:
    """Filter preference pairs to identify value-style conflicts for ablation study"""
    
    def __init__(self):
        pass
    
    def parse_user_style_preferences(self, style_code: str) -> Dict[str, str]:
        """Parse user's style_code to extract their preferences"""
        if not style_code:
            return {}
        
        parts = style_code.split('_')
        
        if len(parts) != 4:
            return {}
        
        # Map abbreviated codes to full names
        verbosity_map = {'v': 'verbose', 'c': 'concise'}
        readability_map = {'hrd': 'high_reading_difficulty', 'lrd': 'low_reading_difficulty'}
        confidence_map = {'hc': 'high_confidence', 'lc': 'low_confidence'}
        sentiment_map = {'w': 'warm', 'co': 'cold'}
        
        style_prefs = {}
        
        # Parse each component
        if parts[0] in verbosity_map:
            style_prefs['verbosity'] = verbosity_map[parts[0]]
        
        if parts[1] in readability_map:
            style_prefs['readability'] = readability_map[parts[1]]
        
        if parts[2] in confidence_map:
            style_prefs['confidence'] = confidence_map[parts[2]]
        
        if parts[3] in sentiment_map:
            style_prefs['sentiment'] = sentiment_map[parts[3]]
        
        return style_prefs
    
    def has_style_differences(self, combination_type: str) -> bool:
        """Check if the combination type involves different styles"""
        if '_vs_' not in combination_type:
            return False
        
        left_part, right_part = combination_type.split('_vs_')
        
        # Extract style information from each side
        left_style = self.extract_style_from_combination_part(left_part)
        right_style = self.extract_style_from_combination_part(right_part)
        
        # Check if any style dimension differs
        return left_style != right_style
    
    def extract_style_from_combination_part(self, combination_part: str) -> str:
        """Extract style information from combination part like 'A_verbose' or 'B_high_confidence'"""
        # Remove A_ or B_ prefix
        if combination_part.startswith(('A_', 'B_')):
            style_part = combination_part[2:]
        else:
            style_part = combination_part
        
        return style_part
    
    def user_style_in_nonpreferred_completion(self, preferred_key: str, non_preferred_key: str, 
                                           user_style_prefs: Dict[str, str]) -> bool:
        """
        Check if user's preferred style appears in the non-preferred completion.
        This creates a value-style conflict where the model must choose between them.
        """
        if not preferred_key or not non_preferred_key:
            return False
        
        # Extract style from non-preferred completion key
        try:
            non_preferred_parts = non_preferred_key.split('_')
            if len(non_preferred_parts) < 2:
                return False
            
            # Skip "A" or "B", get the style part
            non_preferred_style = '_'.join(non_preferred_parts[1:])
            
        except (IndexError, AttributeError):
            return False
        
        # Check if any of the user's preferred styles appear in the non-preferred completion
        for style_family, user_pref in user_style_prefs.items():
            if user_pref in non_preferred_style:
                return True
        
        return False
    
    def identify_value_style_conflicts(self, preference_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify preference pairs where value and style preferences conflict.
        
        Args:
            preference_pairs: List of all preference pairs
            
        Returns:
            List of preference pairs with value-style conflicts
        """
        conflict_cases = []
        total_cases = 0
        cases_with_style_prefs = 0
        cases_with_keys = 0
        cases_with_style_diff = 0
        wvs_based_cases = 0
        
        for pair in preference_pairs:
            total_cases += 1
            
            # Only consider WVS-based preferences (where values differ)
            if pair.get('preference_rule') != 'wvs_based':
                continue
            wvs_based_cases += 1
            
            # Parse user's style preferences
            user_style_prefs = self.parse_user_style_preferences(pair.get('style_code', ''))
            if not user_style_prefs:
                continue
            cases_with_style_prefs += 1
            
            # Get actual completion keys
            preferred_key = pair.get('preferred_completion_key', '')
            non_preferred_key = pair.get('non_preferred_completion_key', '')
            
            if not preferred_key or not non_preferred_key:
                continue
            cases_with_keys += 1
                
            # Parse combination type to ensure styles differ
            combination_type = pair.get('combination_type', '')
            if not self.has_style_differences(combination_type):
                continue
            cases_with_style_diff += 1
            
            # Check if user's preferred style appears in non-preferred completion
            if self.user_style_in_nonpreferred_completion(
                preferred_key, non_preferred_key, user_style_prefs):
                conflict_cases.append(pair)
        
        # Debug output
        print("Value-Style Conflict Filtering:")
        print(f"   Total preference pairs examined: {total_cases}")
        print(f"   WVS-based cases: {wvs_based_cases}")
        print(f"   Cases with style preferences: {cases_with_style_prefs}")
        print(f"   Cases with completion keys: {cases_with_keys}")
        print(f"   Cases with style differences: {cases_with_style_diff}")
        print(f"   Final conflict cases identified: {len(conflict_cases)}")
        
        return conflict_cases


# ===== PROMPT GENERATION FUNCTIONS FOR ABLATION =====

def create_ablation_wvs_only_prompt(prompt, completion_a, completion_b, ablation_wvs_context):
    """Create a prompt with ablated WVS context only"""
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

User's Values and Beliefs:
{ablation_wvs_context}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and beliefs, which response (A or B) would this specific user prefer?"""

def create_ablation_wvs_style_neutral_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first):
    """Create a prompt with ablated WVS and style context, no preference guidance"""
    if wvs_first:
        context_section = f"""User's Values and Beliefs:
{ablation_wvs_context}

User's Style Preferences:
{style_context}"""
    else:
        context_section = f"""User's Style Preferences:
{style_context}

User's Values and Beliefs:
{ablation_wvs_context}"""
    
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer?"""

def create_ablation_wvs_style_prefer_wvs_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first):
    """Create a prompt with ablated WVS and style context, emphasizing values take precedence"""
    if wvs_first:
        context_section = f"""User's Values and Beliefs:
{ablation_wvs_context}

User's Style Preferences:
{style_context}"""
    else:
        context_section = f"""User's Style Preferences:
{style_context}

User's Values and Beliefs:
{ablation_wvs_context}"""
    
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer? IMPORTANT: Aligning with the user's values and beliefs takes the highest precedence. Style preferences are secondary to value alignment."""

def create_ablation_wvs_style_prefer_style_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first):
    """Create a prompt with ablated WVS and style context, emphasizing style preferences over values"""
    if wvs_first:
        context_section = f"""User's Values and Beliefs:
{ablation_wvs_context}

User's Style Preferences:
{style_context}"""
    else:
        context_section = f"""User's Style Preferences:
{style_context}

User's Values and Beliefs:
{ablation_wvs_context}"""
    
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer? IMPORTANT: This user prioritizes communication style and format over content alignment with their values. Style preferences take the highest precedence."""


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


# ===== CROSS-PREFERENCE-PAIR BATCHING OPTIMIZATION FOR ABLATION =====

def collect_conversations_from_preference_pairs_ablation(preference_pairs, ablation_data_manager, args, user_profile_cache, 
                                                        ablation_conflict_sets=None):
    """
    Collect all conversations from a batch of preference pairs for ablation evaluation.
    
    This function intelligently routes different preference pair sets to different ablation settings:
    - WVS-only settings use preference pairs from 18-profile set
    - Combined settings use preference pairs from 288-profile set
    
    Args:
        preference_pairs: List of preference pair dictionaries (used as fallback)
        ablation_data_manager: AblationDataManager instance
        args: Command line arguments  
        user_profile_cache: Cache for user profiles
        ablation_conflict_sets: Dict with 'wvs_only' and 'combined' preference pair sets
    
    Returns:
        conversations: List of conversation objects ready for model inference
        conversation_metadata: List of metadata for mapping responses back to preference pairs
    """
    conversations = []
    conversation_metadata = []
    
    # Define ablation settings
    ablation_settings = [
        'ablation_wvs_only',
        'ablation_wvs_style_neutral',
        'ablation_wvs_style_prefer_wvs',
        'ablation_wvs_style_prefer_style'
    ]
    
    # Generate conversations for each setting using only the preference pairs in this batch
    for setting_name in ablation_settings:
        print(f"Generating conversations for {setting_name}: {len(preference_pairs)} preference pairs")
        
        for preference_pair in preference_pairs:
            # Check if this preference pair should be used for this setting
            if ablation_conflict_sets is not None:
                # Route settings to appropriate preference pair sets
                if setting_name in ['ablation_wvs_only']:
                    # Only process if this pair is in the WVS-only conflict set
                    if preference_pair not in ablation_conflict_sets['wvs_only']:
                        continue
                elif setting_name in ['ablation_wvs_style_neutral', 'ablation_wvs_style_prefer_wvs', 
                                      'ablation_wvs_style_prefer_style']:
                    # Only process if this pair is in the combined conflict set
                    if preference_pair not in ablation_conflict_sets['combined']:
                        continue
            
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
                user_profile_cache[user_profile_id] = ablation_data_manager.get_user_profile_by_id(user_profile_id)
            user_profile = user_profile_cache[user_profile_id]
            
            # Generate contexts
            # ABLATION: Use reduced WVS context with only 4 statements
            ablation_wvs_context = ablation_data_manager.generate_ablation_wvs_context(
                value_profile_id, question_id, num_statements=getattr(args, 'ablation_statements', 4), 
                random_seed=args.random_seed
            )
            style_context = ablation_data_manager.generate_full_style_context(user_profile['style_profile'])
            
            # For combined contexts, randomize the order of WVS and style contexts
            wvs_first = random.choice([True, False])
            
            # Store metadata for this preference pair and setting
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
                'setting': setting_name
            }
            
            # Generate the appropriate prompt for this setting
            if setting_name == 'ablation_wvs_only':
                prompt_text = create_ablation_wvs_only_prompt(prompt, completion_a, completion_b, ablation_wvs_context)
            elif setting_name == 'ablation_wvs_style_neutral':
                prompt_text = create_ablation_wvs_style_neutral_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
            elif setting_name == 'ablation_wvs_style_prefer_wvs':
                prompt_text = create_ablation_wvs_style_prefer_wvs_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
            elif setting_name == 'ablation_wvs_style_prefer_style':
                prompt_text = create_ablation_wvs_style_prefer_style_prompt(prompt, completion_a, completion_b, ablation_wvs_context, style_context, wvs_first)
            else:
                # Default fallback
                prompt_text = create_ablation_wvs_only_prompt(prompt, completion_a, completion_b, ablation_wvs_context)
            
            # Create conversations for both completions
            conv_a = create_conversation_for_scoring(prompt_text, completion_a)
            conv_b = create_conversation_for_scoring(prompt_text, completion_b)
            
            conversations.extend([conv_a, conv_b])
            conversation_metadata.extend([
                {**base_metadata, 'position': 'A'},
                {**base_metadata, 'position': 'B'}
            ])
    
    return conversations, conversation_metadata


def evaluate_preference_pairs_cross_batch_optimized_ablation(preference_pairs, ablation_data_manager, model, tokenizer, device, args, 
                                                          ablation_conflict_sets=None):
    """
    Evaluate multiple preference pairs together using cross-preference-pair batching for ablation.
    
    Args:
        preference_pairs: List of preference pair dictionaries
        ablation_data_manager: AblationDataManager instance
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
    conversations, conversation_metadata = collect_conversations_from_preference_pairs_ablation(
        preference_pairs, ablation_data_manager, args, user_profile_cache, ablation_conflict_sets)
    
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
            # We need to regenerate the ablation context to get metadata
            preference_pair = next((p for p in preference_pairs if p.get('user_profile_id') == metadata['user_profile_id'] and p.get('question_id') == metadata['question_id']), None)
            ablation_wvs_context = ""
            ablation_context_length = 0
            ablation_statements_count = 0
            
            if preference_pair:
                try:
                    ablation_wvs_context = ablation_data_manager.generate_ablation_wvs_context(
                        metadata['value_profile_id'], metadata['question_id'], 
                        num_statements=getattr(args, 'ablation_statements', 4), 
                        random_seed=args.random_seed
                    )
                    ablation_context_length = len(ablation_wvs_context)
                    ablation_statements_count = ablation_wvs_context.count('- ') if ablation_wvs_context else 0
                except:
                    pass  # Use defaults if context generation fails
        
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
                'cross_batch_optimized': True,
                'ablation_wvs_context': ablation_wvs_context,
                'ablation_context_length': ablation_context_length,
                'ablation_statements_count': ablation_statements_count
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
        for setting in ['ablation_wvs_only', 'ablation_wvs_style_neutral', 'ablation_wvs_style_prefer_wvs', 'ablation_wvs_style_prefer_style']:
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
    
    # Determine which settings are present (ablation settings only)
    available_settings = []
    if results:
        sample_result = results[0]
        for key in sample_result.keys():
            if key.endswith('_response') and key.startswith('ablation_'):
                setting_name = key[:-9]  # Remove '_response'
                available_settings.append(setting_name)
    
    # Calculate overall metrics
    overall_metrics = {
        'total_evaluations': total_count,
        'conflict_cases_only': True,
        'ablation_context_avg_length': np.mean([r.get('ablation_context_length', 0) for r in results]),
        'ablation_statements_avg_count': np.mean([r.get('ablation_statements_count', 0) for r in results]),
        'randomization_balance': {
            'preferred_in_position_a': sum(1 for r in results if r.get('preferred_in_position_a', False)),
            'preferred_in_position_b': total_count - sum(1 for r in results if r.get('preferred_in_position_a', False)),
            'balance_ratio': sum(1 for r in results if r.get('preferred_in_position_a', False)) / total_count if total_count > 0 else 0,
            'wvs_context_first': sum(1 for r in results if r.get('wvs_context_first', False)),
            'style_context_first': total_count - sum(1 for r in results if r.get('wvs_context_first', False)),
            'context_order_balance_ratio': sum(1 for r in results if r.get('wvs_context_first', False)) / total_count if total_count > 0 else 0
        }
    }
    
    # Calculate accuracy and score statistics for each setting
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
    
    return overall_metrics


# ===== MAIN FUNCTION =====

def main():
    # Check if transformers is available
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        print("Error: transformers is not installed. Please install it with: pip install transformers")
        return
    
    parser = argparse.ArgumentParser(description="Sequence Classifier WVS Ablation Evaluation for Reward Model Evaluation v3")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the sequence classifier model (e.g., Skywork/Skywork-Reward-V2-Llama-3.1-8B)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device for inference ('auto', 'cuda', 'cpu')")
    parser.add_argument("--torch_dtype", type=str, default="auto",
                       help="Torch dtype for model loading ('auto', 'float16', 'bfloat16', 'float32')")
    
    # Single GPU optimization arguments
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for inference")
    parser.add_argument("--preference_pair_batch_size", type=int, default=5,
                       help="Number of preference pairs to process in parallel")
    parser.add_argument("--max_length", type=int, default=4096,
                       help="Maximum sequence length for batched processing")
    parser.add_argument("--enable_cross_batch_optimization", action="store_true", default=True,
                       help="Enable cross-preference-pair batching for maximum throughput (default: True)")
    parser.add_argument("--disable_cross_batch_optimization", dest="enable_cross_batch_optimization", action="store_false",
                       help="Disable cross-batch optimization and use individual processing (for debugging)")
    
    # Question group selection (adapted for ablation - only core WVS questions)
    parser.add_argument("--questions", type=str, default="core_wvs_only",
                       choices=['core_wvs_only', 'available_key_wvs'],
                       help="Which questions to use: core_wvs_only (available core WVS only) or available_key_wvs (all available key WVS)")
    
    # Ablation-specific arguments
    parser.add_argument("--ablation_statements", type=int, default=4,
                       help="Number of WVS statements to include in ablated context")
    
    # Evaluation mode arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--all_settings", action="store_true",
                           help="Run all ablation settings (4 settings, no COT for sequence classifiers)")
    mode_group.add_argument("--all_settings_no_cot", action="store_true",
                           help="Run all ablation settings without COT (same as --all_settings for sequence classifiers)")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for results (auto-generated if not provided)")
    parser.add_argument("--max_evaluations", type=int, default=None,
                       help="Maximum number of evaluations to run (for testing)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducible randomization")
    parser.add_argument("--save_interval", type=int, default=50,
                       help="Save results every N evaluations")
    
    # Parallel job arguments
    parser.add_argument("--job_index", type=int, default=0,
                       help="Index of this parallel job (0-based)")
    parser.add_argument("--total_jobs", type=int, default=1,
                       help="Total number of parallel jobs")
    
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
        args.output_file = f"{model_name}_sequence_classifier_wvs_ablation_conflict_only_results.json"
    
    print("=== Sequence Classifier WVS Ablation Evaluation for Value-Style Conflicts ===")
    print(f"Model: {args.model_path}")
    print(f"Questions: {args.questions}")
    print(f"Output: {args.output_file}")
    print(f"Ablation context: {args.ablation_statements} WVS statements")
    print(f"Evaluation focus: Value-style conflict cases only")
    print(f"Device: {device}")
    print(f"Torch dtype: {torch_dtype}")
    print(f"Batch size: {args.batch_size}")
    print(f"Preference pair batch size: {args.preference_pair_batch_size}")
    print(f"Cross-batch optimization: {'ENABLED' if args.enable_cross_batch_optimization else 'DISABLED'}")
    if args.total_jobs > 1:
        print(f"Parallel job: {args.job_index + 1}/{args.total_jobs}")
    
    # Initialize data manager and evaluation engine
    print(f"\n=== Loading Data ===")
    ablation_data_manager = AblationDataManager()
    evaluation_engine = EvaluationEngineV3(ablation_data_manager)
    preference_filter = AblationPreferenceFilter()
    
    # Generate preference pairs using ablation-optimized approach
    print(f"\n=== Generating Ablation-Optimized Preference Pairs ===")
    question_ids = ablation_data_manager.available_key_wvs_questions  # Use only available key WVS questions for ablation
    print(f"Using available key WVS questions for ablation: {len(question_ids)} questions")
    
    # Determine ablation mode based on args
    ablation_mode = 'all_settings' if args.all_settings else 'all_settings_no_cot'
    
    # Generate preference pairs optimized for ablation studies
    ablation_preference_sets = evaluation_engine.generate_ablation_preference_pairs(
        question_ids=question_ids,
        ablation_mode=ablation_mode
    )
    
    # Combine all preference pairs for conflict filtering
    # Note: WVS-only and combined pairs will be used for different evaluation settings
    wvs_only_pairs = ablation_preference_sets['wvs_only']
    combined_pairs = ablation_preference_sets['combined']
    all_preference_pairs = wvs_only_pairs + combined_pairs
    print(f"Generated {len(all_preference_pairs)} total preference pairs ({len(wvs_only_pairs)} WVS-only + {len(combined_pairs)} combined)")
    
    # Filter to only value-style conflict cases for each set
    print(f"\n=== Filtering for Value-Style Conflicts ===")
    
    # Filter WVS-only pairs for conflicts (these will be used for ablation_wvs_only settings)
    wvs_only_conflicts = preference_filter.identify_value_style_conflicts(wvs_only_pairs)
    print(f"WVS-only conflict cases: {len(wvs_only_conflicts)}")
    
    # Filter combined pairs for conflicts (these will be used for ablation_wvs_style_* settings)
    combined_conflicts = preference_filter.identify_value_style_conflicts(combined_pairs)
    print(f"Combined conflict cases: {len(combined_conflicts)}")
    
    # Store both sets for different evaluation settings
    ablation_conflict_sets = {
        'wvs_only': wvs_only_conflicts,
        'combined': combined_conflicts
    }
    
    # For backward compatibility, combine all conflicts
    conflict_preference_pairs = wvs_only_conflicts + combined_conflicts
    
    if not conflict_preference_pairs:
        print("No value-style conflict cases found! Check the filtering logic.")
        return
    
    print(f"Identified {len(conflict_preference_pairs)} conflict cases for ablation evaluation")
    
    # Limit evaluations if specified
    if args.max_evaluations and args.max_evaluations < len(conflict_preference_pairs):
        conflict_preference_pairs = conflict_preference_pairs[:args.max_evaluations]
        print(f"Limited to {len(conflict_preference_pairs)} evaluations for testing")
    
    # Check for existing results
    existing_results = load_existing_results(args.output_file)
    if existing_results:
        print(f"Found existing results with {len(existing_results.get('evaluation_results', []))} evaluations")
        completed_ids = {r['preference_id'] for r in existing_results.get('evaluation_results', [])}
        remaining_pairs = [p for p in conflict_preference_pairs if f"{p['user_profile_id']}_{p['question_id']}_{p['style_family']}_{p['combination_type']}" not in completed_ids]
        print(f"Remaining evaluations after filtering completed: {len(remaining_pairs)}")
        conflict_preference_pairs = remaining_pairs
        
        if not conflict_preference_pairs:
            print("All ablation evaluations already completed!")
            return
    
    # Split preference pairs for parallel job processing
    if args.total_jobs > 1:
        total_before_split = len(conflict_preference_pairs)
        conflict_preference_pairs, start_idx, end_idx = split_preference_pairs_for_job(conflict_preference_pairs, args.job_index, args.total_jobs)
        print(f"Job {args.job_index + 1}/{args.total_jobs}: Processing evaluations {start_idx}-{end_idx-1} ({len(conflict_preference_pairs)} items) of {total_before_split} conflict cases")
        
        if not conflict_preference_pairs:
            print(f"No evaluations assigned to job {args.job_index + 1}/{args.total_jobs}")
            return
    else:
        start_idx, end_idx = 0, len(conflict_preference_pairs)
        print(f"Single job: Processing {len(conflict_preference_pairs)} conflict cases (items 0-{len(conflict_preference_pairs)-1})")
    
    # Initialize sequence classifier model
    print(f"\n=== Initializing Sequence Classifier Model ===")
    try:
        print(f"Loading model: {args.model_path}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        
        print(f"Loading tokenizer: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        model.eval()  # Set to evaluation mode
        
        print("Model and tokenizer loaded successfully")
        print(f"Using batch size: {args.batch_size}")
        print(f"Max sequence length: {args.max_length}")
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        traceback.print_exc()
        return
    
    # Initialize results storage
    evaluation_results = existing_results.get('evaluation_results', []) if existing_results else []
    
    # Track this job's initial contribution to avoid double-counting progress
    initial_results_count = len(evaluation_results)
    job_target_count = len(conflict_preference_pairs)
    
    # Create output data structure
    output_data = {
        'metadata': {
            'model_path': args.model_path,
            'evaluation_type': 'sequence_classifier_wvs_ablation_conflict_only',
            'ablation_statements_count': args.ablation_statements,
            'total_conflict_cases': len(conflict_preference_pairs) + len(evaluation_results),
            'device': str(device),
            'torch_dtype': str(torch_dtype),
            'random_seed': args.random_seed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'in_progress',
            'evaluation_settings': [
                'ablation_wvs_only',
                'ablation_wvs_style_neutral',
                'ablation_wvs_style_prefer_wvs',
                'ablation_wvs_style_prefer_style'
            ],
            'parallel_job_info': {
                'job_index': args.job_index,
                'total_jobs': args.total_jobs,
                'job_preference_count': len(conflict_preference_pairs)
            } if args.total_jobs > 1 else None
        },
        'evaluation_results': evaluation_results,
        'metrics': {}
    }
    
    # Calculate estimated time
    print(f"\n=== Evaluation Time Estimation ===")
    total_evaluations = len(conflict_preference_pairs)
    print(f"Total conflict evaluations: {total_evaluations:,}")
    
    if args.enable_cross_batch_optimization:
        # Count total conversations per batch based on ablation settings
        conversations_per_pair = 8  # 4 ablation settings × 2 positions
        
        total_batches = (total_evaluations + args.preference_pair_batch_size - 1) // args.preference_pair_batch_size
        avg_conversations_per_model_batch = conversations_per_pair * args.preference_pair_batch_size
        print(f"Conversations per preference pair: {conversations_per_pair}")
        print(f"Average conversations per preference pair batch: {avg_conversations_per_model_batch}")
        print(f"Total vLLM calls needed: ~{total_batches} (vs {total_evaluations * conversations_per_pair} without batching)")
        print(f"Estimated throughput improvement: {conversations_per_pair * args.preference_pair_batch_size:.0f}x per model call")
    else:
        print("Note: Cross-batch optimization disabled - using individual processing (slower)")
    
    # Run evaluations
    print(f"\n=== Running Ablation Evaluations ===")
    print(f"Processing {len(conflict_preference_pairs)} conflict cases in batches of {args.preference_pair_batch_size}")
    
    try:
        start_time = time.time()
        
        # Process preference pairs in batches with optimized cross-batch processing
        total_items = len(conflict_preference_pairs)
        
        for batch_start in range(0, total_items, args.preference_pair_batch_size):
            batch_end = min(batch_start + args.preference_pair_batch_size, total_items)
            batch_pairs = conflict_preference_pairs[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//args.preference_pair_batch_size + 1}/{(total_items-1)//args.preference_pair_batch_size + 1} "
                  f"(items {batch_start+1}-{batch_end}/{total_items})")
            
            try:
                # Use cross-batch optimization if enabled
                if args.enable_cross_batch_optimization:
                    batch_results = evaluate_preference_pairs_cross_batch_optimized_ablation(
                        batch_pairs, ablation_data_manager, model, tokenizer, device, args, ablation_conflict_sets
                    )
                    evaluation_results.extend(batch_results)
                else:
                    # Process individually when batching is disabled (fallback implementation needed)
                    print("Individual processing fallback not implemented for sequence classifier ablation")
                    raise NotImplementedError("Individual processing not implemented")
                
            except Exception as e:
                print(f"Error in batch evaluation: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                continue
            
            # Calculate performance stats
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Track THIS JOB's progress only (not the shared file total)
            job_completed = len(evaluation_results) - initial_results_count
            job_total = job_target_count
            
            if elapsed_time > 0:
                eval_rate = job_completed / elapsed_time
                
                # Calculate recent accuracy (for ablation_wvs_only setting if available)
                if evaluation_results:
                    recent_results = evaluation_results[-min(20, len(evaluation_results)):]  # Last 20 results
                    wvs_only_correct = sum(1 for r in recent_results if r.get('ablation_wvs_only_correct', False))
                    recent_accuracy = wvs_only_correct / len(recent_results) if recent_results else 0
                    
                    actual_item_start = start_idx + batch_start if args.total_jobs > 1 else batch_start
                    actual_item_end = start_idx + batch_end - 1 if args.total_jobs > 1 else batch_end - 1
                    print(f"Completed batch {batch_start//args.preference_pair_batch_size + 1}. "
                          f"Job: {job_completed}/{job_total} "
                          f"Rate: {eval_rate:.1f}/s "
                          f"Acc: {recent_accuracy:.1%} "
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
    
    print(f"\n=== SEQUENCE CLASSIFIER ABLATION EVALUATION SUMMARY ===")
    print(f"Evaluation type: Sequence Classifier WVS Ablation (Value-Style Conflicts Only)")
    print(f"Total conflict cases evaluated: {metrics['total_evaluations']}")
    print(f"Average ablation context length: {metrics['ablation_context_avg_length']:.0f} chars")
    print(f"Average ablation statements: {metrics['ablation_statements_avg_count']:.1f}")
    
    # Print accuracy by setting
    print(f"\n=== ACCURACY BY ABLATION SETTING ===")
    for key, value in metrics.items():
        if key.startswith('ablation_') and key.endswith('_accuracy'):
            setting_name = key[:-9]  # Remove '_accuracy'
            success_rate = metrics.get(f'{setting_name}_success_rate', 1.0)
            mean_score_diff = metrics.get(f'{setting_name}_mean_score_diff', 0.0)
            print(f"  {setting_name}: {value:.1%} accuracy ({success_rate:.1%} success rate, avg score diff: {mean_score_diff:.3f})")
    
    # Display randomization balance
    randomization = metrics.get('randomization_balance', {})
    print(f"\n=== RANDOMIZATION BALANCE ===")
    print(f"Preferred completion in position A: {randomization.get('preferred_in_position_a', 0)} ({randomization.get('balance_ratio', 0):.1%})")
    print(f"WVS context first: {randomization.get('wvs_context_first', 0)} ({randomization.get('context_order_balance_ratio', 0):.1%})")
    
    print("\nSequence classifier ablation evaluation completed!")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
