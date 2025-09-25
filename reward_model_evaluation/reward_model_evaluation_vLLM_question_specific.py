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

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_management.data_manager_v3 import DataManagerV3
from evaluation_engine.evaluation_engine_v3 import EvaluationEngineV3

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("Warning: vLLM not installed. Please install with: pip install vllm")
    LLM = None
    SamplingParams = None

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: transformers not installed. Please install with: pip install transformers")
    AutoTokenizer = None

# Pydantic for structured JSON output
try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:
    print("Warning: Pydantic not installed. Please install with: pip install pydantic")
    BaseModel = None
    Field = None
    ValidationError = None

# COT Response schema
if BaseModel is not None:
    class COTResponse(BaseModel):
        """Structured response schema for both COT and non-COT evaluations"""
        reasoning: Optional[str] = Field(default=None, description="Step-by-step analysis of user preferences and response alignment (COT only)")
        final_choice: Literal["A", "B"] = Field(description="The final answer: either A or B")
        confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Optional confidence score between 0 and 1")


# ===== PROMPT GENERATION FUNCTIONS (adapted from v2) =====

def create_simple_prompt(prompt, completion_a, completion_b):
    """Create a simple prompt without any context"""
    return f"""You are an AI assistant that helps evaluate which response better answers a given question.

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Which response (A or B) better answers the question? Consider factors like relevance, helpfulness, accuracy, and overall quality.

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

def create_full_wvs_only_prompt(prompt, completion_a, completion_b, wvs_context):
    """Create a prompt with full WVS context only (no style context)"""
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

User's Values and Beliefs:
{wvs_context}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and beliefs, which response (A or B) would this specific user prefer?

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

def create_full_style_only_prompt(prompt, completion_a, completion_b, style_context):
    """Create a prompt with full style context only (no WVS context)"""
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

User's Style Preferences:
{style_context}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's style preferences, which response (A or B) would this specific user prefer?

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

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

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer? IMPORTANT: Aligning with the user's values and beliefs takes the highest precedence. Style preferences are secondary to value alignment.

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

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

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer? IMPORTANT: This user prioritizes communication style and format over content alignment with their values. Style preferences take the highest precedence.

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

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

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer?

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

# ===== COT PROMPT FUNCTIONS =====

def create_full_wvs_only_cot_prompt(prompt, completion_a, completion_b, wvs_context):
    """Create a COT prompt with full WVS context only - returns JSON"""
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

User's Values and Beliefs:
{wvs_context}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Please analyze step-by-step:

1. First, analyze the user's values and beliefs to identify which aspects are most relevant to this question and the given responses.

2. Consider how each response aligns with those relevant aspects of the user's values and beliefs.

3. Based on your analysis, determine which response this specific user would prefer given their values.

Respond with valid JSON in exactly this format:
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

def create_full_style_only_cot_prompt(prompt, completion_a, completion_b, style_context):
    """Create a COT prompt with full style context only - returns JSON"""
    return f"""You are an AI assistant that helps evaluate which response better answers a given question for a specific user.

User's Style Preferences:
{style_context}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Please analyze step-by-step:

1. First, analyze the user's style preferences to identify which aspects are most relevant to this question and the given responses.

2. Consider how each response aligns with those relevant aspects of the user's style preferences.

3. Based on your analysis, determine which response this specific user would prefer given their style preferences.

Respond with valid JSON in exactly this format:
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

def create_full_wvs_style_cot_prefer_wvs_prompt(prompt, completion_a, completion_b, wvs_context, style_context, wvs_first=True):
    """Create a COT prompt with full WVS and style context, emphasizing values take precedence - returns JSON"""
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

Please analyze step-by-step:

1. First, analyze the user's profile to identify which aspects of their values and style preferences are most relevant to this question and the given responses.

2. Consider how each response aligns with those relevant aspects, prioritizing alignment with the user's values and beliefs over style preferences.

3. Based on your analysis, determine which response this specific user would prefer. Remember that aligning with the user's values and beliefs takes the highest precedence, with style preferences being secondary.

Respond with valid JSON in exactly this format:
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

def create_full_wvs_style_cot_prefer_style_prompt(prompt, completion_a, completion_b, wvs_context, style_context, wvs_first=True):
    """Create a COT prompt with full WVS and style context, emphasizing style preferences take precedence - returns JSON"""
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

Please analyze step-by-step:

1. First, analyze the user's profile to identify which aspects of their values and style preferences are most relevant to this question and the given responses.

2. Consider how each response aligns with those relevant aspects, prioritizing alignment with the user's style preferences over their values and beliefs.

3. Based on your analysis, determine which response this specific user would prefer. Remember that this user prioritizes communication style and format over content alignment with their values, so style preferences take the highest precedence.

Respond with valid JSON in exactly this format:
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

def create_full_wvs_style_cot_neutral_prompt(prompt, completion_a, completion_b, wvs_context, style_context, wvs_first=True):
    """Create a COT prompt with full WVS and style context, no preference guidance - returns JSON"""
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

Please analyze step-by-step:

1. First, analyze the user's profile to identify which aspects of their values and style preferences are most relevant to this question and the given responses.

2. Consider how each response aligns with those relevant aspects of the user's profile.

3. Based on your analysis, determine which response this specific user would prefer given their overall profile.

Respond with valid JSON in exactly this format:
{{
    "reasoning": "Your detailed step-by-step analysis here",
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""

# ===== JSON PARSING FOR COT =====

def _extract_reasoning_from_text(response_text: str) -> str:
    """
    Helper function to extract reasoning from response text when JSON parsing fails.
    Uses regex to find reasoning content.
    """
    try:
        import re
        # Look for reasoning patterns in the text
        reasoning_patterns = [
            r'"reasoning"\s*:\s*"([^"]+)"',
            r'"reasoning"\s*:\s*\'([^\']+)\'',
            r'reasoning["\']?\s*:\s*["\']([^"\']+)["\']',
            r'analysis["\']?\s*:\s*["\']([^"\']+)["\']',
            r'step-by-step["\']?\s*:\s*["\']([^"\']+)["\']'
        ]
        
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if matches:
                reasoning = matches[-1].strip()  # Take the last match
                # Clean up the reasoning text
                reasoning = reasoning.replace('\\n', '\n').replace('\\"', '"')
                return reasoning
        
        # If no structured reasoning found, try to extract the first substantial paragraph
        # that looks like reasoning
        lines = response_text.split('\n')
        reasoning_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 20 and ('analyze' in line.lower() or 'consider' in line.lower() or 
                                   'response' in line.lower() or 'prefer' in line.lower() or
                                   'user' in line.lower() or 'values' in line.lower()):
                reasoning_lines.append(line)
        
        if reasoning_lines:
            return ' '.join(reasoning_lines[:3])  # Take first 3 relevant lines
        
        # Fallback: return first non-empty substantial line
        for line in lines:
            line = line.strip()
            if len(line) > 20 and not line.startswith('{') and not line.startswith('}'):
                return line
        
        return "No reasoning extracted"
    except Exception as e:
        print(f"Error extracting reasoning: {e}")
        return "Reasoning extraction failed"


def parse_json_response(response_text: str) -> dict:
    """
    Parse structured JSON response from COT evaluation with robust error handling.
    Returns a dict with 'final_choice' and 'reasoning', or {'final_choice': 'ERROR', 'reasoning': None} if parsing fails.
    """
    if not response_text or not response_text.strip():
        print("Empty response")
        return {"final_choice": "ERROR", "reasoning": None}
    
    # Clean the response - remove any markdown formatting
    cleaned_response = response_text.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()
    
    # First attempt: Clean control characters and fix common JSON issues
    try:
        # Replace problematic control characters
        import re
        # Remove or replace control characters (except newlines, tabs, carriage returns)
        cleaned_response = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', cleaned_response)
        
        # Fix common JSON formatting issues
        # 1. Replace unescaped quotes within strings (this is a heuristic approach)
        # Look for patterns like: "text with "quotes" inside"
        # This is complex, so we'll try a simpler approach first
        
        # 2. Ensure proper newline handling within JSON strings
        # Replace literal newlines within JSON string values with escaped newlines
        lines = cleaned_response.split('\n')
        in_string = False
        escape_next = False
        result_lines = []
        current_line = ""
        
        for line in lines:
            i = 0
            while i < len(line):
                char = line[i]
                
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                
                i += 1
            
            # If we're not in a string or this looks like the end of a value, add as separate line
            if not in_string or line.strip().endswith('",') or line.strip().endswith('"'):
                current_line += line
                result_lines.append(current_line)
                current_line = ""
                in_string = False  # Reset for safety
            else:
                # We're in a string that continues to next line - escape the newline
                current_line += line + "\\n"
        
        # Add any remaining content
        if current_line:
            result_lines.append(current_line)
        
        cleaned_response = '\n'.join(result_lines)
        
        print(f"Attempting to parse cleaned JSON: {cleaned_response[:200]}...")
        
    except Exception as e:
        print(f"Error in JSON cleaning: {e}")
        # Continue with original response if cleaning fails
    
    # Second attempt: Try parsing the cleaned JSON
    try:
        response_data = json.loads(cleaned_response)
        print("Successfully parsed JSON")
        
        # Validate using Pydantic schema
        if BaseModel is not None:
            validated_response = COTResponse(**response_data)
            final_choice = validated_response.final_choice
            reasoning = validated_response.reasoning
            print(f"Pydantic validation successful: {final_choice}")
            return {"final_choice": final_choice, "reasoning": reasoning}
        else:
            # Fallback validation if Pydantic not available
            final_choice = response_data.get('final_choice', '').upper()
            reasoning = response_data.get('reasoning', '')
            if final_choice in ['A', 'B']:
                print(f"Manual validation successful: {final_choice}")
                return {"final_choice": final_choice, "reasoning": reasoning}
            else:
                print(f"Invalid final_choice value: {final_choice}")
                return {"final_choice": "ERROR", "reasoning": reasoning}
                
    except json.JSONDecodeError as e:
        print(f"JSON parsing error after cleaning: {e}")
        
        # Third attempt: More aggressive cleaning and regex extraction
        try:
            print("Attempting regex extraction fallback...")
            
            # Look for final_choice pattern directly
            import re
            
            # Try to find "final_choice": "A" or "final_choice": "B"
            choice_patterns = [
                r'"final_choice"\s*:\s*"([AB])"',
                r'"final_choice"\s*:\s*\'([AB])\'',
                r'final_choice["\']?\s*:\s*["\']?([AB])["\']?',
                r'"choice"\s*:\s*"([AB])"',
                r'"answer"\s*:\s*"([AB])"'
            ]
            
            for pattern in choice_patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                if matches:
                    choice = matches[-1].upper()  # Take the last match
                    if choice in ['A', 'B']:
                        print(f"Regex extraction successful: {choice}")
                        # Try to extract reasoning from the response
                        reasoning = _extract_reasoning_from_text(response_text)
                        return {"final_choice": choice, "reasoning": reasoning}
            
            # Fourth attempt: Look for any clear indication of A or B choice
            # This is more permissive but still looks for structured indicators
            if 'final_choice' in response_text.lower():
                text_upper = response_text.upper()
                # Look for A or B near "final_choice"
                final_choice_pos = text_upper.find('FINAL_CHOICE')
                if final_choice_pos != -1:
                    # Look in a 50-character window after "final_choice"
                    window = text_upper[final_choice_pos:final_choice_pos + 50]
                    if '"A"' in window or "'A'" in window or ': A' in window:
                        print("Window extraction successful: A")
                        reasoning = _extract_reasoning_from_text(response_text)
                        return {"final_choice": "A", "reasoning": reasoning}
                    elif '"B"' in window or "'B'" in window or ': B' in window:
                        print("Window extraction successful: B")
                        reasoning = _extract_reasoning_from_text(response_text)
                        return {"final_choice": "B", "reasoning": reasoning}
            
            print("All parsing attempts failed")
            return {"final_choice": "ERROR", "reasoning": None}
            
        except Exception as regex_error:
            print(f"Regex extraction also failed: {regex_error}")
            return {"final_choice": "ERROR", "reasoning": None}
        
    except ValidationError as e:
        print(f"Pydantic validation error: {e}")
        return {"final_choice": "ERROR", "reasoning": None}
    except Exception as e:
        print(f"Unexpected parsing error: {e}")
        return {"final_choice": "ERROR", "reasoning": None}

# ===== MODEL INTERACTION =====

def wrap_as_chat(prompt: str, tok: AutoTokenizer) -> str:
    # Both COT and non-COT now use JSON responses, so we can use the same system message
    system_message = "You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON."
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    return tok.apply_chat_template(messages,
                                tokenize=False,
                                add_generation_prompt=True)

def query_vllm_model(llm_model, sampling_params, prompt, args, max_retries=3, is_cot=False):
    """
    Query the vLLM model with retry logic.
    
    Args:
        llm_model: The vLLM model instance
        sampling_params: Sampling parameters for generation
        prompt: The input prompt
        args: Arguments namespace
        max_retries: Maximum number of retry attempts
        is_cot: Whether this is a COT query (includes reasoning field)
    
    Returns:
        dict: Response dictionary with final_choice and reasoning (COT) or final_choice only (non-COT)
        str: "ERROR" if all attempts fail (for backward compatibility)
    """
    for attempt in range(max_retries):
        try:
            # For COT queries, try to use guided generation if available
            current_sampling_params = sampling_params
            if is_cot:
                try:
                    # Try to enable JSON schema guidance for COT responses
                    # Note: This may not be available in all vLLM versions
                    json_schema = COTResponse.model_json_schema() if BaseModel is not None else None
                    if json_schema and hasattr(sampling_params, 'guided_json'):
                        current_sampling_params = SamplingParams(
                            temperature=sampling_params.temperature,
                            top_p=sampling_params.top_p,
                            max_tokens=sampling_params.max_tokens,
                            guided_json=json_schema
                        )
                        print("Using guided JSON generation for COT")
                except Exception as e:
                    print(f"Note: Guided JSON generation not available: {e}")
                    # Fall back to regular sampling
                    pass
            
            # Generate response using vLLM
            outputs = llm_model.generate([prompt], current_sampling_params)
            
            if outputs and len(outputs) > 0 and outputs[0].outputs:
                result = outputs[0].outputs[0].text.strip()
                
                print(f"Attempt {attempt + 1}: Raw response: '{result[:100]}...'")
                
                # If empty response, retry
                if not result:
                    print(f"Empty response on attempt {attempt + 1}, retrying...")
                    continue
                
                # Both COT and non-COT now use JSON parsing
                parsed_result = parse_json_response(result)
                if parsed_result["final_choice"] != "ERROR":
                    return parsed_result
                else:
                    print(f"JSON parsing failed (attempt {attempt + 1}), retrying...")
                    continue
            else:
                print(f"Empty or invalid response structure (attempt {attempt + 1})")
                continue
                
        except Exception as e:
            print(f"Error querying model (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
    
    print(f"All {max_retries} attempts failed, returning ERROR")
    # Return consistent format
    return {"final_choice": "ERROR", "reasoning": None}

# ===== BATCHED EVALUATION FUNCTIONS =====

def collect_prompts_from_preference_pairs(preference_pairs, data_manager, args, user_profile_cache, tokenizer=None):
    """
    Collect all prompts from a batch of preference pairs for batched vLLM inference.
    
    Args:
        preference_pairs: List of preference pair dictionaries
        data_manager: DataManagerV3 instance
        args: Command line arguments
        user_profile_cache: Cache for user profiles
        tokenizer: Pre-loaded tokenizer (optional)
    
    Returns:
        prompts: List of formatted prompts ready for vLLM
        prompt_metadata: List of metadata for mapping responses back to preference pairs
    """
    prompts = []
    prompt_metadata = []
    
    # Use provided tokenizer or try to load one
    tok = tokenizer
    if tok is None and AutoTokenizer is not None:
        try:
            tok = AutoTokenizer.from_pretrained(args.model_path)
        except:
            print("Warning: Could not load tokenizer, using raw prompts")
    
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
        
        # ===== GENERATE PROMPTS FOR EACH SETTING =====
        
        # 1. Simple prompting evaluation (baseline)
        if args.simple_only or args.all_settings:
            simple_prompt = create_simple_prompt(prompt, completion_a, completion_b)
            if tok:
                simple_prompt = wrap_as_chat(simple_prompt, tok)
            prompts.append(simple_prompt)
            prompt_metadata.append({**base_metadata, 'setting': 'simple', 'is_cot': False})
        
        # 2. Full WVS context only
        if args.full_wvs_only or args.all_settings:
            full_wvs_only_prompt = create_full_wvs_only_prompt(prompt, completion_a, completion_b, wvs_context)
            if tok:
                full_wvs_only_prompt = wrap_as_chat(full_wvs_only_prompt, tok)
            prompts.append(full_wvs_only_prompt)
            prompt_metadata.append({**base_metadata, 'setting': 'full_wvs_only', 'is_cot': False})
        
        # 3. Full style context only
        if args.full_style_only or args.all_settings:
            full_style_only_prompt = create_full_style_only_prompt(prompt, completion_a, completion_b, style_context)
            if tok:
                full_style_only_prompt = wrap_as_chat(full_style_only_prompt, tok)
            prompts.append(full_style_only_prompt)
            prompt_metadata.append({**base_metadata, 'setting': 'full_style_only', 'is_cot': False})
        
        # 4. Full WVS + Style context (prefer WVS)
        if args.full_combined or args.all_settings:
            full_wvs_style_prefer_wvs_prompt = create_full_wvs_style_prefer_wvs_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            if tok:
                full_wvs_style_prefer_wvs_prompt = wrap_as_chat(full_wvs_style_prefer_wvs_prompt, tok)
            prompts.append(full_wvs_style_prefer_wvs_prompt)
            prompt_metadata.append({**base_metadata, 'setting': 'full_wvs_style_prefer_wvs', 'is_cot': False})
        
        # 5. Full WVS + Style context (prefer style)
        if args.full_combined or args.all_settings:
            full_wvs_style_prefer_style_prompt = create_full_wvs_style_prefer_style_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            if tok:
                full_wvs_style_prefer_style_prompt = wrap_as_chat(full_wvs_style_prefer_style_prompt, tok)
            prompts.append(full_wvs_style_prefer_style_prompt)
            prompt_metadata.append({**base_metadata, 'setting': 'full_wvs_style_prefer_style', 'is_cot': False})
        
        # 6. Full WVS + Style context (neutral)
        if args.full_combined or args.all_settings:
            full_wvs_style_neutral_prompt = create_full_wvs_style_neutral_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            if tok:
                full_wvs_style_neutral_prompt = wrap_as_chat(full_wvs_style_neutral_prompt, tok)
            prompts.append(full_wvs_style_neutral_prompt)
            prompt_metadata.append({**base_metadata, 'setting': 'full_wvs_style_neutral', 'is_cot': False})
        
        # ===== COT SETTINGS =====
        
        if args.cot_only or args.all_settings:
            # COT version of full WVS only
            full_wvs_only_cot_prompt = create_full_wvs_only_cot_prompt(prompt, completion_a, completion_b, wvs_context)
            if tok:
                full_wvs_only_cot_prompt = wrap_as_chat(full_wvs_only_cot_prompt, tok)
            prompts.append(full_wvs_only_cot_prompt)
            prompt_metadata.append({**base_metadata, 'setting': 'full_wvs_only_cot', 'is_cot': True})
            
            # COT version of full style only
            full_style_only_cot_prompt = create_full_style_only_cot_prompt(prompt, completion_a, completion_b, style_context)
            if tok:
                full_style_only_cot_prompt = wrap_as_chat(full_style_only_cot_prompt, tok)
            prompts.append(full_style_only_cot_prompt)
            prompt_metadata.append({**base_metadata, 'setting': 'full_style_only_cot', 'is_cot': True})
            
            # COT version of full combined (prefer WVS)
            full_wvs_style_cot_prefer_wvs_prompt = create_full_wvs_style_cot_prefer_wvs_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            if tok:
                full_wvs_style_cot_prefer_wvs_prompt = wrap_as_chat(full_wvs_style_cot_prefer_wvs_prompt, tok)
            prompts.append(full_wvs_style_cot_prefer_wvs_prompt)
            prompt_metadata.append({**base_metadata, 'setting': 'full_wvs_style_cot_prefer_wvs', 'is_cot': True})
            
            # COT version of full combined (prefer style)
            full_wvs_style_cot_prefer_style_prompt = create_full_wvs_style_cot_prefer_style_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            if tok:
                full_wvs_style_cot_prefer_style_prompt = wrap_as_chat(full_wvs_style_cot_prefer_style_prompt, tok)
            prompts.append(full_wvs_style_cot_prefer_style_prompt)
            prompt_metadata.append({**base_metadata, 'setting': 'full_wvs_style_cot_prefer_style', 'is_cot': True})
            
            # COT version of full combined (neutral)
            full_wvs_style_cot_neutral_prompt = create_full_wvs_style_cot_neutral_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            if tok:
                full_wvs_style_cot_neutral_prompt = wrap_as_chat(full_wvs_style_cot_neutral_prompt, tok)
            prompts.append(full_wvs_style_cot_neutral_prompt)
            prompt_metadata.append({**base_metadata, 'setting': 'full_wvs_style_cot_neutral', 'is_cot': True})
    
    return prompts, prompt_metadata


def query_vllm_model_batch(llm_model, sampling_params, prompts, prompt_metadata, args, max_retries=3):
    """
    Query the vLLM model with a batch of prompts with retry logic.
    
    Args:
        llm_model: The vLLM model instance
        sampling_params: Sampling parameters for generation
        prompts: List of formatted prompts ready for vLLM
        prompt_metadata: List of metadata for mapping responses back
        args: Arguments namespace
        max_retries: Maximum number of retry attempts
    
    Returns:
        List of response dictionaries with final_choice and reasoning
    """
    for attempt in range(max_retries):
        try:
            print(f"Sending batch of {len(prompts)} prompts to vLLM (attempt {attempt + 1})")
            
            # Generate responses using vLLM batch inference
            outputs = llm_model.generate(prompts, sampling_params)
            
            if not outputs or len(outputs) != len(prompts):
                print(f"Invalid batch response structure (attempt {attempt + 1})")
                continue
            
            # Parse all responses
            parsed_responses = []
            failed_count = 0
            
            for i, output in enumerate(outputs):
                if output.outputs and len(output.outputs) > 0:
                    result = output.outputs[0].text.strip()
                    metadata = prompt_metadata[i]
                    
                    if not result:
                        print(f"Empty response for prompt {i} (attempt {attempt + 1})")
                        failed_count += 1
                        parsed_responses.append({"final_choice": "ERROR", "reasoning": None})
                        continue
                    
                    # Parse JSON response (both COT and non-COT use JSON now)
                    parsed_result = parse_json_response(result)
                    parsed_responses.append(parsed_result)
                    
                    if parsed_result["final_choice"] == "ERROR":
                        failed_count += 1
                else:
                    print(f"Empty output structure for prompt {i} (attempt {attempt + 1})")
                    failed_count += 1
                    parsed_responses.append({"final_choice": "ERROR", "reasoning": None})
            
            # If most responses parsed successfully, return the batch
            success_rate = (len(prompts) - failed_count) / len(prompts)
            if success_rate >= 0.8:  # At least 80% success rate
                print(f"Batch completed with {success_rate:.1%} success rate ({failed_count} failures)")
                return parsed_responses
            else:
                print(f"Batch failed with {success_rate:.1%} success rate (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    continue
                else:
                    print(f"Returning batch with failures after {max_retries} attempts")
                    return parsed_responses
                
        except Exception as e:
            print(f"Error in batch inference (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
    
    print(f"All {max_retries} batch attempts failed, returning errors")
    return [{"final_choice": "ERROR", "reasoning": None} for _ in prompts]


def evaluate_preference_pairs_batch_optimized(preference_pairs, data_manager, llm_model, sampling_params, args, tokenizer=None):
    """
    Evaluate a batch of preference pairs using optimized batched vLLM inference.
    
    Args:
        preference_pairs: List of preference pair dictionaries
        data_manager: DataManagerV3 instance
        llm_model: vLLM model instance  
        sampling_params: Sampling parameters
        args: Command line arguments
        tokenizer: Pre-loaded tokenizer (optional, for efficiency)
        
    Returns:
        List of evaluation result dictionaries
    """
    user_profile_cache = {}
    
    # Collect all prompts from the batch
    print(f"Collecting prompts from {len(preference_pairs)} preference pairs...")
    prompts, prompt_metadata = collect_prompts_from_preference_pairs(
        preference_pairs, data_manager, args, user_profile_cache, tokenizer)
    
    print(f"Generated {len(prompts)} total prompts for batch inference")
    
    # Query vLLM with all prompts at once
    responses = query_vllm_model_batch(llm_model, sampling_params, prompts, prompt_metadata, args)
    
    # Group responses back by preference pair
    results_by_preference = defaultdict(dict)
    
    for response, metadata in zip(responses, prompt_metadata):
        preference_id = metadata['preference_id']
        setting = metadata['setting']
        
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
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        
        # Add response data for this setting
        final_choice = response['final_choice']
        reasoning = response.get('reasoning', None)
        correct_answer = metadata['correct_answer']
        
        results_by_preference[preference_id][f'{setting}_response'] = final_choice
        results_by_preference[preference_id][f'{setting}_reasoning'] = reasoning
        results_by_preference[preference_id][f'{setting}_correct'] = final_choice == correct_answer if final_choice != "ERROR" else False
    
    # Convert back to list format
    evaluation_results = list(results_by_preference.values())
    
    print(f"Successfully processed {len(evaluation_results)} preference pairs")
    return evaluation_results

# ===== MAIN EVALUATION FUNCTION =====

def evaluate_preference_pair_all_settings(preference_pair, data_manager, llm_model, sampling_params, args, user_profile=None):
    """
    Evaluate a single preference pair using selected evaluation settings.
    
    Args:
        preference_pair: Preference pair dictionary from evaluation engine
        data_manager: DataManagerV3 instance
        llm_model: vLLM model instance
        sampling_params: Sampling parameters
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
    
    # Get tokenizer for chat formatting
    tok = None
    if AutoTokenizer is not None:
        try:
            tok = AutoTokenizer.from_pretrained(args.model_path)
        except:
            print("Warning: Could not load tokenizer, using raw prompts")
    
    # All evaluation responses
    responses = {}
    
    # For combined contexts, randomize the order of WVS and style contexts
    wvs_first = random.choice([True, False])
    
    # ===== EVALUATION SETTINGS =====
    
    # 1. Simple prompting evaluation (baseline)
    if args.simple_only or args.all_settings:
        simple_prompt = create_simple_prompt(prompt, completion_a, completion_b)
        if tok:
            simple_prompt = wrap_as_chat(simple_prompt, tok)
        responses['simple'] = query_vllm_model(llm_model, sampling_params, simple_prompt, args)
    
    # 2. Full WVS context only
    if args.full_wvs_only or args.all_settings:
        full_wvs_only_prompt = create_full_wvs_only_prompt(prompt, completion_a, completion_b, wvs_context)
        if tok:
            full_wvs_only_prompt = wrap_as_chat(full_wvs_only_prompt, tok)
        responses['full_wvs_only'] = query_vllm_model(llm_model, sampling_params, full_wvs_only_prompt, args)
    
    # 3. Full style context only
    if args.full_style_only or args.all_settings:
        full_style_only_prompt = create_full_style_only_prompt(prompt, completion_a, completion_b, style_context)
        if tok:
            full_style_only_prompt = wrap_as_chat(full_style_only_prompt, tok)
        responses['full_style_only'] = query_vllm_model(llm_model, sampling_params, full_style_only_prompt, args)
    
    # 4. Full WVS + Style context (prefer WVS)
    if args.full_combined or args.all_settings:
        full_wvs_style_prefer_wvs_prompt = create_full_wvs_style_prefer_wvs_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        if tok:
            full_wvs_style_prefer_wvs_prompt = wrap_as_chat(full_wvs_style_prefer_wvs_prompt, tok)
        responses['full_wvs_style_prefer_wvs'] = query_vllm_model(llm_model, sampling_params, full_wvs_style_prefer_wvs_prompt, args)
    
    # 5. Full WVS + Style context (prefer style)
    if args.full_combined or args.all_settings:
        full_wvs_style_prefer_style_prompt = create_full_wvs_style_prefer_style_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        if tok:
            full_wvs_style_prefer_style_prompt = wrap_as_chat(full_wvs_style_prefer_style_prompt, tok)
        responses['full_wvs_style_prefer_style'] = query_vllm_model(llm_model, sampling_params, full_wvs_style_prefer_style_prompt, args)
    
    # 6. Full WVS + Style context (neutral)
    if args.full_combined or args.all_settings:
        full_wvs_style_neutral_prompt = create_full_wvs_style_neutral_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        if tok:
            full_wvs_style_neutral_prompt = wrap_as_chat(full_wvs_style_neutral_prompt, tok)
        responses['full_wvs_style_neutral'] = query_vllm_model(llm_model, sampling_params, full_wvs_style_neutral_prompt, args)
    
    # ===== COT SETTINGS =====
    
    if args.cot_only or args.all_settings:
        # COT version of full WVS only
        full_wvs_only_cot_prompt = create_full_wvs_only_cot_prompt(prompt, completion_a, completion_b, wvs_context)
        if tok:
            full_wvs_only_cot_prompt = wrap_as_chat(full_wvs_only_cot_prompt, tok)
        responses['full_wvs_only_cot'] = query_vllm_model(llm_model, sampling_params, full_wvs_only_cot_prompt, args, is_cot=True)
        
        # COT version of full style only
        full_style_only_cot_prompt = create_full_style_only_cot_prompt(prompt, completion_a, completion_b, style_context)
        if tok:
            full_style_only_cot_prompt = wrap_as_chat(full_style_only_cot_prompt, tok)
        responses['full_style_only_cot'] = query_vllm_model(llm_model, sampling_params, full_style_only_cot_prompt, args, is_cot=True)
        
        # COT version of full combined (prefer WVS)
        full_wvs_style_cot_prefer_wvs_prompt = create_full_wvs_style_cot_prefer_wvs_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        if tok:
            full_wvs_style_cot_prefer_wvs_prompt = wrap_as_chat(full_wvs_style_cot_prefer_wvs_prompt, tok)
        responses['full_wvs_style_cot_prefer_wvs'] = query_vllm_model(llm_model, sampling_params, full_wvs_style_cot_prefer_wvs_prompt, args, is_cot=True)
        
        # COT version of full combined (prefer style)
        full_wvs_style_cot_prefer_style_prompt = create_full_wvs_style_cot_prefer_style_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        if tok:
            full_wvs_style_cot_prefer_style_prompt = wrap_as_chat(full_wvs_style_cot_prefer_style_prompt, tok)
        responses['full_wvs_style_cot_prefer_style'] = query_vllm_model(llm_model, sampling_params, full_wvs_style_cot_prefer_style_prompt, args, is_cot=True)
        
        # COT version of full combined (neutral)
        full_wvs_style_cot_neutral_prompt = create_full_wvs_style_cot_neutral_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        if tok:
            full_wvs_style_cot_neutral_prompt = wrap_as_chat(full_wvs_style_cot_neutral_prompt, tok)
        responses['full_wvs_style_cot_neutral'] = query_vllm_model(llm_model, sampling_params, full_wvs_style_cot_neutral_prompt, args, is_cot=True)
    
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
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add all responses and correctness
    for setting_name, response in responses.items():
        # All responses now return dictionaries with final_choice and optional reasoning
        if isinstance(response, dict) and 'final_choice' in response:
            final_choice = response['final_choice']
            reasoning = response.get('reasoning', None)  # COT has reasoning, non-COT doesn't
            result[f'{setting_name}_response'] = final_choice
            result[f'{setting_name}_reasoning'] = reasoning
            result[f'{setting_name}_correct'] = final_choice == correct_answer if final_choice != "ERROR" else False
        else:
            # Fallback for any legacy string responses (shouldn't happen with new implementation)
            result[f'{setting_name}_response'] = response if response != "ERROR" else "ERROR"
            result[f'{setting_name}_reasoning'] = None
            result[f'{setting_name}_correct'] = response == correct_answer if response != "ERROR" else False
    
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
            if key.endswith('_response') and not key.endswith('_reasoning'):
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
    # Check if vLLM is available
    if LLM is None or SamplingParams is None:
        print("Error: vLLM is not installed. Please install it with: pip install vllm")
        return
    
    parser = argparse.ArgumentParser(description="Unified Reward Model Evaluation v3")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model (HuggingFace model path or local path)")
    parser.add_argument("--max_tokens", type=int, default=800,
                       help="Maximum tokens for model response (increased for COT)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p sampling parameter")
    parser.add_argument("--tensor_parallel_size", type=int, default=4,
                       help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max_model_len", type=int, default=None,
                       help="Maximum model sequence length")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Number of preference pairs to process in parallel per batch")
    parser.add_argument("--enable_vllm_batching", action="store_true", default=True,
                       help="Enable optimized vLLM batching for maximum throughput (default: True)")
    parser.add_argument("--disable_vllm_batching", dest="enable_vllm_batching", action="store_false",
                       help="Disable vLLM batching and use individual calls (for debugging)")
    
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
    setting_group.add_argument("--cot_only", action="store_true",
                              help="Run COT versions only (WVS, style, and combined)")
    setting_group.add_argument("--all_settings", action="store_true",
                              help="Run all evaluation settings")
    
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
        print(f"Random seed set to: {args.random_seed}")
    
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
        elif args.cot_only:
            setting_suffix = "cot"
        else:
            setting_suffix = "all_settings"
        
        args.output_file = f"{model_name}_v3_{setting_suffix}_{args.questions}_results.json"
    
    print("=== Unified Reward Model Evaluation v3 with Batch Processing ===")
    print(f"Model: {args.model_path}")
    print(f"Question group: {args.questions}")
    print(f"Output: {args.output_file}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"vLLM batching: {'ENABLED' if args.enable_vllm_batching else 'DISABLED'}")
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
    elif args.cot_only:
        print("COT versions (WVS, style, combined)")
    elif args.all_settings:
        print("All evaluation settings (simple, WVS, style, combined, COT)")
    
    # Determine evaluation type early
    if args.simple_only:
        evaluation_type = 'simple_prompting_v3'
    elif args.full_wvs_only:
        evaluation_type = 'full_wvs_only_v3'
    elif args.full_style_only:
        evaluation_type = 'full_style_only_v3'
    elif args.full_combined:
        evaluation_type = 'full_combined_v3'
    elif args.cot_only:
        evaluation_type = 'cot_only_v3'
    else:
        evaluation_type = 'all_settings_v3'

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
    elif args.cot_only:
        setting_name = 'cot_only'
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
    
    # Initialize vLLM model
    print(f"\n=== Initializing Model ===")
    try:
        llm_kwargs = {
            "model": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "trust_remote_code": True
        }
        
        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = args.max_model_len
            print(f"Using max_model_len: {args.max_model_len}")
        
        llm_model = LLM(**llm_kwargs)
        
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        
        print("Model loaded successfully")
        
        # Pre-load tokenizer for chat formatting (optimization for batching)
        shared_tokenizer = None
        if AutoTokenizer is not None and args.enable_vllm_batching:
            try:
                shared_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
                print("Tokenizer loaded for batching optimization")
            except:
                print("Warning: Could not load tokenizer, will use raw prompts")
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Initialize results storage
    evaluation_results = existing_results.get('evaluation_results', []) if existing_results else []
    
    # Track this job's initial contribution to avoid double-counting progress
    initial_results_count = len(evaluation_results)
    job_target_count = len(preference_pairs)
    
    # Create output data structure
    output_data = {
        'metadata': {
            'model_path': args.model_path,
            'evaluation_type': evaluation_type,
            'total_preferences': len(preference_pairs) + len(evaluation_results),
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'tensor_parallel_size': args.tensor_parallel_size,
            'random_seed': args.random_seed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'in_progress',
            'parallel_job_info': {
                'job_index': args.job_index,
                'total_jobs': args.total_jobs,
                'job_preference_count': len(preference_pairs)
            } if args.total_jobs > 1 else None
        },
        'evaluation_results': evaluation_results,
        'metrics': {}
    }
    
    # Calculate estimated time
    print(f"\n=== Evaluation Time Estimation ===")

    total_evaluations = len(preference_pairs)
    print(f"Total evaluations: {total_evaluations:,}")
    
    if args.enable_vllm_batching:
        # Count total prompts per batch based on selected settings
        prompts_per_pair = 0
        if args.simple_only: prompts_per_pair = 1
        elif args.full_wvs_only: prompts_per_pair = 1  
        elif args.full_style_only: prompts_per_pair = 1
        elif args.full_combined: prompts_per_pair = 3
        elif args.cot_only: prompts_per_pair = 5
        elif args.all_settings: prompts_per_pair = 11
        
        total_batches = (total_evaluations + args.batch_size - 1) // args.batch_size
        avg_prompts_per_vllm_call = prompts_per_pair * args.batch_size
        print(f"Prompts per preference pair: {prompts_per_pair}")
        print(f"Average prompts per vLLM call: {avg_prompts_per_vllm_call}")
        print(f"Total vLLM calls needed: ~{total_batches} (vs {total_evaluations * prompts_per_pair} without batching)")
        print(f"Estimated throughput improvement: {prompts_per_pair * args.batch_size:.0f}x per vLLM call")
    else:
        print("Note: vLLM batching disabled - using individual calls (slower)")
    
    # Run evaluations with batch processing
    print(f"\n=== Running Evaluations ===")
    print(f"Processing {len(preference_pairs)} preference pairs in batches of {args.batch_size}")
    
    try:
        start_time = time.time()
        
        # Process preference pairs in batches with optimized vLLM batching
        total_items = len(preference_pairs)
        
        for batch_start in range(0, total_items, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total_items)
            batch_pairs = preference_pairs[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//args.batch_size + 1}/{(total_items-1)//args.batch_size + 1} "
                  f"(items {batch_start+1}-{batch_end}/{total_items})")
            
            try:
                # Use optimized batch evaluation if enabled, otherwise fallback to individual processing
                if args.enable_vllm_batching:
                    batch_results = evaluate_preference_pairs_batch_optimized(
                        batch_pairs, data_manager, llm_model, sampling_params, args, shared_tokenizer
                    )
                    evaluation_results.extend(batch_results)
                else:
                    # Process individually when batching is disabled
                    user_profile_cache = {}
                    for preference_pair in batch_pairs:
                        try:
                            # Get user profile from cache or load it
                            user_profile_id = preference_pair['user_profile_id']
                            if user_profile_id not in user_profile_cache:
                                user_profile_cache[user_profile_id] = data_manager.get_user_profile_by_id(user_profile_id)
                            user_profile = user_profile_cache[user_profile_id]
                            
                            # Evaluate this preference pair individually
                            result = evaluate_preference_pair_all_settings(
                                preference_pair, data_manager, llm_model, sampling_params, args, user_profile
                            )
                            
                            evaluation_results.append(result)
                            
                        except Exception as e2:
                            print(f"Error evaluating preference pair {preference_pair.get('preference_id', 'unknown')}: {e2}")
                            continue
                
            except Exception as e:
                print(f"Error in batch evaluation: {e}")
                print(f"Traceback: {traceback.format_exc()}")
                
                # Fallback to individual processing if batch fails
                print("Falling back to individual processing for this batch...")
                user_profile_cache = user_profile_cache if 'user_profile_cache' in locals() else {}
                
                for preference_pair in batch_pairs:
                    try:
                        # Get user profile from cache or load it
                        user_profile_id = preference_pair['user_profile_id']
                        if user_profile_id not in user_profile_cache:
                            user_profile_cache[user_profile_id] = data_manager.get_user_profile_by_id(user_profile_id)
                        user_profile = user_profile_cache[user_profile_id]
                        
                        # Evaluate this preference pair individually
                        result = evaluate_preference_pair_all_settings(
                            preference_pair, data_manager, llm_model, sampling_params, args, user_profile
                        )
                        
                        evaluation_results.append(result)
                        
                    except Exception as e2:
                        print(f"Error evaluating individual preference pair {preference_pair.get('preference_id', 'unknown')}: {e2}")
                        continue
            
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
                    
                    actual_item_start = start_idx + batch_start if args.total_jobs > 1 else batch_start
                    actual_item_end = start_idx + batch_end - 1 if args.total_jobs > 1 else batch_end - 1
                    print(f"Completed batch {batch_start//args.batch_size + 1}. "
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
            print(f"  {setting_name}: {value:.1%} accuracy ({success_rate:.1%} success rate)")
    
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