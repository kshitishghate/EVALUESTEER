import json
import pandas as pd
import numpy as np
import random
import asyncio
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
    import openai
    from openai import AsyncOpenAI
except ImportError:
    print("Warning: OpenAI not installed. Please install with: pip install openai")
    openai = None
    AsyncOpenAI = None

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

# ===== PROMPT GENERATION FUNCTIONS (adapted from v3 for OpenAI chat format) =====

def create_simple_prompt(prompt, completion_a, completion_b):
    """Create a simple prompt without any context for OpenAI chat format"""
    return [
        {"role": "system", "content": "You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON."},
        {"role": "user", "content": f"""Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Which response (A or B) better answers the question? Consider factors like relevance, helpfulness, accuracy, and overall quality.

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""}
    ]

def create_full_wvs_only_prompt(prompt, completion_a, completion_b, wvs_context):
    """Create a prompt with full WVS context only for OpenAI chat format"""
    return [
        {"role": "system", "content": "You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON."},
        {"role": "user", "content": f"""User's Values and Beliefs:
{wvs_context}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and beliefs, which response (A or B) would this specific user prefer?

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""}
    ]

def create_full_style_only_prompt(prompt, completion_a, completion_b, style_context):
    """Create a prompt with full style context only for OpenAI chat format"""
    return [
        {"role": "system", "content": "You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON."},
        {"role": "user", "content": f"""User's Style Preferences:
{style_context}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's style preferences, which response (A or B) would this specific user prefer?

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""}
    ]

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
    
    return [
        {"role": "system", "content": "You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON."},
        {"role": "user", "content": f"""{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer? IMPORTANT: Aligning with the user's values and beliefs takes the highest precedence. Style preferences are secondary to value alignment.

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""}
    ]

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
    
    return [
        {"role": "system", "content": "You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON."},
        {"role": "user", "content": f"""{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer? IMPORTANT: This user prioritizes communication style and format over content alignment with their values. Style preferences take the highest precedence.

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""}
    ]

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
    
    return [
        {"role": "system", "content": "You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON."},
        {"role": "user", "content": f"""{context_section}

Question: {prompt}

Response A: {completion_a}

Response B: {completion_b}

Given what you know about this user's values and style preferences, which response (A or B) would this specific user prefer?

Respond with valid JSON in exactly this format:
{{
    "final_choice": "A"
}}

Where final_choice must be either "A" or "B"."""}
    ]

# ===== COT PROMPT FUNCTIONS =====

def create_full_wvs_only_cot_prompt(prompt, completion_a, completion_b, wvs_context):
    """Create a COT prompt with full WVS context only - returns JSON"""
    return [
        {"role": "system", "content": "You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON."},
        {"role": "user", "content": f"""User's Values and Beliefs:
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

Where final_choice must be either "A" or "B"."""}
    ]

def create_full_style_only_cot_prompt(prompt, completion_a, completion_b, style_context):
    """Create a COT prompt with full style context only - returns JSON"""
    return [
        {"role": "system", "content": "You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON."},
        {"role": "user", "content": f"""User's Style Preferences:
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

Where final_choice must be either "A" or "B"."""}
    ]

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
    
    return [
        {"role": "system", "content": "You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON."},
        {"role": "user", "content": f"""{context_section}

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

Where final_choice must be either "A" or "B"."""}
    ]

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
    
    return [
        {"role": "system", "content": "You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON."},
        {"role": "user", "content": f"""{context_section}

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

Where final_choice must be either "A" or "B"."""}
    ]

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
    
    return [
        {"role": "system", "content": "You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON."},
        {"role": "user", "content": f"""{context_section}

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

Where final_choice must be either "A" or "B"."""}
    ]

# ===== JSON PARSING FOR COT (reused from v3) =====

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

# ===== ASYNC OPENAI MODEL INTERACTION =====

async def query_openai_model_async(
    openai_client, messages, model_name, max_tokens=800, temperature=0.0, 
    max_retries=3, retry_delay=1, is_cot=False
):
    """
    Query the OpenAI model with retry logic using async API.
    
    Args:
        openai_client: AsyncOpenAI client instance
        messages: List of message dictionaries for chat completion
        model_name: OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        is_cot: Whether this is a COT query (includes reasoning field)
        
    Returns:
        dict: Response dictionary with final_choice and reasoning (COT) or final_choice only (non-COT)
    """
    for attempt in range(max_retries):
        try:
            # Make async API call
            response = await openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=30  # 30 second timeout
            )
            
            if response and response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content.strip()
                
                print(f"Attempt {attempt + 1}: Raw response: '{result[:100]}...'")
                
                # If empty response, retry
                if not result:
                    print(f"Empty response on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(retry_delay)
                    continue
                
                # Both COT and non-COT now use JSON parsing
                parsed_result = parse_json_response(result)
                if parsed_result["final_choice"] != "ERROR":
                    return parsed_result
                else:
                    print(f"JSON parsing failed (attempt {attempt + 1}), retrying...")
                    await asyncio.sleep(retry_delay)
                    continue
            else:
                print(f"Empty or invalid response structure (attempt {attempt + 1})")
                await asyncio.sleep(retry_delay)
                continue
                
        except Exception as e:
            print(f"Error querying model (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            continue
    
    print(f"All {max_retries} attempts failed, returning ERROR")
    # Return consistent format
    return {"final_choice": "ERROR", "reasoning": None}

async def batch_query_openai_models(
    openai_client, prompts_and_settings, model_name, max_tokens=800, 
    temperature=0.0, max_retries=3, retry_delay=1, max_workers=5
):
    """
    Query OpenAI model for multiple prompts in parallel with rate limiting.
    
    Args:
        openai_client: AsyncOpenAI client instance
        prompts_and_settings: List of tuples (messages, is_cot, setting_name)
        model_name: OpenAI model name
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        max_workers: Maximum number of concurrent requests
        
    Returns:
        Dictionary mapping setting_name to response
    """
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_workers)
    
    async def bounded_query(messages, is_cot, setting_name):
        async with semaphore:
            result = await query_openai_model_async(
                openai_client, messages, model_name, max_tokens, temperature,
                max_retries, retry_delay, is_cot
            )
            return setting_name, result
    
    # Create tasks for all prompts
    tasks = [
        bounded_query(messages, is_cot, setting_name)
        for messages, is_cot, setting_name in prompts_and_settings
    ]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    response_dict = {}
    for result in results:
        if isinstance(result, Exception):
            print(f"Error in batch query: {result}")
            continue
        
        setting_name, response = result
        response_dict[setting_name] = response
    
    return response_dict

# ===== MAIN EVALUATION FUNCTION =====

async def evaluate_preference_pair_all_settings_async(preference_pair, data_manager, openai_client, args, user_profile=None):
    """
    Evaluate a single preference pair using selected evaluation settings with async OpenAI API.
    
    Args:
        preference_pair: Preference pair dictionary from evaluation engine
        data_manager: DataManagerV3 instance
        openai_client: AsyncOpenAI client instance
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
    
    # Prepare all prompts for batch processing
    prompts_and_settings = []
    
    # ===== EVALUATION SETTINGS =====
    
    # 1. Simple prompting evaluation (baseline)
    if args.simple_only or args.all_settings:
        simple_messages = create_simple_prompt(prompt, completion_a, completion_b)
        prompts_and_settings.append((simple_messages, False, 'simple'))
    
    # 2. Full WVS context only
    if args.full_wvs_only or args.all_settings:
        full_wvs_only_messages = create_full_wvs_only_prompt(prompt, completion_a, completion_b, wvs_context)
        prompts_and_settings.append((full_wvs_only_messages, False, 'full_wvs_only'))
    
    # 3. Full style context only
    if args.full_style_only or args.all_settings:
        full_style_only_messages = create_full_style_only_prompt(prompt, completion_a, completion_b, style_context)
        prompts_and_settings.append((full_style_only_messages, False, 'full_style_only'))
    
    # 4. Full WVS + Style context (prefer WVS)
    if args.full_combined or args.all_settings:
        full_wvs_style_prefer_wvs_messages = create_full_wvs_style_prefer_wvs_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        prompts_and_settings.append((full_wvs_style_prefer_wvs_messages, False, 'full_wvs_style_prefer_wvs'))
    
    # 5. Full WVS + Style context (prefer style)
    if args.full_combined or args.all_settings:
        full_wvs_style_prefer_style_messages = create_full_wvs_style_prefer_style_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        prompts_and_settings.append((full_wvs_style_prefer_style_messages, False, 'full_wvs_style_prefer_style'))
    
    # 6. Full WVS + Style context (neutral)
    if args.full_combined or args.all_settings:
        full_wvs_style_neutral_messages = create_full_wvs_style_neutral_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        prompts_and_settings.append((full_wvs_style_neutral_messages, False, 'full_wvs_style_neutral'))
    
    # ===== COT SETTINGS =====
    
    if args.cot_only or args.all_settings:
        # COT version of full WVS only
        full_wvs_only_cot_messages = create_full_wvs_only_cot_prompt(prompt, completion_a, completion_b, wvs_context)
        prompts_and_settings.append((full_wvs_only_cot_messages, True, 'full_wvs_only_cot'))
        
        # COT version of full style only
        full_style_only_cot_messages = create_full_style_only_cot_prompt(prompt, completion_a, completion_b, style_context)
        prompts_and_settings.append((full_style_only_cot_messages, True, 'full_style_only_cot'))
        
        # COT version of full combined (prefer WVS)
        full_wvs_style_cot_prefer_wvs_messages = create_full_wvs_style_cot_prefer_wvs_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        prompts_and_settings.append((full_wvs_style_cot_prefer_wvs_messages, True, 'full_wvs_style_cot_prefer_wvs'))
        
        # COT version of full combined (prefer style)
        full_wvs_style_cot_prefer_style_messages = create_full_wvs_style_cot_prefer_style_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        prompts_and_settings.append((full_wvs_style_cot_prefer_style_messages, True, 'full_wvs_style_cot_prefer_style'))
        
        # COT version of full combined (neutral)
        full_wvs_style_cot_neutral_messages = create_full_wvs_style_cot_neutral_prompt(
            prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
        prompts_and_settings.append((full_wvs_style_cot_neutral_messages, True, 'full_wvs_style_cot_neutral'))
    
    # Execute all prompts in parallel
    responses = await batch_query_openai_models(
        openai_client, prompts_and_settings, args.model_name, 
        args.max_tokens, args.temperature, args.max_retries, args.retry_delay, args.max_workers
    )
    
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

# ===== RESULT MANAGEMENT (reused from v3) =====

def save_incremental_results(output_data, output_file):
    """Save results incrementally to avoid data loss"""
    temp_file = output_file + ".temp"
    try:
        with open(temp_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Atomic move
        os.rename(temp_file, output_file)
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving results: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

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

# ===== UTILITY FUNCTIONS (adapted from v3) =====

def estimate_time_per_evaluation(args):
    """
    Estimate time per evaluation based on model and settings.
    
    Returns estimated seconds per evaluation.
    """
    # Base times for different OpenAI models (rough estimates)
    model_name = args.model_name.lower()
    
    if 'gpt-4' in model_name:
        base_time = 2.0  # GPT-4 is slower but more capable
    elif 'gpt-3.5-turbo' in model_name:
        base_time = 0.8  # GPT-3.5-turbo is faster
    else:
        base_time = 1.5  # default estimate
    
    # Adjust for settings complexity
    setting_multiplier = 1.0
    if args.all_settings:
        setting_multiplier = 1.5  # Parallel processing helps, but still more work
    elif args.cot_only:
        setting_multiplier = 1.8  # COT takes longer
    elif args.full_combined:
        setting_multiplier = 1.4  # Combined context is longer
    elif args.full_wvs_only or args.full_style_only:
        setting_multiplier = 1.2  # Context adds time
    
    # Adjust for generation parameters
    token_multiplier = args.max_tokens / 800  # Based on default of 800
    temp_multiplier = 1.0 + (args.temperature * 0.1)  # Higher temp = slightly slower
    
    # Adjust for parallel processing (more workers = faster, but with API limits)
    parallel_speedup = min(args.max_workers * 0.7, args.max_workers)  # Diminishing returns due to API limits
    
    # Adjust for batch processing (batch_size preference pairs processed in parallel)
    batch_speedup = min(args.batch_size * 0.8, args.batch_size)  # Diminishing returns from batching overhead
    
    estimated_time = (base_time * setting_multiplier * token_multiplier * temp_multiplier) / (parallel_speedup * batch_speedup)
    
    return max(estimated_time, 0.3)  # Minimum 0.3 seconds per eval due to API latency

def format_duration(seconds):
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    else:
        days = seconds / 86400
        return f"{days:.1f} days"

def get_evaluation_count_estimate(args, data_manager):
    """
    Calculate estimated evaluation counts for all settings using proper filtering.
    
    Returns a dictionary with evaluation counts for each setting.
    """
    # Initialize evaluation engine to get proper counts
    evaluation_engine = EvaluationEngineV3(data_manager)
    
    counts = {}
    
    if args.simple_only:
        setting_counts = evaluation_engine.get_evaluation_counts_for_setting('simple_only')
        counts['simple'] = setting_counts['total_evaluations']
        
    if args.full_wvs_only:
        setting_counts = evaluation_engine.get_evaluation_counts_for_setting('full_wvs_only')
        counts['full_wvs_only'] = setting_counts['total_evaluations']
        
    if args.full_style_only:
        setting_counts = evaluation_engine.get_evaluation_counts_for_setting('full_style_only')
        counts['full_style_only'] = setting_counts['total_evaluations']
        
    if args.full_combined:
        # Combined has 3 priority settings
        setting_counts = evaluation_engine.get_evaluation_counts_for_setting('full_combined')
        counts['full_combined'] = setting_counts['total_evaluations'] * 3
        
    if args.cot_only:
        # COT versions of WVS, style, and combined
        wvs_counts = evaluation_engine.get_evaluation_counts_for_setting('full_wvs_only')
        style_counts = evaluation_engine.get_evaluation_counts_for_setting('full_style_only')
        combined_counts = evaluation_engine.get_evaluation_counts_for_setting('full_combined')
        cot_total = wvs_counts['total_evaluations'] + style_counts['total_evaluations'] + (combined_counts['total_evaluations'] * 3)
        counts['cot'] = cot_total
        
    if args.all_settings:
        # All of the above
        simple_counts = evaluation_engine.get_evaluation_counts_for_setting('simple_only')
        wvs_counts = evaluation_engine.get_evaluation_counts_for_setting('full_wvs_only')
        style_counts = evaluation_engine.get_evaluation_counts_for_setting('full_style_only')
        combined_counts = evaluation_engine.get_evaluation_counts_for_setting('full_combined')
        
        # Calculate total: simple + wvs + style + combined*3 + cot versions
        base_total = (simple_counts['total_evaluations'] + 
                     wvs_counts['total_evaluations'] + 
                     style_counts['total_evaluations'] + 
                     (combined_counts['total_evaluations'] * 3))
        cot_total = (wvs_counts['total_evaluations'] + 
                    style_counts['total_evaluations'] + 
                    (combined_counts['total_evaluations'] * 3))
        counts['all_settings'] = base_total + cot_total
    
    return counts

# ===== MAIN FUNCTION =====

async def main():
    # Check if OpenAI is available
    if AsyncOpenAI is None:
        print("Error: OpenAI is not installed. Please install it with: pip install openai")
        return
    
    parser = argparse.ArgumentParser(description="Unified Reward Model Evaluation v3 - OpenAI API Version")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, required=True,
                       help="OpenAI model name (e.g., 'gpt-4', 'gpt-3.5-turbo', 'gpt-4o')")
    parser.add_argument("--max_tokens", type=int, default=800,
                       help="Maximum tokens for model response (increased for COT)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--max_retries", type=int, default=3,
                       help="Maximum number of retry attempts for failed requests")
    parser.add_argument("--retry_delay", type=float, default=1.0,
                       help="Delay between retries in seconds")
    parser.add_argument("--max_workers", type=int, default=5,
                       help="Maximum number of concurrent API requests")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Number of preference pairs to process in parallel per batch")
    
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
    
    args = parser.parse_args()
    
    # Set random seed
    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        print(f"Random seed set to: {args.random_seed}")
    
    # Generate output filename if not provided
    if not args.output_file:
        model_name = args.model_name.lower().replace('-', '_').replace('.', '_')
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
        
        args.output_file = f"{model_name}_v3_openai_{setting_suffix}_results.json"
    
    print("=== Unified Reward Model Evaluation v3 - OpenAI API with Batch Processing ===")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_file}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Max workers: {args.max_workers}")
    print(f"Batch size: {args.batch_size}")
    
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
        evaluation_type = 'simple_prompting_v3_openai'
    elif args.full_wvs_only:
        evaluation_type = 'full_wvs_only_v3_openai'
    elif args.full_style_only:
        evaluation_type = 'full_style_only_v3_openai'
    elif args.full_combined:
        evaluation_type = 'full_combined_v3_openai'
    elif args.cot_only:
        evaluation_type = 'cot_only_v3_openai'
    else:
        evaluation_type = 'all_settings_v3_openai'

    # Initialize data manager and evaluation engine
    print(f"\n=== Loading Data ===")
    data_manager = DataManagerV3()
    evaluation_engine = EvaluationEngineV3(data_manager)
    
    # Show evaluation count estimates for all settings (helpful for planning)
    print(f"\n=== Evaluation Count Estimates ===")
    all_estimates = get_evaluation_count_estimate(argparse.Namespace(
        simple_only=True, full_wvs_only=True, full_style_only=True, 
        full_combined=True, cot_only=True, all_settings=True
    ), data_manager)
    
    for setting, count in all_estimates.items():
        est_time = estimate_time_per_evaluation(args) * count
        print(f"  {setting:15s}: {count:8,} evaluations (~{format_duration(est_time)})")
    
    print(f"\n  📌 Your selection: {evaluation_type}")
    if args.max_evaluations:
        print(f"  📌 Limited to: {args.max_evaluations:,} evaluations (testing mode)")
    print()
    
    # Generate preference pairs
    print(f"\n=== Generating Preference Pairs ===")
    
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
        test_questions = [evaluation_engine.synthetic_data[i]['question_id'] 
                        for i in range(min(3, len(evaluation_engine.synthetic_data)))]
        test_profiles = [evaluation_engine.user_profiles['user_profiles'][i]['user_profile_id'] 
                       for i in range(min(6, len(evaluation_engine.user_profiles['user_profiles'])))]
        
        preference_pairs = evaluation_engine.generate_all_preference_pairs(
            question_ids=test_questions,
            user_profile_ids=test_profiles,
            setting_name=setting_name
        )
        print("Generated test subset with setting-specific filtering")
    else:
        # Generate preference pairs using setting-specific filtering
        preference_pairs = evaluation_engine.generate_all_preference_pairs(
            setting_name=setting_name
        )
        print(f"Generated preference pairs for setting: {setting_name}")
    
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
        print(f"Remaining evaluations: {len(remaining_pairs)}")
        preference_pairs = remaining_pairs
        
        if not preference_pairs:
            print("All evaluations already completed!")
            return
    
    # Initialize OpenAI client
    print(f"\n=== Initializing OpenAI Client ===")
    try:
        # Get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            return
        
        openai_client = AsyncOpenAI(api_key=api_key)
        print("OpenAI client initialized successfully")
        
        # Test the client with a simple request
        test_response = await openai_client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        print("OpenAI API connection verified")
        
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return
    
    # Initialize results storage
    evaluation_results = existing_results.get('evaluation_results', []) if existing_results else []
    
    # Create output data structure
    output_data = {
        'metadata': {
            'model_name': args.model_name,
            'evaluation_type': evaluation_type,
            'total_preferences': len(preference_pairs) + len(evaluation_results),
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'max_workers': args.max_workers,
            'random_seed': args.random_seed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'in_progress'
        },
        'evaluation_results': evaluation_results,
        'metrics': {}
    }
    
    # Calculate estimated time
    print(f"\n=== Evaluation Time Estimation ===")
    estimated_time_per_eval = estimate_time_per_evaluation(args)
    total_evaluations = len(preference_pairs)
    total_estimated_time = total_evaluations * estimated_time_per_eval
    
    print(f"Total evaluations: {total_evaluations:,}")
    print(f"Estimated time per evaluation: {estimated_time_per_eval:.2f} seconds")
    print(f"Estimated total time: {format_duration(total_estimated_time)}")
    
    # Run evaluations with batch processing
    print(f"\n=== Running Evaluations ===")
    print(f"Processing {len(preference_pairs)} preference pairs in batches of {args.batch_size}")
    
    try:
        user_profile_cache = {}  # Cache user profiles to avoid repeated loading
        start_time = time.time()
        
        # Process items in batches
        total_items = len(preference_pairs)
        
        for batch_start in range(0, total_items, args.batch_size):
            batch_end = min(batch_start + args.batch_size, total_items)
            batch_pairs = preference_pairs[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//args.batch_size + 1}/{(total_items-1)//args.batch_size + 1} "
                  f"(items {batch_start+1}-{batch_end}/{total_items})")
            
            # Process batch asynchronously
            async def process_batch():
                tasks = []
                for preference_pair in batch_pairs:
                    # Get user profile from cache or load it
                    user_profile_id = preference_pair['user_profile_id']
                    if user_profile_id not in user_profile_cache:
                        user_profile_cache[user_profile_id] = data_manager.get_user_profile_by_id(user_profile_id)
                    user_profile = user_profile_cache[user_profile_id]
                    
                    task = evaluate_preference_pair_all_settings_async(
                        preference_pair, data_manager, openai_client, args, user_profile
                    )
                    tasks.append(task)
                
                # Execute all evaluations in the batch concurrently
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and handle exceptions
                successful_results = []
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        print(f"Error evaluating preference pair {batch_pairs[i].get('preference_id', 'unknown')}: {result}")
                        print(f"Traceback: {traceback.format_exc()}")
                    else:
                        successful_results.append(result)
                
                return successful_results
            
            # Run the batch
            batch_results = await process_batch()
            evaluation_results.extend(batch_results)
            
            # Calculate performance stats
            current_time = time.time()
            elapsed_time = current_time - start_time
            completed_evals = len(evaluation_results)
            
            if elapsed_time > 0:
                eval_rate = completed_evals / elapsed_time
                remaining_evals = total_evaluations - completed_evals
                remaining_time = remaining_evals / eval_rate if eval_rate > 0 else 0
                
                # Calculate recent accuracy (for simple setting if available)
                if evaluation_results:
                    recent_results = evaluation_results[-min(20, len(evaluation_results)):]  # Last 20 results
                    simple_correct = sum(1 for r in recent_results if r.get('simple_correct', False))
                    recent_accuracy = simple_correct / len(recent_results) if recent_results else 0
                    
                    print(f"Completed batch {batch_start//args.batch_size + 1}. "
                          f"Total: {completed_evals}/{total_evaluations} "
                          f"({completed_evals/total_evaluations:.1%}) "
                          f"Rate: {eval_rate:.1f}/s "
                          f"ETA: {format_duration(remaining_time)} "
                          f"Acc: {recent_accuracy:.1%}")
                else:
                    print(f"Completed batch {batch_start//args.batch_size + 1}. "
                          f"Total: {completed_evals}/{total_evaluations} "
                          f"Rate: {eval_rate:.1f}/s")
            
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
    print(f"Total evaluations: {metrics['total_evaluations']}")
    
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
    asyncio.run(main()) 