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
import uuid
from datetime import datetime, timedelta

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_management.data_manager_v3 import DataManagerV3
from evaluation_engine.evaluation_engine_v3 import EvaluationEngineV3

try:
    from google import genai
    from google.genai import types
    import google.auth
    from google.oauth2 import service_account
except ImportError:
    print("Warning: Google GenAI SDK not installed. Please install with: pip install google-genai")
    genai = None
    types = None

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
        """Response schema for evaluations"""
        reasoning: Optional[str] = Field(default=None, description="Step-by-step analysis of user preferences and response alignment (COT only)")
        final_choice: Literal["A", "B"] = Field(description="The final answer: either A or B")
        confidence: Optional[float] = Field(default=None, ge=0, le=1, description="Optional confidence score between 0 and 1")



def create_simple_prompt(prompt, completion_a, completion_b):
    """Create simple prompt without context"""
    return f"""You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON.

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
    """Create prompt with WVS context"""
    return f"""You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON.

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
    """Create prompt with style context"""
    return f"""You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON.

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
    """Create prompt with WVS and style context, prioritizing values"""
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
    
    return f"""You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON.

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
    """Create prompt prioritizing style over values"""
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
    
    return f"""You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON.

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
    """Create prompt with both contexts, neutral priority"""
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
    
    return f"""You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON.

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
    """Create COT prompt with WVS context"""
    return f"""You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON.

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
    """Create COT prompt with style context"""
    return f"""You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON.

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
    """Create COT prompt with WVS and style context, prioritizing values"""
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
    
    return f"""You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON.

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
    """Create COT prompt with WVS and style context, prioritizing style"""
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
    
    return f"""You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON.

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
    """Create COT prompt with WVS and style context, neutral priority"""
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
    
    return f"""You are an AI assistant that evaluates which answer (A or B) a user would prefer. Always respond with valid JSON.

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

# ===== JSON PARSING (reused from original) =====

def _extract_reasoning_from_text(response_text: str) -> str:
    """Extract reasoning from response text when JSON parsing fails"""
    try:
        import re
        reasoning_patterns = [
            r'"reasoning"\s*:\s*"([^"]+)"',
            r'"reasoning"\s*:\s*\'([^\']+)\'',
            r'reasoning["\']?\s*:\s*["\']([^"\']+)["\']',
        ]
        
        for pattern in reasoning_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if matches:
                reasoning = matches[-1].strip()
                reasoning = reasoning.replace('\\n', '\n').replace('\\"', '"')
                return reasoning
        
        return "No reasoning extracted"
    except Exception:
        return "Reasoning extraction failed"

def parse_json_response(response_text: str) -> dict:
    """Parse JSON response from evaluation with error handling"""
    if not response_text or not response_text.strip():
        return {"final_choice": "ERROR", "reasoning": None}
    
    # Clean the response
    cleaned_response = response_text.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]
    cleaned_response = cleaned_response.strip()
    
    try:
        response_data = json.loads(cleaned_response)
        
        # Validate using Pydantic schema
        if BaseModel is not None:
            validated_response = COTResponse(**response_data)
            return {"final_choice": validated_response.final_choice, "reasoning": validated_response.reasoning}
        else:
            final_choice = response_data.get('final_choice', '').upper()
            reasoning = response_data.get('reasoning', '')
            if final_choice in ['A', 'B']:
                return {"final_choice": final_choice, "reasoning": reasoning}
            else:
                return {"final_choice": "ERROR", "reasoning": reasoning}
                
    except (json.JSONDecodeError, ValidationError):
        # Try regex extraction
        import re
        choice_patterns = [
            r'"final_choice"\s*:\s*"([AB])"',
            r'final_choice["\']?\s*:\s*["\']?([AB])["\']?',
        ]
        
        for pattern in choice_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                choice = matches[-1].upper()
                if choice in ['A', 'B']:
                    reasoning = _extract_reasoning_from_text(response_text)
                    return {"final_choice": choice, "reasoning": reasoning}
        
        return {"final_choice": "ERROR", "reasoning": None}

# ===== VERTEX AI BATCH PROCESSING =====

class VertexAIBatchProcessor:
    """Handle Google GenAI batch processing"""
    
    def __init__(self, project_id: str, location: str = "global"):
        self.project_id = project_id
        self.location = location
        self.client = None
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize Google GenAI client"""
        try:
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )
            print(f"Google GenAI client initialized for project {self.project_id}")
        except Exception as e:
            print(f"Failed to initialize Google GenAI client: {e}")
            raise
    
    async def call_genai_model(self, prompt_text: str, model_name: str, 
                              max_tokens: int = 800, temperature: float = 0.0, seed: int = 0) -> str:
        """Make API call to Google GenAI"""
        try:
            # Create contents for the request
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=prompt_text)]
                )
            ]
            
            # Configure generation parameters
            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=1,
                seed=seed,
                max_output_tokens=max_tokens,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="OFF"
                    )
                ],
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT", 
                    "properties": {
                        "final_choice": {"type": "STRING", "enum": ["A", "B"]},
                        "reasoning": {"type": "STRING"}
                    },
                    "required": ["final_choice"]
                },
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                ),
            )
            
            # Generate response
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model_name,
                contents=contents,
                config=generate_content_config
            )
            
            # Extract text from response
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()
            
            # Handle cases where response exists but no text
            if response:
                # Check for candidates and finish reasons
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    
                    # Check finish reason
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = str(candidate.finish_reason)
                        
                        if 'MAX_TOKENS' in finish_reason or 'LENGTH' in finish_reason:
                            return '{"final_choice": "ERROR", "reasoning": "Response truncated due to model context limit"}'
                        elif 'SAFETY' in finish_reason:
                            return '{"final_choice": "ERROR", "reasoning": "Response blocked by safety filters"}'
                        elif 'RECITATION' in finish_reason:
                            return '{"final_choice": "ERROR", "reasoning": "Response blocked due to recitation"}'
                        elif 'OTHER' in finish_reason:
                            return '{"final_choice": "ERROR", "reasoning": "Response blocked for other reasons"}'
                        else:
                            return f'{{"final_choice": "ERROR", "reasoning": "Unexpected finish reason: {finish_reason}"}}'
                
                # Log usage metadata if available
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    if hasattr(usage, 'prompt_token_count') and hasattr(usage, 'total_token_count'):
                        prompt_tokens = usage.prompt_token_count
                        total_tokens = usage.total_token_count
                        output_tokens = total_tokens - prompt_tokens
                        return response.text.strip(), prompt_tokens, output_tokens
                
                return '{"final_choice": "ERROR", "reasoning": "No text content in response"}'
            
            return '{"final_choice": "ERROR", "reasoning": "Empty response from model"}'
                
        except Exception as e:
            error_msg = str(e)
            print(f"Error calling GenAI model: {error_msg}")
            
            # Handle specific error types
            if "quota" in error_msg.lower() or "limit" in error_msg.lower():
                return '{"final_choice": "ERROR", "reasoning": "API quota exceeded"}'
            elif "safety" in error_msg.lower():
                return '{"final_choice": "ERROR", "reasoning": "Content blocked by safety filters"}'
            else:
                return f'{{"final_choice": "ERROR", "reasoning": "API error: {error_msg[:100]}"}}'
    
    async def submit_batch_job(self, prompts_and_settings: List[Tuple], model_name: str, 
                              max_tokens: int = 800, temperature: float = 0.0, 
                              seed: int = 0, max_concurrent: int = 10) -> Dict:
        """Submit requests to Vertex AI with concurrency control"""
        
        print(f"Processing {len(prompts_and_settings)} prompts with max {max_concurrent} concurrent requests")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_request(prompt_text: str, is_cot: bool, setting_name: str):
            async with semaphore:
                try:
                    result = await self.call_genai_model(
                        prompt_text, model_name, max_tokens, temperature, seed
                    )
                    
                    # Handle result - could be just text or tuple with token counts
                    if isinstance(result, tuple):
                        response_text, prompt_tokens, response_tokens = result
                    else:
                        response_text = result
                        # Fallback token estimates
                        prompt_tokens = len(prompt_text.split()) * 1.3
                        response_tokens = len(response_text.split()) * 1.3 if response_text else 10
                    
                    return {
                        "custom_id": setting_name,
                        "response": response_text,
                        "status": "success",
                        "prompt_tokens": int(prompt_tokens),
                        "response_tokens": int(response_tokens)
                    }
                    
                except Exception as e:
                    return {
                        "custom_id": setting_name,
                        "response": None,
                        "status": "error",
                        "error": str(e),
                        "prompt_tokens": int(len(prompt_text.split()) * 1.3),
                        "response_tokens": 0
                    }
        
        # Create tasks for all requests
        tasks = []
        for prompt_text, is_cot, setting_name in prompts_and_settings:
            task = process_single_request(prompt_text, is_cot, setting_name)
            tasks.append(task)
        
        # Execute all tasks with progress tracking
        results = []
        completed = 0
        total = len(tasks)
        
        # Process in chunks to show progress
        chunk_size = max(1, min(50, total // 10))  # Show progress every 10% or 50 requests
        for i in range(0, total, chunk_size):
            chunk = tasks[i:i + chunk_size]
            chunk_results = await asyncio.gather(*chunk, return_exceptions=True)
            
            for result in chunk_results:
                if isinstance(result, Exception):
                    results.append({
                        "custom_id": f"error_{completed}",
                        "response": None,
                        "status": "error",
                        "error": str(result),
                        "prompt_tokens": 0,
                        "response_tokens": 0
                    })
                else:
                    results.append(result)
                completed += 1
            
            if total > 10:  # Only show progress for larger batches
                print(f"Progress: {completed}/{total} ({completed/total:.1%})")
        
        job_id = f"reward-model-eval-{uuid.uuid4().hex[:8]}"
        return {"job_id": job_id, "results": results}

# ===== MAIN EVALUATION FUNCTION =====

async def evaluate_preference_pair_all_settings_batch(preference_pairs, data_manager, vertex_processor, 
                                                     args):
    """Evaluate multiple preference pairs using batch processing"""
    
    all_prompts_and_settings = []
    pair_to_prompts = {}  # Map preference pairs to their prompts
    
    # Generate all prompts for all preference pairs
    for pair_idx, preference_pair in enumerate(preference_pairs):
        # Get basic information
        user_profile_id = preference_pair['user_profile_id']
        value_profile_id = preference_pair['value_profile_id']
        
        prompt = preference_pair['prompt']
        preferred_completion = preference_pair['preferred_completion']
        non_preferred_completion = preference_pair['non_preferred_completion']
        
        # Randomly decide whether preferred_completion goes in position A or B
        prefer_in_position_a = random.choice([True, False])
        
        if prefer_in_position_a:
            completion_a = preferred_completion
            completion_b = non_preferred_completion
        else:
            completion_a = non_preferred_completion
            completion_b = preferred_completion
        
        # Get user profile
        user_profile = data_manager.get_user_profile_by_id(user_profile_id)
        
        # Generate contexts
        wvs_context = data_manager.generate_full_wvs_context(value_profile_id)
        style_context = data_manager.generate_full_style_context(user_profile['style_profile'])
        
        # For combined contexts, randomize the order
        wvs_first = random.choice([True, False])
        
        # Store metadata for this pair
        pair_to_prompts[pair_idx] = {
            'preference_pair': preference_pair,
            'prefer_in_position_a': prefer_in_position_a,
            'wvs_first': wvs_first,
            'prompts': []
        }
        
        # Generate prompts based on selected settings
        prompts_for_pair = []
        
        # 1. Simple prompting
        if args.simple_only or args.all_settings:
            simple_prompt = create_simple_prompt(prompt, completion_a, completion_b)
            prompts_for_pair.append((simple_prompt, False, f'simple_{pair_idx}'))
        
        # 2. Full WVS context only
        if args.full_wvs_only or args.all_settings:
            full_wvs_prompt = create_full_wvs_only_prompt(prompt, completion_a, completion_b, wvs_context)
            prompts_for_pair.append((full_wvs_prompt, False, f'full_wvs_only_{pair_idx}'))
        
        # 3. Full style context only
        if args.full_style_only or args.all_settings:
            full_style_prompt = create_full_style_only_prompt(prompt, completion_a, completion_b, style_context)
            prompts_for_pair.append((full_style_prompt, False, f'full_style_only_{pair_idx}'))
        
        # 4. Full combined contexts
        if args.full_combined or args.all_settings:
            wvs_style_prefer_wvs = create_full_wvs_style_prefer_wvs_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            prompts_for_pair.append((wvs_style_prefer_wvs, False, f'full_wvs_style_prefer_wvs_{pair_idx}'))
            
            wvs_style_prefer_style = create_full_wvs_style_prefer_style_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            prompts_for_pair.append((wvs_style_prefer_style, False, f'full_wvs_style_prefer_style_{pair_idx}'))
            
            wvs_style_neutral = create_full_wvs_style_neutral_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            prompts_for_pair.append((wvs_style_neutral, False, f'full_wvs_style_neutral_{pair_idx}'))
        
        # 5. COT versions
        if args.cot_only or args.all_settings:
            wvs_cot = create_full_wvs_only_cot_prompt(prompt, completion_a, completion_b, wvs_context)
            prompts_for_pair.append((wvs_cot, True, f'full_wvs_only_cot_{pair_idx}'))
            
            style_cot = create_full_style_only_cot_prompt(prompt, completion_a, completion_b, style_context)
            prompts_for_pair.append((style_cot, True, f'full_style_only_cot_{pair_idx}'))
            
            wvs_style_cot_prefer_wvs = create_full_wvs_style_cot_prefer_wvs_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            prompts_for_pair.append((wvs_style_cot_prefer_wvs, True, f'full_wvs_style_cot_prefer_wvs_{pair_idx}'))
            
            wvs_style_cot_prefer_style = create_full_wvs_style_cot_prefer_style_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            prompts_for_pair.append((wvs_style_cot_prefer_style, True, f'full_wvs_style_cot_prefer_style_{pair_idx}'))
            
            wvs_style_cot_neutral = create_full_wvs_style_cot_neutral_prompt(
                prompt, completion_a, completion_b, wvs_context, style_context, wvs_first)
            prompts_for_pair.append((wvs_style_cot_neutral, True, f'full_wvs_style_cot_neutral_{pair_idx}'))
        
        pair_to_prompts[pair_idx]['prompts'] = prompts_for_pair
        all_prompts_and_settings.extend(prompts_for_pair)
    
    # Submit batch job
    print(f"Submitting batch job with {len(all_prompts_and_settings)} total prompts...")
    
    batch_results = await vertex_processor.submit_batch_job(
        all_prompts_and_settings, args.model_name, args.max_tokens, args.temperature, args.seed, max_concurrent=10
    )
    
    # Process results
    results_by_id = {r["custom_id"]: r for r in batch_results["results"]}
    
    evaluation_results = []
    
    for pair_idx, pair_data in pair_to_prompts.items():
        preference_pair = pair_data['preference_pair']
        prefer_in_position_a = pair_data['prefer_in_position_a']
        wvs_first = pair_data['wvs_first']
        correct_answer = 'A' if prefer_in_position_a else 'B'
        
        # Create result dictionary
        result = {
            'preference_id': preference_pair.get('preference_id', 
                f"{preference_pair['user_profile_id']}_{preference_pair['question_id']}_{preference_pair['style_family']}_{preference_pair['combination_type']}"),
            'user_profile_id': preference_pair['user_profile_id'],
            'value_profile_id': preference_pair['value_profile_id'],
            'question_id': preference_pair['question_id'],
            'style_family': preference_pair.get('style_family', 'unknown'),
            'combination_type': preference_pair.get('combination_type', 'unknown'),
            'preference_rule': preference_pair.get('preference_rule', 'unknown'),
            'quadrant': preference_pair.get('quadrant', 'unknown'),
            'style_code': preference_pair.get('style_code', 'unknown'),
            'correct_answer': correct_answer,
            'preferred_in_position_a': prefer_in_position_a,
            'wvs_context_first': wvs_first,
            'preferred_completion_key': preference_pair['preferred_completion_key'],
            'non_preferred_completion_key': preference_pair['non_preferred_completion_key'],
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Process responses for this pair
        for prompt_text, is_cot, setting_id in pair_data['prompts']:
            if setting_id in results_by_id:
                batch_result = results_by_id[setting_id]
                
                if batch_result['status'] == 'success':
                    response_text = batch_result['response']
                    parsed_response = parse_json_response(response_text)
                    
                    setting_name = setting_id.split('_')[:-1]  # Remove pair index
                    setting_name = '_'.join(setting_name)
                    
                    result[f'{setting_name}_response'] = parsed_response['final_choice']
                    result[f'{setting_name}_reasoning'] = parsed_response.get('reasoning')
                    result[f'{setting_name}_correct'] = (parsed_response['final_choice'] == correct_answer 
                                                        if parsed_response['final_choice'] != "ERROR" else False)
                    
                else:
                    setting_name = setting_id.split('_')[:-1]
                    setting_name = '_'.join(setting_name)
                    
                    result[f'{setting_name}_response'] = "ERROR"
                    result[f'{setting_name}_reasoning'] = None
                    result[f'{setting_name}_correct'] = False
        
        evaluation_results.append(result)
    
    return evaluation_results

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
        raise ValueError(f"Invalid question group: {question_group}")

def estimate_evaluation_size(args, data_manager, question_group):
    """Estimate the size of the evaluation"""
    question_ids = get_question_ids_for_group(data_manager, question_group)
    
    # Estimate based on question count and settings
    num_questions = len(question_ids)
    
    # Rough estimates for different settings
    settings_count = 1
    if args.all_settings:
        settings_count = 11  # Simple + WVS + Style + 3*Combined + 5*COT
    elif args.cot_only:
        settings_count = 5   # COT versions
    elif args.full_combined:
        settings_count = 3   # Combined variants
    
    # Estimate based on evaluation engine's expected counts
    evaluation_engine = EvaluationEngineV3(data_manager)
    total_evaluations = len(evaluation_engine.generate_all_preference_pairs(question_ids=question_ids))
    
    if args.max_evaluations:
        total_evaluations = min(total_evaluations, args.max_evaluations)
    
    total_calls = total_evaluations * settings_count
    
    return {
        'total_evaluations': total_evaluations,
        'total_calls': total_calls
    }

# ===== RESULT MANAGEMENT =====

def save_incremental_results(output_data, output_file):
    """Save results incrementally to avoid data loss"""
    temp_file = output_file + ".temp"
    try:
        with open(temp_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
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
    
    # Determine available settings
    if results:
        sample_result = results[0]
        available_settings = []
        for key in sample_result.keys():
            if key.endswith('_response') and not key.endswith('_reasoning'):
                setting_name = key[:-9]
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
        }
    }
    
    # Calculate accuracy for each setting
    for setting in available_settings:
        correct_count = sum(1 for r in results if r.get(f'{setting}_correct', False))
        successful_count = sum(1 for r in results if r.get(f'{setting}_response') not in [None, "ERROR"])
        overall_metrics[f'{setting}_accuracy'] = correct_count / successful_count if successful_count > 0 else 0
        overall_metrics[f'{setting}_success_rate'] = successful_count / total_count if total_count > 0 else 0
    
    return overall_metrics

# ===== MAIN FUNCTION =====

async def main():
    # Check if Google GenAI is available
    if genai is None or types is None:
        print("Error: Google GenAI SDK is not installed. Please install with: pip install google-genai")
        return
        
    parser = argparse.ArgumentParser(description="Unified Reward Model Evaluation v3 - Google GenAI SDK Version")
    
    # Model and infrastructure arguments
    parser.add_argument("--model_name", type=str, required=True,
                       choices=['gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-1.5-pro', 'gemini-1.5-flash'],
                       help="Google GenAI model name")
    parser.add_argument("--project_id", type=str, required=True,
                       help="Google Cloud project ID")
    parser.add_argument("--location", type=str, default="global",
                       help="Google GenAI location")
    
    # Question group selection
    parser.add_argument("--questions", type=str, required=True,
                       choices=['key_wvs_only', 'wvq_only', 'both'],
                       help="Which question group to evaluate: key_wvs_only (14 questions), wvq_only (10 questions), or both (24 questions)")
    
    # Evaluation setting arguments
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
                              help="Run COT versions only")
    setting_group.add_argument("--all_settings", action="store_true",
                              help="Run all evaluation settings")
    
    # Generation parameters
    parser.add_argument("--max_tokens", type=int, default=800,
                       help="Maximum tokens for model response")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for deterministic generation")
    
    # Batch processing
    parser.add_argument("--batch_size", type=int, default=50,
                       help="Number of preference pairs to process in each batch")
    
    # Output and testing
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for results")
    parser.add_argument("--max_evaluations", type=int, default=None,
                       help="Maximum number of evaluations to run (for testing)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducible randomization")
    
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
        
        args.output_file = f"{model_name}_v3_vertex_{setting_suffix}_{args.questions}_results.json"
    
    print("Unified Reward Model Evaluation v3 - Google GenAI SDK")
    print(f"Model: {args.model_name}")
    print(f"Project: {args.project_id}")
    print(f"Location: {args.location}")
    print(f"Question group: {args.questions}")
    print(f"Output: {args.output_file}")
    
    # Initialize data manager
    print("\nLoading Data")
    data_manager = DataManagerV3()
    evaluation_engine = EvaluationEngineV3(data_manager)
    
    # Get question IDs for selected group
    question_ids = get_question_ids_for_group(data_manager, args.questions)
    print(f"Selected question group: {args.questions} ({len(question_ids)} questions)")
    
    # Show evaluation size estimation
    print("\nEvaluation Size Estimation")
    size_estimate = estimate_evaluation_size(args, data_manager, args.questions)
    print(f"Estimated evaluations: {size_estimate['total_evaluations']:,}")
    print(f"Estimated API calls: {size_estimate['total_calls']:,}")
    
    # Generate preference pairs with question filtering
    print("\nGenerating Preference Pairs")
    
    if args.max_evaluations and args.max_evaluations < 100:
        # Testing mode - use subset
        test_questions = question_ids[:min(3, len(question_ids))]
        test_profiles = [evaluation_engine.user_profiles['user_profiles'][i]['user_profile_id'] 
                       for i in range(min(6, len(evaluation_engine.user_profiles['user_profiles'])))]
        
        preference_pairs = evaluation_engine.generate_all_preference_pairs(
            question_ids=test_questions,
            user_profile_ids=test_profiles
        )
        print(f"Generated test subset: {len(preference_pairs)} pairs")
    else:
        preference_pairs = evaluation_engine.generate_all_preference_pairs(
            question_ids=question_ids
        )
        print(f"Generated {len(preference_pairs)} preference pairs")
    
    # Limit evaluations if specified
    if args.max_evaluations and args.max_evaluations < len(preference_pairs):
        preference_pairs = preference_pairs[:args.max_evaluations]
        print(f"Limited to {len(preference_pairs)} evaluations for testing")
    
    # Initialize Google GenAI processor
    print("\nInitializing Google GenAI")
    try:
        vertex_processor = VertexAIBatchProcessor(args.project_id, args.location)
    except Exception as e:
        print(f"Failed to initialize Google GenAI: {e}")
        return
    
    # Load existing results
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
    
    evaluation_results = existing_results.get('evaluation_results', []) if existing_results else []
    
    # Initialize output data
    output_data = {
        'metadata': {
            'model_name': args.model_name,
            'question_group': args.questions,
            'total_preferences': len(preference_pairs) + len(evaluation_results),
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
            'seed': args.seed,
            'batch_size': args.batch_size,
            'random_seed': args.random_seed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'in_progress'
        },
        'evaluation_results': evaluation_results,
        'metrics': {}
    }
    
    # Run evaluations in batches
    print("\nRunning Evaluations")
    print(f"Processing {len(preference_pairs)} preference pairs in batches of {args.batch_size}")
    
    try:
        start_time = time.time()
        
        for batch_start in range(0, len(preference_pairs), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(preference_pairs))
            batch_pairs = preference_pairs[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//args.batch_size + 1}/{(len(preference_pairs)-1)//args.batch_size + 1}")
            
            # Process batch
            batch_results = await evaluate_preference_pair_all_settings_batch(
                batch_pairs, data_manager, vertex_processor, args
            )
            
            evaluation_results.extend(batch_results)
            
            # Update output data
            output_data['evaluation_results'] = evaluation_results
            output_data['metrics'] = calculate_metrics(evaluation_results)
            
            # Save incremental results
            save_incremental_results(output_data, args.output_file)
            
            # Progress update
            completed = len(evaluation_results)
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"Completed: {completed}/{len(preference_pairs)} ({completed/len(preference_pairs):.1%})")
            print(f"Rate: {rate:.1f} evals/sec")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
    
    # Finalize results
    print("\nFinalizing Results")
    output_data['evaluation_results'] = evaluation_results
    output_data['metrics'] = calculate_metrics(evaluation_results)
    output_data['metadata']['status'] = 'completed'
    output_data['metadata']['completed_evaluations'] = len(evaluation_results)
    
    save_incremental_results(output_data, args.output_file)
    
    # Print summary
    metrics = output_data['metrics']
    
    print("\nEvaluation Summary")
    print(f"Question group: {args.questions}")
    print(f"Total evaluations: {metrics['total_evaluations']}")
    
    # Print accuracy by setting
    print("\nAccuracy by Setting")
    for key, value in metrics.items():
        if key.endswith('_accuracy'):
            setting_name = key[:-9]
            success_rate = metrics.get(f'{setting_name}_success_rate', 1.0)
            print(f"  {setting_name}: {value:.1%} accuracy ({success_rate:.1%} success rate)")
    
    print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    asyncio.run(main()) 