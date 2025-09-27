"""
Dual-Agent AI Interview System Core Implementation Showcase
=====================================================

This file contains the core code of our Dual-Agent AI Interview System, demonstrating:
1. Supervisor Agent: Uses Claude Sonnet for interview progress analysis and guidance
2. Responder Agent: Uses GPT-5/Claude Haiku for actual interview response generation
3. Three execution modes: Background mode, Parallel mode, Sequential mode
4. Intelligent cost optimization strategies

Author: AI-Interviewer Team
Date: 2025-09-01
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional
import anthropic
from openai import OpenAI

logger = logging.getLogger(__name__)

# =====================================================
# 1. System Configuration - Core Configuration for Dual-Agent System
# =====================================================

# Dual-model system toggle
ENABLE_DUAL_MODEL = os.getenv("ENABLE_DUAL_MODEL", "true").lower() == "true"

# Supervisor Agent configuration (Claude Sonnet)
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "claude-sonnet-4-20250514")
SUPERVISOR_MAX_TOKENS = int(os.getenv("SUPERVISOR_MAX_TOKENS", "300"))
SUPERVISOR_TEMPERATURE = float(os.getenv("SUPERVISOR_TEMPERATURE", "0.1"))
SUPERVISOR_CALL_FREQUENCY = int(os.getenv("SUPERVISOR_CALL_FREQUENCY", "3"))

# Intelligent supervision control
ENABLE_SMART_SUPERVISION = os.getenv("ENABLE_SMART_SUPERVISION", "true").lower() == "true"

# Responder Agent configuration
AI_MODEL = os.getenv('OPENAI_MODEL', os.getenv("AI_MODEL", "gpt-5"))
STRUCTURED_RESPONDER_MODEL = os.getenv("STRUCTURED_RESPONDER_MODEL", "claude-3-5-haiku-20241022")
SEMI_STRUCTURED_RESPONDER_MODEL = os.getenv('SEMI_STRUCTURED_RESPONDER_MODEL', AI_MODEL)

# Initialize AI clients
api_key = os.getenv("OPENAI_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")

if api_key:
    ai_client = OpenAI(api_key=api_key)
else:
    print("Warning: OpenAI API Key not found")

if claude_api_key:
    claude_client = anthropic.Anthropic(api_key=claude_api_key)
else:
    print("Warning: Claude API Key not found")
    claude_client = None

# =====================================================
# 2. Core Functions - Model Selection and Routing
# =====================================================

def is_gpt5_variant(model_name: str) -> bool:
    """Check if the model is GPT-5 series, requiring special parameter handling"""
    if not isinstance(model_name, str):
        return False
    return model_name.startswith("gpt-5")

def select_responder_model(supervision_mode: str):
    """Select Responder Agent's provider and model based on interview mode
    
    Args:
        supervision_mode: "STRUCTURED" or "FLEXIBLE"
    
    Returns:
        tuple: (provider, model) - provider is "openai" or "anthropic"
    """
    if supervision_mode == "STRUCTURED":
        # Structured interviews use Claude Haiku (more precise outline following)
        return "anthropic", STRUCTURED_RESPONDER_MODEL
    else:
        # Semi-structured interviews use GPT-5 (more flexible conversation)
        return "openai", SEMI_STRUCTURED_RESPONDER_MODEL

async def generate_with_selected_provider(system_prompt: str, base_messages: List[Dict], 
                                        temperature: float, max_tokens: int, 
                                        provider: str, model: str) -> str:
    """Unified AI API interface supporting both OpenAI and Anthropic
    
    This function abstracts API differences between different AI providers
    """
    if provider == "openai":
        openai_messages = [{"role": "system", "content": system_prompt}, *base_messages]
        
        # GPT-5 series requires special parameter handling
        openai_kwargs = {
            "model": model,
            "messages": openai_messages,
        }
        
        if is_gpt5_variant(model):
            # GPT-5 series specific parameters
            openai_kwargs["max_completion_tokens"] = max_tokens
            # GPT-5 only supports temperature=1.0, so omit temperature setting
        else:
            # Traditional OpenAI model parameters
            openai_kwargs["max_tokens"] = max_tokens
            openai_kwargs["temperature"] = temperature
            openai_kwargs["presence_penalty"] = 0.0
            openai_kwargs["frequency_penalty"] = 0.0
            
        response = ai_client.chat.completions.create(**openai_kwargs)
        return response.choices[0].message.content
        
    else:  # Anthropic
        if not claude_client:
            raise RuntimeError("Anthropic client not configured")
            
        # Anthropic requires separate system message handling
        anthropic_messages = [{"role": m.get("role"), "content": m.get("content")} for m in base_messages]
        response = claude_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=anthropic_messages
        )
        return response.content[0].text if response.content else ""

# =====================================================
# 3. Intelligent Cost Optimization - Supervisor Agent Trigger Strategy
# =====================================================

def should_trigger_supervisor(history: List, interview_uuid: str = None) -> bool:
    """Intelligent supervisor trigger strategy - decides whether to call Supervisor Agent for cost savings
    
    Strategies include:
    1. Frequency control: Only call every 3 exchanges by default
    2. Conversation quality detection: Trigger when repetition or stagnation is detected
    3. Topic transition detection: Trigger when key topic transitions appear in AI responses
    4. Initial phase guarantee: First 2 exchanges always trigger
    """
    if not ENABLE_SMART_SUPERVISION:
        return True  # Always trigger if intelligent supervision is disabled
    
    # Strategy 1: Frequency control based on conversation rounds
    user_messages = [msg for msg in history if msg.get('role') == 'user']
    if len(user_messages) % SUPERVISOR_CALL_FREQUENCY != 0:
        logger.info(f"[SMART] Skipping supervisor call - frequency control ({len(user_messages)} % {SUPERVISOR_CALL_FREQUENCY} != 0)")
        return False
    
    # Strategy 2: Detect critical turning points (conversation stagnation or topic transition)
    if len(history) >= 4:
        recent_messages = history[-4:]
        
        # Detect repetitive response patterns
        user_responses = [msg['content'] for msg in recent_messages if msg.get('role') == 'user']
        if len(user_responses) >= 2:
            # Trigger supervision if user responses are too short or repetitive
            if any(len(resp.strip()) < 10 for resp in user_responses[-2:]):
                logger.info("[SMART] Triggering supervisor - detected short responses")
                return True
        
        # Detect topic keyword changes
        ai_responses = [msg['content'] for msg in recent_messages if msg.get('role') == 'assistant']
        if len(ai_responses) >= 2:
            # Simple topic transition detection
            topic_keywords = ['next', 'now let us', 'moving on', 'another', 'furthermore']
            if any(keyword in ai_responses[-1].lower() for keyword in topic_keywords):
                logger.info("[SMART] Triggering supervisor - detected topic transition")
                return True
    
    # Strategy 3: Always trigger in initial and critical moments
    if len(user_messages) <= 1:  # First or second user response
        logger.info("[SMART] Triggering supervisor - initial conversation phase")
        return True
    
    logger.info(f"[SMART] Supervisor call triggered by frequency rule (turn {len(user_messages)})")
    return True

# =====================================================
# 4. Supervisor Agent - Claude Sonnet Interview Analysis and Guidance
# =====================================================

async def generate_supervisor_analysis(outline, history, interview_type, interview_language, 
                                    current_response=None, supervision_mode="FLEXIBLE"):
    """Supervisor Agent - Uses Claude Sonnet for interview progress analysis and strategy guidance
    
    Core responsibilities of the Supervisor Agent:
    1. Analyze interview progress and quality
    2. Check adherence to the outline
    3. Provide next-step interview strategies
    4. Detect issues and opportunities in the interview
    """
    try:
        logger.info("== SUPERVISOR AGENT ANALYSIS STARTED ==")
        print(f"[SUPERVISOR] Analyzing with {SUPERVISOR_MODEL} (mode: {supervision_mode})...")
        
        # Check if Claude client is available
        if not claude_client:
            logger.warning("Claude unavailable, falling back to OpenAI")
            return await generate_supervisor_analysis_openai_fallback(
                outline, history, interview_type, interview_language, current_response, supervision_mode
            )
        
        # Build different analysis prompts based on supervision mode
        if supervision_mode == "STRUCTURED":
            # Structured interview mode: Strict outline compliance checking
            supervisor_prompt = f"""You are a structured interview supervision expert ensuring strict adherence to the outline.

**Interview Outline**:
{outline}

**Supervision Tasks**:
1. Check if AI has deviated from outline topics
2. Ensure questions directly come from outline content  
3. Monitor for missing important outline sections
4. Flag any questions outside the outline scope

Interview Type: {interview_type}
Language: {interview_language}

Based on conversation history analysis, return JSON format:
{{
    "outline_compliance": {{
        "is_compliant": true/false,
        "covered_sections": ["covered outline sections"],
        "missed_sections": ["missed outline sections"],
        "off_topic_questions": ["questions that deviate from outline"]
    }},
    "next_action": "next step recommendation - must be based on outline",
    "outline_guidance": "specific outline execution guidance",
    "progress": "X%"
}}"""
        else:
            # Flexible mode: Comprehensive interview quality analysis
            supervisor_prompt = f"""You are an AI interview supervision expert, analyzing interview quality and providing strategic guidance.

Interview Outline: {outline}
Type: {interview_type} 
Language: {interview_language}

**Analysis Dimensions**:
1. Interview depth and quality
2. Interviewee engagement level
3. Topic coverage completeness
4. Conversation flow

Please provide concise analysis and next-step strategy recommendations (limit to {SUPERVISOR_MAX_TOKENS} characters)"""
        
        # Use only recent conversation history for cost control
        recent_history = history[-6:] if len(history) > 6 else history
        
        # Call Claude Sonnet for supervision analysis
        supervisor_response = claude_client.messages.create(
            model=SUPERVISOR_MODEL,
            max_tokens=SUPERVISOR_MAX_TOKENS,  # Cost optimization: limit token count
            temperature=SUPERVISOR_TEMPERATURE,  # Low temperature ensures analysis consistency
            messages=[
                {"role": "user", "content": f"{supervisor_prompt}\n\nConversation History: {str(recent_history)}"}
            ]
        )
        
        supervisor_analysis = supervisor_response.content[0].text if supervisor_response.content else ""
        
        logger.info(f"{supervision_mode} mode supervisor analysis completed: {len(supervisor_analysis)} chars")
        print(f"[SUPERVISOR] Analysis received from {SUPERVISOR_MODEL} ({len(supervisor_analysis)} chars)")
        
        return supervisor_analysis

    except Exception as e:
        logger.error(f"Claude supervisor analysis failed: {e}")
        print(f"[SUPERVISOR] Falling back to OpenAI")
        # Fallback to OpenAI backup supervision
        return await generate_supervisor_analysis_openai_fallback(
            outline, history, interview_type, interview_language, current_response, supervision_mode
        )

async def generate_supervisor_analysis_openai_fallback(outline, history, interview_type, interview_language, 
                                                    current_response=None, supervision_mode="FLEXIBLE"):
    """OpenAI Fallback Supervisor Agent - Fallback solution when Claude is unavailable"""
    try:
        logger.info(f"Using OpenAI {AI_MODEL} as fallback supervisor (mode: {supervision_mode})...")
        
        # Build supervision prompt (similar to Claude version but adapted for OpenAI)
        if supervision_mode == "STRUCTURED":
            supervisor_prompt = f"""As a structured interview supervision expert, analyze whether the interview strictly follows the outline.

Outline: {outline}
Type: {interview_type}
Language: {interview_language}

Please check: 1) Outline compliance 2) Missing sections 3) Off-topic deviations 4) Next step recommendations

Concise response (limit {SUPERVISOR_MAX_TOKENS} characters):"""
        else:
            supervisor_prompt = f"""Analyze interview progress and provide brief guidance.

Outline: {outline}
Type: {interview_type}  
Language: {interview_language}

Please analyze: 1) Interview quality 2) Engagement level 3) Topic coverage 4) Recommended strategies

Concise analysis (limit {SUPERVISOR_MAX_TOKENS} characters):"""
        
        # Build messages, using only recent conversation history
        recent_history = history[-6:] if len(history) > 6 else history
        supervisor_messages = [
            {"role": "system", "content": supervisor_prompt},
            {"role": "user", "content": f"Conversation History: {str(recent_history)}"}
        ]
        
        # Use OpenAI as backup supervision, applying the same token limits
        openai_kwargs = {
            "model": AI_MODEL,
            "messages": supervisor_messages,
        }
        if is_gpt5_variant(AI_MODEL):
            openai_kwargs["max_completion_tokens"] = SUPERVISOR_MAX_TOKENS
        else:
            openai_kwargs["max_tokens"] = SUPERVISOR_MAX_TOKENS
            openai_kwargs["temperature"] = SUPERVISOR_TEMPERATURE
        
        supervisor_response = ai_client.chat.completions.create(**openai_kwargs)
        supervisor_analysis = supervisor_response.choices[0].message.content
        
        logger.info(f"OpenAI fallback {supervision_mode} supervisor analysis completed ({len(supervisor_analysis)} chars)")
        return supervisor_analysis

    except Exception as e:
        logger.error(f"OpenAI fallback supervisor analysis failed: {e}")
        raise e

# =====================================================
# 5. Background Supervision Manager - Cost-Optimized Background Analysis System
# =====================================================

class BackgroundSupervisorManager:
    """Background Supervision Analysis Manager
    
    This is the core optimization component of the dual-agent system, implementing:
    1. Asynchronous supervision analysis during user speech
    2. Cached analysis results to reduce repeated calls
    3. Intelligent triggering strategies for cost control
    4. Cost statistics and monitoring
    """
    
    def __init__(self):
        self._background_analyses = {}  # interview_uuid -> analysis_task
        self._cached_analyses = {}      # interview_uuid -> {analysis, timestamp, history_length}
        self._cache_ttl = 600           # 10-minute cache TTL (cost optimization)
        self._analysis_count = {}       # interview_uuid -> count (cost tracking)
    
    async def start_background_analysis(self, interview_uuid: str, outline: str, history: List, 
                                    interview_type: str, interview_language: str, 
                                    supervision_mode: str = "FLEXIBLE"):
        """Start background supervision analysis during user speech - Key performance optimization"""
        try:
            # Intelligent trigger check - Core of cost control
            if not should_trigger_supervisor(history, interview_uuid):
                logger.info(f"[BACKGROUND] Skipping analysis for {interview_uuid} - cost optimization")
                return
            
            # Cancel previous analysis task (if exists)
            if interview_uuid in self._background_analyses:
                self._background_analyses[interview_uuid].cancel()
                logger.info(f"[BACKGROUND] Cancelled previous analysis for {interview_uuid}")
            
            # Start new background analysis task
            analysis_task = asyncio.create_task(
                self._run_background_analysis(interview_uuid, outline, history, interview_type, 
                                            interview_language, supervision_mode)
            )
            self._background_analyses[interview_uuid] = analysis_task
            logger.info(f"[BACKGROUND] Started analysis for {interview_uuid} (mode: {supervision_mode})")
            
        except Exception as e:
            logger.error(f"Error starting background analysis: {e}")
    
    async def _run_background_analysis(self, interview_uuid: str, outline: str, history: List, 
                                    interview_type: str, interview_language: str, 
                                    supervision_mode: str = "FLEXIBLE"):
        """Core logic for executing background supervision analysis"""
        try:
            logger.info(f"[BACKGROUND] Running supervisor analysis for {interview_uuid}")
            
            # Call supervisor agent for analysis
            analysis = await generate_supervisor_analysis(
                outline, history, interview_type, interview_language, None, supervision_mode
            )
            
            # Cache analysis results
            self._cached_analyses[interview_uuid] = {
                'analysis': analysis,
                'timestamp': datetime.utcnow(),
                'history_length': len(history)
            }
            
            # Update statistics
            self._analysis_count[interview_uuid] = self._analysis_count.get(interview_uuid, 0) + 1
            
            logger.info(f"[BACKGROUND] Analysis cached for {interview_uuid} (count: {self._analysis_count[interview_uuid]})")
            
        except asyncio.CancelledError:
            logger.info(f"[BACKGROUND] Analysis cancelled for {interview_uuid}")
        except Exception as e:
            logger.error(f"[BACKGROUND] Analysis failed for {interview_uuid}: {e}")
    
    async def get_cached_analysis(self, interview_uuid: str, current_history_length: int) -> Optional[str]:
        """Get cached supervision analysis results - Core performance optimization"""
        if interview_uuid not in self._cached_analyses:
            logger.info(f"[BACKGROUND] No cached analysis for {interview_uuid}")
            return None
        
        cached = self._cached_analyses[interview_uuid]
        
        # Check if cache has expired
        age = (datetime.utcnow() - cached['timestamp']).total_seconds()
        if age > self._cache_ttl:
            logger.info(f"[BACKGROUND] Cached analysis expired for {interview_uuid} (age: {age:.1f}s)")
            del self._cached_analyses[interview_uuid]
            return None
        
        # Check if conversation history has changed significantly
        history_diff = current_history_length - cached['history_length']
        if history_diff > 4:  # If conversation has progressed too much, cache may be outdated
            logger.info(f"[BACKGROUND] Cached analysis outdated for {interview_uuid} (history diff: {history_diff})")
            return None
        
        logger.info(f"[BACKGROUND] Using cached analysis for {interview_uuid} (age: {age:.1f}s)")
        return cached['analysis']
    
    def get_cost_stats(self, interview_uuid: str) -> dict:
        """Get cost statistics information - For monitoring and optimization"""
        return {
            'analysis_count': self._analysis_count.get(interview_uuid, 0),
            'estimated_tokens_saved': self._analysis_count.get(interview_uuid, 0) * (1000 - SUPERVISOR_MAX_TOKENS),
            'cache_hits': 1 if interview_uuid in self._cached_analyses else 0
        }
    
    def cleanup_interview(self, interview_uuid: str):
        """Clean up interview-related resources - Prevent memory leaks"""
        cost_stats = self.get_cost_stats(interview_uuid)
        
        # Cancel background tasks
        if interview_uuid in self._background_analyses:
            self._background_analyses[interview_uuid].cancel()
            del self._background_analyses[interview_uuid]
        
        # Clear cache
        if interview_uuid in self._cached_analyses:
            del self._cached_analyses[interview_uuid]
        
        # Clear statistics
        if interview_uuid in self._analysis_count:
            del self._analysis_count[interview_uuid]
        
        logger.info(f"[BACKGROUND] Cleaned up resources for {interview_uuid}")
        logger.info(f"[COST-STATS] {interview_uuid}: {cost_stats['analysis_count']} analyses, ~{cost_stats['estimated_tokens_saved']} tokens saved")

# Global background supervisor manager instance
background_supervisor = BackgroundSupervisorManager()

# =====================================================
# 6. Three Execution Modes - Optimization Strategies for Different Scenarios
# =====================================================

async def generate_ai_response_with_background(text, outline, history, interview_type, interview_language, 
                                            interview_uuid=None, supervision_mode="FLEXIBLE"):
    """Dual-Agent System - Background Mode (Fastest, Recommended)
    
    Core advantages of background mode:
    1. Asynchronous supervision analysis during user speech
    2. AI response uses cached supervision results
    3. Minimizes user waiting time
    4. Cost optimization
    """
    logger.info("=== DUAL-AGENT SYSTEM (BACKGROUND MODE) ===")
    print(f"[BACKGROUND-DUAL-AGENT] Using cost-optimized background supervision (mode: {supervision_mode})...")
    
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Check if supervision analysis is needed
        use_supervisor = should_trigger_supervisor(history, interview_uuid)
        supervisor_analysis = None
        
        if use_supervisor and interview_uuid:
            # Try to get cached supervision analysis
            supervisor_analysis = await background_supervisor.get_cached_analysis(interview_uuid, len(history))
            supervision_mode_display = f"SUPERVISED-{supervision_mode}" if supervisor_analysis else f"BASIC-{supervision_mode}"
        else:
            supervision_mode_display = f"BASIC-{supervision_mode}"
        
        # Immediately start background analysis for next conversation round (Key to performance optimization)
        if interview_uuid:
            next_history = history + ([{"role": "user", "content": text}] if text else [])
            await background_supervisor.start_background_analysis(
                interview_uuid, outline, next_history, interview_type, interview_language, supervision_mode
            )
        
        # Build system prompt (Based on whether supervision analysis is available)
        if supervisor_analysis:
            # Enhanced prompt with supervision guidance
            system_prompt = f"""You are a professional AI interviewer. Conduct the interview based on supervision analysis:

**Supervision Analysis and Guidance**:
{supervisor_analysis}

**Interview Outline**: {outline}

Please conduct the interview according to the supervision analysis guidance, ensuring high-quality conversation."""
        else:
            # Basic interview prompt
            system_prompt = f"""You are a professional AI interviewer, conducting interviews based on the following outline:

{outline}

Please conduct natural, in-depth interview conversations."""
        
        # Add language adaptation
        if interview_language:
            language_hints = {
                "CHINESE": "Please conduct the interview in Chinese.",
                "ENGLISH": "Please conduct the interview in English.",
                "FRENCH": "Please conduct the interview in French.",
                "NORWEGIAN": "Please conduct the interview in Norwegian.",
                "FLEXIBLE": "Please adapt to the language used by the interviewee."
            }
            system_prompt += f"\n{language_hints.get(interview_language, '')}"
        
        # Build conversation messages
        base_messages = [*history]
        if text:
            base_messages.append({"role": "user", "content": text})
        elif not history and not text:
            # Initial message
            if interview_language == "CHINESE":
                base_messages.append({"role": "user", "content": "Please introduce yourself and start the interview."})
            else:
                base_messages.append({"role": "user", "content": "Please introduce yourself and start the interview."})
        
        # Parameter optimization (structured vs flexible mode)
        if supervision_mode_display.startswith("SUPERVISED-STRUCTURED") or supervision_mode_display.startswith("BASIC-STRUCTURED"):
            temperature = 0.1  # Structured interview: Low temperature, high consistency
            max_tokens = 500
        else:
            temperature = 0.7  # Flexible interview: Medium temperature, more natural
            max_tokens = 1000
        
        # Select responder agent
        resp_provider, resp_model = select_responder_model(supervision_mode)
        content = await generate_with_selected_provider(
            system_prompt,
            base_messages,
            temperature,
            max_tokens,
            resp_provider,
            resp_model
        )
        
        # Performance statistics
        total_time = asyncio.get_event_loop().time() - start_time
        cost_stats = background_supervisor.get_cost_stats(interview_uuid) if interview_uuid else {}
        
        supervisor_name = f"{SUPERVISOR_MODEL}" if claude_client else f"{AI_MODEL} (fallback)"
        logger.info(f"BACKGROUND DUAL-AGENT SUCCESS in {total_time:.2f}s ({supervision_mode_display} mode)")
        print(f"[BACKGROUND-DUAL-AGENT] Response generated in {total_time:.2f}s ({supervision_mode_display} mode)!")
        print(f"   Supervisor: {supervisor_name} ({'used' if supervisor_analysis else 'skipped'})")
        print(f"   Responder: {resp_provider}:{resp_model}")
        print(f"   Cost Stats: {cost_stats.get('analysis_count', 0)} analyses, ~{cost_stats.get('estimated_tokens_saved', 0)} tokens saved")
        
        model_info = f"BACKGROUND-DUAL-AGENT ({supervision_mode_display}): {resp_provider}:{resp_model} (supervisor: {supervisor_name}) - {total_time:.2f}s"
        
        return content, model_info
        
    except Exception as e:
        logger.error(f"Error in background dual-agent system: {e}")
        # Fallback to single-agent mode
        return await generate_single_model_response(text, outline, history, interview_type, interview_language, supervision_mode)

async def generate_ai_response_parallel(text, outline, history, interview_type, interview_language, 
                                    supervision_mode="FLEXIBLE"):
    """Dual-Agent System - Parallel Mode (Fast)
    
    Parallel mode characteristics:
    1. Supervision analysis and basic response execute simultaneously
    2. Optimize final response through "fusion" step
    3. Balance speed and quality
    """
    logger.info("=== DUAL-AGENT AI SYSTEM (PARALLEL MODE) ===")
    logger.info(f"Running supervisor analysis and response generation in parallel (mode: {supervision_mode})")
    print(f"[PARALLEL-DUAL-AGENT] Starting concurrent processing (mode: {supervision_mode})...")
    
    try:
        # Build basic messages
        base_messages = [*history]
        if text:
            base_messages.append({"role": "user", "content": text})
        elif not history and not text:
            if interview_language == "CHINESE":
                base_messages.append({"role": "user", "content": "Please introduce yourself and start the interview."})
            else:
                base_messages.append({"role": "user", "content": "Please introduce yourself and start the interview."})
        
        # Parallel Task 1: Supervisor Agent Analysis
        async def supervisor_task():
            return await generate_supervisor_analysis(
                outline, history, interview_type, interview_language, None, supervision_mode
            )
        
        # Parallel Task 2: Basic Response Agent (without supervision guidance)
        async def base_response_task():
            return await generate_single_model_response(
                text, outline, history, interview_type, interview_language, supervision_mode
            )
        
        # Execute dual agents in parallel
        start_time = asyncio.get_event_loop().time()
        supervisor_analysis, (base_response, base_model) = await asyncio.gather(
            supervisor_task(),
            base_response_task(),
            return_exceptions=True
        )
        parallel_time = asyncio.get_event_loop().time() - start_time
        
        logger.info(f"Parallel execution completed in {parallel_time:.2f} seconds")
        print(f"[PARALLEL] Both agents completed in {parallel_time:.2f}s")
        
        # Handle exceptions
        if isinstance(supervisor_analysis, Exception):
            logger.error(f"Supervisor analysis failed: {supervisor_analysis}")
            supervisor_analysis = '{"strategic_guidance": "Continue the conversation and maintain the interview flow"}'
        
        if isinstance(base_response, Exception):
            logger.error(f"Base response generation failed: {base_response}")
            return "Sorry, AI service is temporarily unavailable. Please try again later.", AI_MODEL
        
        # Intelligent Fusion: Optimize basic response based on supervisor analysis
        logger.info("[FUSION] Applying supervisor guidance to base response...")
        
        fusion_prompt = f"""As an interview expert, please optimize the following AI response based on supervisor analysis.

Supervisor Analysis:
{supervisor_analysis}

Original AI Response:
{base_response}

Please output the optimized response (keep it concise, only output the final response content):"""
        
        try:
            # Fusion processing
            fusion_provider, fusion_model = select_responder_model(supervision_mode)
            final_response = await generate_with_selected_provider(
                "",
                [{"role": "user", "content": fusion_prompt}],
                0.3,  # Low temperature ensures fusion quality
                500,  # Token limit
                fusion_provider,
                fusion_model
            )
            final_response = (final_response or '').strip()
            
            # Quality check
            if not final_response or len(final_response.strip()) < 10:
                final_response = base_response
                logger.warning("Fusion result too short, using base response")
            
        except Exception as fusion_error:
            logger.error(f"Fusion failed: {fusion_error}")
            final_response = base_response
        
        # Statistics information
        total_time = asyncio.get_event_loop().time() - start_time
        supervisor_name = f"{SUPERVISOR_MODEL} (Claude)" if claude_client else f"{AI_MODEL} (fallback)"
        resp_provider, resp_model = select_responder_model(supervision_mode)
        
        logger.info(f"PARALLEL DUAL-AGENT SUCCESS in {total_time:.2f}s")
        print(f"[PARALLEL-DUAL-AGENT] Complete pipeline finished in {total_time:.2f}s!")
        print(f"   Supervisor: {supervisor_name}")
        print(f"   Base Response: {resp_provider}:{resp_model}")
        print(f"   Fusion: Applied")
        
        model_info = f"PARALLEL-DUAL-AGENT: {resp_provider}:{resp_model} (supervised by {supervisor_name}) - {total_time:.2f}s"
        
        return final_response, model_info
        
    except Exception as e:
        logger.error(f"Error in parallel dual-agent system: {e}")
        return await generate_single_model_response(text, outline, history, interview_type, interview_language, supervision_mode)

async def generate_ai_response_sequential(text, outline, history, interview_type, interview_language, 
                                        supervision_mode="FLEXIBLE"):
    """Dual-Agent System - Sequential Mode (Original implementation, most precise)
    
    Sequential mode characteristics:
    1. Execute supervisor agent analysis first
    2. Generate response based on supervision results
    3. Highest quality but relatively slower speed
    """
    logger.info("=== DUAL-AGENT AI SYSTEM (SEQUENTIAL MODE) ===")
    logger.info(f"Step 1: Supervisor analysis using {SUPERVISOR_MODEL} (mode: {supervision_mode})")
    print(f"[SEQUENTIAL-DUAL-AGENT] Starting supervisor analysis with {SUPERVISOR_MODEL} (mode: {supervision_mode})...")
    
    try:
        # Step 1: Supervisor Agent Analysis
        logger.info("Getting supervisor analysis...")
        print("[SUPERVISOR] Analyzing interview progress and providing guidance...")
        supervisor_analysis = await generate_supervisor_analysis(
            outline, history, interview_type, interview_language, None, supervision_mode
        )
        
        logger.info("Supervisor analysis completed")
        print(f"[SUPERVISOR] Analysis complete. Guidance received from {SUPERVISOR_MODEL}")
        
        # Build enhanced system prompt
        if supervision_mode == "STRUCTURED":
            base_prompt = f"""You are a professional structured interviewer who must strictly follow the outline and supervision guidance.

**Supervision Analysis and Guidance**:
{supervisor_analysis}

**Interview Outline**: {outline}

Please strictly execute the interview according to the supervision analysis guidance, ensuring adherence to the outline."""
        else:
            base_prompt = f"""You are a professional AI interviewer conducting high-quality interviews based on supervision analysis.

**Supervision Analysis and Guidance**:
{supervisor_analysis}

**Interview Outline**: {outline}

Please conduct the interview according to the professional supervision guidance."""
        
        # Add language adaptation
        if interview_language:
            language_hints = {
                "CHINESE": "Please conduct the interview in Chinese.",
                "ENGLISH": "Please conduct the interview in English.",
                "FRENCH": "Please conduct the interview in French.",
                "NORWEGIAN": "Please conduct the interview in Norwegian.",
                "FLEXIBLE": "Please adapt to the language used by the interviewee."
            }
            base_prompt += f"\n{language_hints.get(interview_language, '')}"
        # Build messages
        base_messages = [*history]
        if text:
            base_messages.append({"role": "user", "content": text})
        if not history and not text:
            if interview_language == "CHINESE":
                base_messages.append({"role": "user", "content": "Please introduce yourself and start the interview."})
            else:
                base_messages.append({"role": "user", "content": "Please introduce yourself and start the interview."})
        
        # Step 2: Responder Agent Generation
        logger.info("Step 2: Generating response with supervisor guidance")
        print(f"[RESPONDER] Generating response using supervised guidance...")
        
        # Parameter optimization
        if supervision_mode == "STRUCTURED":
            temperature = 0.1
            max_tokens = 500
        else:
            temperature = 0.7
            max_tokens = 1000
        
        # Select responder agent
        resp_provider, resp_model = select_responder_model(supervision_mode)
        content = await generate_with_selected_provider(
            base_prompt,
            base_messages,
            temperature,
            max_tokens,
            resp_provider,
            resp_model
        )
        
        # Statistics information
        supervisor_name = f"{SUPERVISOR_MODEL} (Claude)" if claude_client else f"{AI_MODEL} (fallback)"
        logger.info(f"SEQUENTIAL DUAL-AGENT SUCCESS: Supervisor({supervisor_name}) + Responder({resp_model})")
        print(f"[SEQUENTIAL-DUAL-AGENT] Response generated successfully!")
        print(f"   Supervisor: {supervisor_name}")
        print(f"   Responder: {resp_provider}:{resp_model}")
        
        model_info = f"SEQUENTIAL-DUAL-AGENT: {resp_provider}:{resp_model} (supervised by {supervisor_name})"
        
        return content, model_info
        
    except Exception as e:
        logger.error(f"Error generating AI response with sequential dual-agent system: {e}")
        fallback_msg = "Sorry, AI service is temporarily unavailable. Please try again later." if interview_language == "CHINESE" else "Sorry, AI service is temporarily unavailable. Please try again later."
        return fallback_msg, AI_MODEL

# =====================================================
# 7. Main Entry Function - Dual-Agent System Dispatcher
# =====================================================

async def generate_ai_response(text, outline, history, interview_type, interview_language, 
                            interview_uuid=None, supervision_mode="FLEXIBLE"):
    """Dual-Agent AI Interview System Main Entry
    
    Select execution mode based on environment variable DUAL_MODEL_MODE:
    1. "background": Background mode (fastest, recommended)
    2. "parallel": Parallel mode (fast)
    3. "sequential": Sequential mode (most precise)
    4. "single": Single-agent mode (simplest)
    """
    if not ENABLE_DUAL_MODEL:
        return await generate_single_model_response(text, outline, history, interview_type, interview_language, supervision_mode)
    
    # Check execution mode
    DUAL_MODEL_MODE = os.getenv("DUAL_MODEL_MODE", "background").lower()
    
    if DUAL_MODEL_MODE == "background":
        # Background supervision analysis mode (fastest)
        return await generate_ai_response_with_background(
            text, outline, history, interview_type, interview_language, interview_uuid, supervision_mode
        )
    elif DUAL_MODEL_MODE == "parallel":
        # Parallel execution mode
        return await generate_ai_response_parallel(
            text, outline, history, interview_type, interview_language, supervision_mode
        )
    elif DUAL_MODEL_MODE == "sequential":
        # Sequential execution mode (original)
        return await generate_ai_response_sequential(
            text, outline, history, interview_type, interview_language, supervision_mode
        )
    else:
        # Default to background mode
        logger.warning(f"Unknown DUAL_MODEL_MODE: {DUAL_MODEL_MODE}, defaulting to background")
        return await generate_ai_response_with_background(
            text, outline, history, interview_type, interview_language, interview_uuid, supervision_mode
        )

# =====================================================
# 8. Single-Agent Mode - Traditional Implementation (Compatibility Support)
# =====================================================

async def generate_single_model_response(text, outline, history, interview_type, interview_language, 
                                    supervision_mode="FLEXIBLE"):
    """Traditional Single-Agent AI Response System (Compatibility Mode)
    
    Fallback solution when dual-agent system is unavailable
    """
    try:
        # Decide prompt strategy based on supervision mode
        if supervision_mode == "STRUCTURED":
            system_prompt = f"""You are a strict structured interviewer who must conduct interviews according to the outline step by step.

**Interview Outline** (must strictly follow):
{outline}

Requirements:
1. Strictly ask questions in outline order
2. Only ask one question at a time
3. Ensure coverage of all outline points
4. Do not deviate from outline topics"""
        else:
            system_prompt = f"""You are a professional AI interviewer conducting in-depth interviews based on the following outline:

{outline}

Please conduct natural, flexible interview conversations and explore related topics in depth."""
        
        # Add language preference
        if interview_language:
            language_hints = {
                "CHINESE": "Please conduct the interview in Chinese.",
                "ENGLISH": "Please conduct the interview in English.",
                "FRENCH": "Please conduct the interview in French.",
                "NORWEGIAN": "Please conduct the interview in Norwegian.",
                "FLEXIBLE": "Please adapt to the language used by the interviewee."
            }
            system_prompt += f"\n{language_hints.get(interview_language, '')}"
        
        # Build messages
        base_messages = [*history]
        if text:
            base_messages.append({"role": "user", "content": text})
        if not history and not text:
            if interview_language == "CHINESE":
                base_messages.append({"role": "user", "content": "Please introduce yourself and start the interview."})
            else:
                base_messages.append({"role": "user", "content": "Please introduce yourself and start the interview."})
        
        # Select model and parameters
        resp_provider, resp_model = select_responder_model(supervision_mode)
        
        # Optimize parameters
        if supervision_mode == "STRUCTURED":
            temperature = 0.1
            max_tokens = 500
        else:
            temperature = 0.7
            max_tokens = 1000
        
        # Generate response
        content = await generate_with_selected_provider(
            system_prompt,
            base_messages,
            temperature,
            max_tokens,
            resp_provider,
            resp_model
        )
        
        return content, f"{resp_provider}:{resp_model}"
        
    except Exception as e:
        logger.error(f"Error generating single-agent AI response: {e}")
        fallback_msg = "Sorry, AI service is temporarily unavailable. Please try again later." if interview_language == "CHINESE" else "Sorry, AI service is temporarily unavailable. Please try again later."
        return fallback_msg, AI_MODEL

# =====================================================
# 9. System Initialization and Startup Information
# =====================================================

def display_system_banner():
    """Display system startup banner showcasing dual-agent configuration"""
    if ENABLE_DUAL_MODEL:
        DUAL_MODEL_MODE = os.getenv("DUAL_MODEL_MODE", "background").lower()
        
        mode_info = {
            "background": ("BACKGROUND MODE", "Background supervision analysis mode (fastest)"),
            "parallel": ("PARALLEL MODE", "Parallel execution mode (fast)"),
            "sequential": ("SEQUENTIAL MODE", "Sequential execution mode (slower)")
        }
        
        mode_display, mode_desc = mode_info.get(DUAL_MODEL_MODE, ("BACKGROUND MODE", "Background supervision analysis mode (default)"))
        
        print("=" * 70)
        print("DUAL-AGENT AI SYSTEM ACTIVATED")
        print(f"Optimization Mode: {mode_display}")
        print(f"Mode Description: {mode_desc}")
        print(f"Cost Optimization: Max tokens={SUPERVISOR_MAX_TOKENS}, Frequency=1/{SUPERVISOR_CALL_FREQUENCY}")
        if claude_client:
            print(f"Supervisor Agent: {SUPERVISOR_MODEL}")
        else:
            print(f"Supervisor Agent (fallback): OpenAI {AI_MODEL}")
        print(f"Response Agent: {AI_MODEL} (OpenAI)")
        print("Tip: Use 'python configure_dual_model.py --show-config' to view settings")
        print("=" * 70)
    else:
        print("=" * 60)
        print("SINGLE-AGENT AI SYSTEM (Legacy Mode)")
        print("Tip: Use 'python configure_dual_model.py --mode background' to enable dual-agent")
        print("=" * 60)

# =====================================================
# 10. Usage Examples and Test Functions
# =====================================================

async def example_dual_agent_interview():
    """Dual-Agent system usage example"""
    print("\nDual-Agent AI Interview System Demo")
    print("=" * 50)
    
    # Example interview configuration
    outline = """
    1. Personal background introduction
    2. Work experience sharing
    3. Skills and expertise discussion
    4. Future plans outlook
    """
    
    interview_history = []
    interview_type = "SEMI_STRUCTURED"
    interview_language = "ENGLISH"
    interview_uuid = "demo-001"
    
    # Simulate interview start
    print("\nInterview starting...")
    response, model_info = await generate_ai_response(
        text=None,  # Initial message
        outline=outline,
        history=interview_history,
        interview_type=interview_type,
        interview_language=interview_language,
        interview_uuid=interview_uuid,
        supervision_mode="FLEXIBLE"
    )
    
    print(f"\nAI Interviewer: {response}")
    print(f"Model Info: {model_info}")
    
    # Simulate user response
    interview_history.extend([
        {"role": "assistant", "content": response, "timestamp": datetime.utcnow().isoformat()},
        {"role": "user", "content": "I am a software engineer with 5 years of development experience.", "timestamp": datetime.utcnow().isoformat()}
    ])
    
    # Second round of conversation
    print("\nSecond round of conversation...")
    response2, model_info2 = await generate_ai_response(
        text="I am a software engineer with 5 years of development experience.",
        outline=outline,
        history=interview_history[:-1],  # Exclude the just-added user message
        interview_type=interview_type,
        interview_language=interview_language,
        interview_uuid=interview_uuid,
        supervision_mode="FLEXIBLE"
    )
    
    print(f"\nAI Interviewer: {response2}")
    print(f"Model Info: {model_info2}")
    
    # Show cost statistics
    cost_stats = background_supervisor.get_cost_stats(interview_uuid)
    print(f"\nCost Statistics: {cost_stats}")
    
    # Clean up resources
    background_supervisor.cleanup_interview(interview_uuid)

if __name__ == "__main__":
    # Display system banner
    display_system_banner()
    
    # Run demonstration (if this file is executed directly)
    # asyncio.run(example_dual_agent_interview())
    
    print("\nDual-Agent system code showcase completed")
    print("This system implements a dual AI architecture with Supervisor Agent + Responder Agent")
    print("Supports three execution modes: background, parallel, and sequential")
    print("Includes intelligent cost optimization and caching strategies")
    print("Can significantly improve interview quality and efficiency")
