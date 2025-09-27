# MimiTalk: Dual-Agent AI Interview System - Research Implementation

## Context

This repository contains the technical implementation of **MimiTalk**, a dual-agent constitutional AI framework designed for scalable and ethical conversational data collection in qualitative research. This system was developed as part of academic research on revolutionizing qualitative research methodologies through Human-AI collaboration.

## Research Publication

This implementation supports the research paper:
**"MimiTalk: Revolutionizing Qualitative Research with Dual-Agent AI"**
*Authors: Fengming Liu (UCL), Shubin Yu (HEC Paris)*

The system has been empirically validated through three comprehensive studies involving 20 participants for usability evaluation, comparative analysis of 121 AI interviews against 1,271 human interviews, and interdisciplinary researcher evaluation with 10 bilingual researchers.

## System Overview

This repository contains a sophisticated dual-agent AI interview system that combines two specialized AI agents to conduct high-quality, automated interviews for qualitative research contexts. The system leverages constitutional AI principles to optimize both interview quality and operational costs while maintaining ethical compliance.

## System Architecture

### Constitutional Dual-Agent Architecture

- **Supervisor Agent**: Uses Claude Sonnet 4.0 for strategic-level interview quality control, ethical compliance monitoring, and real-time directional guidance
- **Responder Agent**: Uses GPT-5 or Claude Haiku for natural conversational generation with intelligent model selection based on interview type
- **Constitutional AI Integration**: Embedded ethical compliance and quality control without sacrificing conversational naturalness

### Research-Validated Features

- **Three Execution Modes**: Background (recommended for cost efficiency), Parallel (balanced speed-quality), and Sequential (highest quality)
- **Intelligent Cost Optimization**: Smart supervision triggering reduces API costs by up to 60% while maintaining research quality
- **Qualitative Research Focus**: Specifically designed for structured and semi-structured interviews in social science research
- **Multi-language Support**: Validated for interviews in English, Chinese, French, and Norwegian
- **Empirical Validation**: Outperforms human interviews in information richness, semantic coherence, and result stability
- **Ethical Compliance**: Constitutional AI framework ensures research ethics and participant safety

## Execution Modes

### 1. Background Mode (Recommended)

- Supervisor analysis runs during user speech
- Optimized for cost efficiency and response speed
- Cached analysis results to reduce API calls

### 2. Parallel Mode

- Supervisor and responder agents run simultaneously
- Balanced approach between speed and quality
- Results are fused for optimal responses

### 3. Sequential Mode

- Traditional step-by-step processing
- Highest quality but slower execution
- Suitable for critical interviews

## Installation & Setup

### Prerequisites

- Python 3.8+
- OpenAI API key (for GPT models)
- Anthropic API key (for Claude models)

### Environment Variables

```bash
# Required API Keys
export OPENAI_API_KEY="your-openai-key"
export CLAUDE_API_KEY="your-claude-key"

# System Configuration
export ENABLE_DUAL_MODEL="true"
export SUPERVISOR_MODEL="claude-sonnet-4-20250514"
export AI_MODEL="gpt-5"

# Cost Optimization
export SUPERVISOR_CALL_FREQUENCY="3"
export ENABLE_SMART_SUPERVISION="true"
```

### Dependencies

```bash
pip install openai anthropic asyncio
```

## Usage Example

```python
import asyncio
from Demo import generate_ai_response, background_supervisor

async def run_interview():
    # Define interview outline
    outline = """
    1. Personal background introduction
    2. Work experience sharing
    3. Skills and expertise discussion
    4. Future plans outlook
    """
  
    # Configure interview parameters
    interview_config = {
        "outline": outline,
        "history": [],
        "interview_type": "SEMI_STRUCTURED",
        "interview_language": "ENGLISH",
        "interview_uuid": "demo-001",
        "supervision_mode": "FLEXIBLE"
    }
  
    # Generate AI interviewer response
    response, model_info = await generate_ai_response(
        text=None,  # Initial greeting
        **interview_config
    )
  
    print(f"AI Interviewer: {response}")
    print(f"Model Info: {model_info}")

# Run the example
asyncio.run(run_interview())
```

## Cost Optimization Features

### Smart Supervision Triggers

- **Frequency Control**: Supervisor only called every N exchanges (default: 3)
- **Quality Detection**: Automatic triggering when conversation issues are detected
- **Topic Transition**: Smart detection of interview phase changes
- **Critical Moments**: Always active during initial conversation phases

### Performance Statistics

The system provides detailed cost tracking:

- API call counts
- Token usage optimization
- Response time metrics
- Cache hit rates

## Configuration Options

### Model Selection

- **Structured Interviews**: Claude Haiku (precise outline following)
- **Semi-structured Interviews**: GPT-5 (flexible conversation)
- **Supervisor Analysis**: Claude Sonnet (strategic guidance)

### Temperature Settings

- **Structured Mode**: 0.1 (high consistency)
- **Flexible Mode**: 0.7 (natural conversation)

### Token Limits

- **Supervisor**: 300 tokens (cost-optimized)
- **Structured Response**: 500 tokens
- **Flexible Response**: 1000 tokens

## Research-Proven Benefits

1. **Enhanced Data Quality**: Empirically validated to outperform human interviews in information richness and semantic coherence
2. **Reduced Interview Anxiety**: Participants report lower anxiety levels compared to traditional human-led interviews
3. **Cost Efficiency**: Intelligent triggering reduces research costs by up to 60% while maintaining quality standards
4. **Scalability for Research**: Asynchronous architecture enables large-scale qualitative data collection
5. **Candid Expression**: AI interviews elicit more honest responses on sensitive topics like academic integrity
6. **Technical Insights**: Capable of generating technical discussions comparable to expert interviews
7. **Ethical Compliance**: Constitutional AI framework ensures participant safety and research ethics

## Technical Highlights

- **Asynchronous Processing**: Full async/await implementation for optimal performance
- **Error Handling**: Robust fallback mechanisms between AI providers
- **Caching System**: Intelligent result caching with TTL management
- **Logging**: Comprehensive logging for debugging and monitoring
- **API Abstraction**: Unified interface for different AI providers

## License

This project is part of an AI interview automation research initiative.

## Contributing

For questions or contributions, please refer to the research team contact information.

## Research Findings

### Key Empirical Results

- **Information Richness**: AI interviews significantly outperform human interviews using NLP metrics including DeBERTa tokenizer-based information entropy
- **Semantic Coherence**: Higher cross-turn semantic similarity compared to traditional interview methods
- **Participant Experience**: Reduced interview anxiety and enhanced candid expression on sensitive topics
- **Technical Discussions**: Capability to elicit technical insights comparable to expert human interviews
- **Cost Efficiency**: Up to 60% reduction in research costs while maintaining quality standards

### Boundary Conditions

- **Human Advantages**: Human interviews excel in capturing cultural context and emotional nuances
- **AI Strengths**: Better performance in technical discussions and sensitive topic exploration
- **Optimal Use Cases**: Semi-structured interviews, large-scale data collection, sensitive topic research

## Citation

If you use this system in your research, please cite:

```bibtex
@article{liu2025mimitalk,
  title={MimiTalk: Revolutionizing Qualitative Research with Dual-Agent AI},
  author={Liu, Fengming and Yu, Shubin},
  journal={Conference Proceedings},
  year={2025},
  institution={University College London and HEC Paris}
}
```

## Research Ethics

This system implements constitutional AI principles to ensure:

- Participant privacy and data protection
- Ethical compliance in research contexts
- Transparent AI-human interaction boundaries
- Quality control without compromising naturalness

---

*This system represents a significant advancement in AI-powered qualitative research, validated through comprehensive empirical studies and designed specifically for academic research contexts.*
