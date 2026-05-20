from LLM import UnifiedLLM
import json
import os
from glob import glob
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

# %%
# Judge prompt template - evaluates ALL domains at once
JUDGE_PROMPT = '''You are an expert evaluator assessing an AI agent's decisions about domain usage and memory requirements for a given user persona.

## User Persona
{persona}

## Decisions to Evaluate
{domains_json}

## Definition of "Use This Domain"
Use should be TRUE when:
- The user's persona, interests, profession, or life circumstances suggest they would plausibly interact with an AI agent for this domain
- There is explicit or strongly implied evidence in the persona

Use should be FALSE when:
- The domain is irrelevant to the user's context, skills, interests, or life stage
- There is no reasonable basis in the persona to suggest usage

## Definition of Memory Required
Memory is required (true) when this user's usage of the domain is:
- Ongoing and accumulative (e.g., tracking progress over time)
- Connected across multiple sessions (e.g., building on past conversations)
- Tied to long-term goals or evolving personal circumstances

Memory is NOT required (false) when this user's usage of the domain is:
- One-time or ad-hoc (e.g., a single lookup with no follow-up)
- Self-contained within a single session
- Not dependent on past interactions

## CRITICAL
Evaluate both decisions based on THIS USER's specific context, not on general assumptions about the domain.
The same domain can have different use and memory_required values for different users.
The reason field is provided to help you understand the intent behind each decision — use it as evidence when judging.

## Your Task
For each domain, evaluate the following:

1. **Use Decision**: Is the use decision reasonable given this user's persona?

2. **Memory Required Decision**:
   - If use=true: evaluate whether memory_required is true or false appropriately for this user's context.
   - If use=false: memory_required must be null. If it is not null, mark as unreasonable.

## Output Format
Return a JSON array with one object per domain:
[
  {{
    "domain_name": "...",
    "use_decision": {{
      "reasonable": true/false,
      "feedback": "One sentence justifying your evaluation."
    }},
    "memory_required_decision": {{
      "reasonable": true/false,
      "feedback": "One sentence justifying your evaluation."
    }}
  }},
  ...
]
'''

# Judge models configuration
JUDGE_MODELS = {
    "openai": ["gpt-5.1", 'o3-mini'],
    # "claude": ["claude-haiku-4-5"],
    # "gemini": ["gemini-3-flash-preview"],
    # "together": ["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "Qwen/Qwen3-235B-A22B-fp8"]
}


def format_persona(persona_dict: dict) -> str:
    """Format persona dict into readable text."""
    persona_text = ''
    for key, value in persona_dict.items():
        if key == 'uuid':
            continue
        persona_text += f"\n[{key}]"
        persona_text += f"\n{value}"
        persona_text += '\n'
    return persona_text


def format_domains_for_prompt(domains: list[dict]) -> str:
    """Format all domains into a JSON string for the prompt."""
    formatted = []
    for d in domains:
        formatted.append({
            "domain_name": d.get('domain_name', ''),
            "use": d.get('use', 'N/A'),
            "memory_required": d.get('memory_required', 'N/A'),
            "reason": d.get('reason', 'N/A')
        })
    return json.dumps(formatted, indent=2, ensure_ascii=False)


def judge_all_domains(
    llm: UnifiedLLM,
    persona_dict: dict,
    domains: list[dict],
    model: str | None = None
) -> list[dict]:
    """Judge all domains in a single LLM call."""
    persona_text = format_persona(persona_dict)
    domains_json = format_domains_for_prompt(domains)
    
    prompt = JUDGE_PROMPT.format(
        persona=persona_text,
        domains_json=domains_json
    )
    
    response = llm.chat(prompt, model=model)
    
    # Parse JSON response
    try:
        results = json.loads(response.replace('```json', '').replace('```', '').strip())
        if not isinstance(results, list):
            results = [results]
    except json.JSONDecodeError:
        # Return error for all domains
        results = [{
            "domain_name": d.get('domain_name', ''),
            "parse_error": True,
            "error_message": f"Failed to parse response: {response[:500]}"
        } for d in domains]
    
    return results


def judge_file(
    filepath: str,
    provider: str,
    model: str,
    domains_to_judge: list[str] | None = None
) -> dict:
    """
    Judge all domains in a single file with ONE LLM call.
    
    Args:
        filepath: Path to the JSON file
        provider: LLM provider
        model: Model name
        domains_to_judge: Optional list of domain names to filter (if None, judge all)
    
    Returns:
        dict with judgments for each domain
    """
    llm = UnifiedLLM(provider, model)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    persona = data.get('persona', {})
    domains = data.get('domains', [])
    
    # Filter domains if specified
    if domains_to_judge:
        domains = [d for d in domains if d.get('domain_name') in domains_to_judge]
    
    results = {
        'uuid': data.get('uuid'),
        'judge_provider': provider,
        'judge_model': model,
        'domain_judgments': []
    }
    
    try:
        judgments = judge_all_domains(llm, persona, domains, model)
        
        # Match judgments with original domain data
        domain_map = {d.get('domain_name'): d for d in domains}
        for judgment in judgments:
            domain_name = judgment.get('domain_name', '')
            original = domain_map.get(domain_name, {})
            judgment['original_use'] = original.get('use')
            judgment['original_memory_required'] = original.get('memory_required')
            results['domain_judgments'].append(judgment)
            
    except Exception as e:
        results['error'] = str(e)
        for d in domains:
            results['domain_judgments'].append({
                'domain_name': d.get('domain_name', ''),
                'error': str(e)
            })
    
    return results


def run_multi_model_judgment(
    input_dir: str,
    output_dir: str,
    models: dict[str, list[str]] | None = None,
    sample_size: int | None = None,
    max_workers: int = 4
):
    """
    Run judgment across multiple LLM families.
    
    Args:
        input_dir: Directory containing generated persona metadata JSON files
        output_dir: Directory to save judgment results
        models: Dict of {provider: [model_names]} to use. If None, uses JUDGE_MODELS
        sample_size: Number of files to sample (if None, process all)
        max_workers: Number of parallel workers
    """
    models = models or JUDGE_MODELS
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files (excluding meta files)
    json_files = glob(os.path.join(input_dir, '*.json'))
    json_files = [f for f in json_files if not os.path.basename(f).startswith('_')]
    
    if sample_size:
        import random
        random.seed(42)
        json_files = random.sample(json_files, min(sample_size, len(json_files)))
    
    print(f"Processing {len(json_files)} files with {len(models)} providers")
    
    all_results = {}
    
    for provider, model_list in models.items():
        for model in model_list:
            model_key = f"{provider}_{model.replace('/', '_')}"
            print(f"\n{'='*60}")
            print(f"Running judgments with: {provider} / {model}")
            print(f"{'='*60}")
            
            model_results = []
            errors = []
            
            for filepath in tqdm(json_files, desc=f"{model_key}"):
                try:
                    result = judge_file(filepath, provider, model)
                    model_results.append(result)
                except Exception as e:
                    errors.append({
                        'file': os.path.basename(filepath),
                        'error': str(e)
                    })
            
            all_results[model_key] = {
                'provider': provider,
                'model': model,
                'results': model_results,
                'errors': errors
            }
            
            # Save intermediate results
            model_output_path = os.path.join(output_dir, f'{model_key}_judgments.json')
            with open(model_output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results[model_key], f, ensure_ascii=False, indent=2)
            
            print(f"Completed: {len(model_results)} files, {len(errors)} errors")
    
    # Save combined results
    combined_path = os.path.join(output_dir, '_all_judgments.json')
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    return all_results

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge memory_required decisions with multiple LLMs")
    parser.add_argument('--input_dir', type=str, default='./persona_metadata_domains_v3',
                        help='Directory containing generated persona metadata')
    parser.add_argument('--output_dir', type=str, default='./judgment_results',
                        help='Directory to save judgment results')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of files to sample (default: all)')
    parser.add_argument('--provider', type=str, default='openai',
                        help='Single provider to use (e.g., openai, claude, gemini)')
    parser.add_argument('--model', type=str, default=None,
                        help='Single model to use (e.g., gpt-5.1, claude-haiku-4-5). Requires --provider')
    parser.add_argument('--analyze_only', action='store_true', default=False,
                        help='Only analyze existing results without running new judgments')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        pass
    else:
        # Determine which models to use
        if args.provider and args.model:
            # Single specific model
            models_to_use = {args.provider: [args.model]}
        elif args.provider:
            # All models from a single provider
            if args.provider not in JUDGE_MODELS:
                print(f"Error: Unknown provider '{args.provider}'. Available: {list(JUDGE_MODELS.keys())}")
                exit(1)
            models_to_use = {args.provider: JUDGE_MODELS[args.provider]}
        else:
            # All models from all providers
            models_to_use = JUDGE_MODELS
        
        print(f"Models to run: {models_to_use}")
        
        results = run_multi_model_judgment(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            models=models_to_use,
            sample_size=args.sample_size
        )
