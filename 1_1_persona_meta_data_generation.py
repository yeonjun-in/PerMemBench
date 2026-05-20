# %%
from LLM import UnifiedLLM
import json
from dotenv import load_dotenv
from datasets import load_dataset
import random
import os
from tqdm import tqdm
import argparse

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("nvidia/Nemotron-Personas-USA")
domain_list = open('domain_list_final.txt', 'r', encoding='utf-8').read()
n_domains = len(domain_list.split('\n'))

prompt_format_all_domain = \
'''You are building a user profile for a personalized AI agent system.

## User Persona
{persona}

## Domains
{domain_list}

## Task
For every domain listed above, determine whether this user would use an AI agent for it, and if so, whether memory is required and how frequently they would use it.

## Instructions
- Evaluate all {n_domains} domains without exception.
- For each domain, first determine whether this user would plausibly use an AI agent for it (use: true/false).
- If use is true, determine whether memory is required (memory_required: true/false) and how frequently this user would use this domain (frequency: "high"/"medium"/"low").
- If use is false, set memory_required to null and frequency to null.
- Base your judgment entirely on this user's specific context, not on general assumptions about the domain.

## Definition of Memory Required
Memory is required (true) when this user's usage of the domain is:
- Ongoing and accumulative (e.g., tracking progress over time)
- Connected across multiple sessions (e.g., building on past conversations)
- Tied to long-term goals or evolving personal circumstances

Memory is NOT required (false) when this user's usage of the domain is:
- One-time or ad-hoc (e.g., a single lookup with no follow-up)
- Self-contained within a single session
- Not dependent on past interactions

## Definition of Frequency
- "high": This user would use an AI agent for this domain very regularly
- "medium": This user would use an AI agent for this domain occasionally
- "low": This user would use an AI agent for this domain rarely

## CRITICAL
Both memory_required and frequency must reflect THIS USER's specific context, not the general nature of the domain.
The same domain can have different memory_required and frequency values for different users.

For example:
- "Recipe Advice & Meal Planning" → memory_required=false, frequency="low" for a user who occasionally looks up recipes, but memory_required=true, frequency="high" for a user who is developing their own menu or working toward opening a restaurant.
- "Fitness & Exercise Planning" → memory_required=false, frequency="low" for a user who casually checks workout tips, but memory_required=true, frequency="high" for a user who follows a structured training routine and tracks progress over time.
- "News & Current Events" → memory_required=false, frequency="low" for a user who reads news casually, but memory_required=true, frequency="high" for a user who actively monitors specific topics for research or professional purposes.

## Output Format
Return a JSON object in the following format:
[
    {{
      "domain_name": "...",
      "use": true/false,
      "memory_required": true/false/null,
      "frequency": "high"/"medium"/"low"/null,
      "reason": "One sentence explaining why, grounded in this user's specific context."
    }}
  ]
'''

# %%
# random sample of personas
random.seed(1995)
total_size = len(ds['train'])
sample_indices = random.sample(range(total_size), 2000)
# create output directory
output_dir = f'./persona_metadata_domains'
os.makedirs(output_dir, exist_ok=True)

# initialize LLM
llm = UnifiedLLM('claude', 'claude-haiku-4-5')

results = []
errors = []

for idx in tqdm(sample_indices, desc="Processing personas"):
    sample = ds['train'][idx]
    uuid = sample['uuid']
    
    # build persona text
    persona = ''
    for key, value in sample.items():
        if key == 'uuid':
            continue
        persona += f"\n[{key}]"
        persona += f"\n{value}"
        persona += '\n'
    

    prompt = prompt_format_all_domain.format(persona=persona, domain_list=domain_list, n_domains=n_domains)
    
    try:
        response = llm.chat(prompt)
        domains = json.loads(response.replace('```json', '').replace('```', '').strip())
        
        result = {
            'uuid': uuid,
            'persona': dict(sample),
            'domains': domains
        }
        results.append(result)
        
        # save per-uuid file
        with open(os.path.join(output_dir, f'{uuid}.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        errors.append({'idx': idx, 'uuid': uuid, 'error': str(e)})
        print(f"Error at idx {idx} (uuid: {uuid}): {e}")

print(f"\nCompleted: {len(results)} / 1000")
print(f"Errors: {len(errors)}")

# also save aggregated results file
with open(os.path.join(output_dir, '_all_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

if errors:
    with open(os.path.join(output_dir, '_errors.json'), 'w', encoding='utf-8') as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)