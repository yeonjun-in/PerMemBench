import json
import os
import argparse
from pathlib import Path
from glob import glob
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from LLM import UnifiedLLM


# ============================================================
# PROMPTS
# ============================================================

AGENT_SYSTEM = """\
You are a helpful, knowledgeable AI agent assisting a user in their daily life.

## Instructions
- Respond helpfully and naturally to the user's messages.
- Do not make assumptions about the user beyond what they explicitly tell you.
- Keep responses conversational and appropriately concise.
- Do NOT ask multiple questions at once. If clarification is needed, ask ONE question at a time.
- Do NOT refer to yourself as an AI unless the user asks.
- If you cannot fulfill a request (cannot generate files, execute code, access external systems),
  begin your response with [CANNOT], briefly explain, and offer a text-based alternative.

## Critical: Do NOT offer or suggest anything unprompted
- Answer exactly what was asked, then stop.
- Do NOT end your response with offers like "Would you like me to...", "If you want, I can...",
  "I can also...", or any similar follow-up suggestion.
- Do NOT propose next steps, additional help, or related actions unless the user explicitly asks.
- Do NOT offer to create files, printable materials, checklists, formatted documents, or summaries.
- Do NOT ask clarifying questions unless the user's request is genuinely impossible to answer without them.

Output only your response. No labels, no meta-commentary.
"""

USER_SYSTEM_TEMPLATE = """\
You are roleplaying as a real person interacting with an AI agent in their daily life.

## Your Persona
{persona}

## Critical Reminders
- You are always the USER seeking help. You are NEVER the assistant.
- Never say things like "feel free to ask me" or "how can I assist".
- You are the one asking questions and requesting assistance.
- Communicate in English only.
- Do NOT ask for printable materials, formatted documents, checklists to print,
  templates to save, or file outputs of any kind. Keep your requests conversational.
"""

USER_OPENING_PROMPT = """\
You are about to start a new conversation with an AI agent about: {domain_name}

## What is happening in your life right now
{event_description}
{prior_context_section}\
{gt_facts_block}\
## How to behave naturally
- Open with only what you need right now — 1 to 3 sentences max. Do NOT front-load all your details upfront.
- Share only what's needed to start the conversation. More details will come naturally as the agent asks follow-up questions.
- Do NOT introduce yourself, list your traits, or summarize these facts upfront.
- Let your personality and situation emerge through how you ask questions and react.
- If the agent says something that doesn't fit who you are (wrong tone, wrong assumption),
  push back or redirect naturally — the way a real person would.
- The agent has NO memory of you. You are starting fresh.
- Do NOT volunteer decisions that haven't been made yet in this conversation.
  Let those emerge naturally as the agent helps you figure them out.

Output only your opening message. No labels, no meta-commentary.
"""

USER_CONTINUE_PROMPT = """\
What would you say next in this conversation?

As you continue:
- Stay true to who you are (see your situation and traits above).
- React honestly — if something the agent said doesn't fit your personality or needs,
  push back or redirect naturally rather than just going along with it.
- Only share more details about yourself when they are directly relevant to the current topic.

Check: have the following aspects of yourself come up clearly yet in this conversation?
{unrevealed_facts}
If any have NOT come up yet, look for a natural moment in this turn to let them surface —
through how you react, what you ask, or how you describe your situation.
Do NOT announce or list them. Let them emerge from how you respond.

If the conversation has reached a natural conclusion and you have nothing more to ask, reply with only: [END]
If the agent's last response starts with [CANNOT], reply with only: [END]
If you have already asked the same question or made the same request more than once without
a satisfying answer, reply with only: [END]

Output only your next message or [END]. No labels, no meta-commentary.
"""


# ============================================================
# HELPERS
# ============================================================

def format_persona(persona_dict: dict) -> str:
    skip_keys = {"uuid"}
    lines = []
    for key, value in persona_dict.items():
        if key in skip_keys:
            continue
        lines.append(f"[{key}]\n{value}")
    return "\n\n".join(lines)


def strip_cannot_token(dialogue: list[dict]) -> list[dict]:
    cleaned = []
    for turn in dialogue:
        content = turn["content"]
        if turn["role"] == "assistant" and content.startswith("[CANNOT]"):
            content = content[len("[CANNOT]"):].lstrip()
        cleaned.append({**turn, "content": content})
    return cleaned


def lookup_event(domain_skeletons: list[dict], domain: str, project_id: int, event_id: int) -> dict | None:
    for ds in domain_skeletons:
        if ds["domain_name"] != domain:
            continue
        for proj in ds.get("skeleton", {}).get("projects", []):
            if proj["project_id"] != project_id:
                continue
            for evt in proj.get("events", []):
                if evt["event_id"] == event_id:
                    return evt
    return None


def lookup_project(domain_skeletons: list[dict], domain: str, project_id: int) -> dict | None:
    for ds in domain_skeletons:
        if ds["domain_name"] != domain:
            continue
        for proj in ds.get("skeleton", {}).get("projects", []):
            if proj["project_id"] == project_id:
                return proj
    return None


_llm_for_topic: UnifiedLLM | None = None  # set at runtime

_OPEN_TOPIC_PROMPT = """\
A memory system will store the following fact after an AI conversation:
"{fact}"

Rewrite this as a short, open uncertainty (1 sentence, present tense) that describes
what the user has NOT yet decided before this conversation.
Do NOT include the answer. Start with "You haven't decided" or "You're unsure about".

Examples:
- Fact: "Amanda decided on a 16-week CST study plan with four 45-minute sessions per week."
  → "You haven't decided on a CST study timeline or how many sessions to do per week."
- Fact: "Melissa chose a local community college GED prep course paired with official online materials."
  → "You're unsure which GED prep option to use — local classes, online, or a combination."

Return only the one sentence.
"""


def _fact_to_open_topic(fact: str) -> str:
    """Convert a decided fact into an open, not-yet-resolved question for the user LLM."""
    if _llm_for_topic is None:
        return f"You haven't yet decided: {fact.lower()}"
    prompt = _OPEN_TOPIC_PROMPT.format(fact=fact)
    try:
        return _llm_for_topic.chat(prompt).strip()
    except Exception:
        return f"You haven't yet decided: {fact}"


def build_gt_facts_section(gt_memory: list[dict]) -> str:
    
    if not gt_memory:
        return ""

    profile_lines = []
    state_hints = []

    for item in gt_memory:
        if item["type"] == "user_profile":
            profile_lines.append(f"- {item['fact']}")
        else:
            topic = _fact_to_open_topic(item["fact"])
            state_hints.append(f"- {topic}")

    sections = []
    if profile_lines:
        sections.append(
            "Your traits and preferences — let these shape how you react, "
            "DO NOT announce or list them upfront:"
        )
        sections.extend(profile_lines)
    if state_hints:
        if sections:
            sections.append("")
        sections.append(
            "Things you have NOT yet decided — you are coming to this conversation "
            "to figure these out. Do NOT state them as already decided:"
        )
        sections.extend(state_hints)
    return "\n".join(sections) + "\n"


def build_gt_facts_block(gt_memory: list[dict]) -> str:
    
    facts_section = build_gt_facts_section(gt_memory)
    if not facts_section:
        return ""
    return f"## Who you are in this conversation\n{facts_section}\n"


def build_prior_context_section(prior_sessions: list[dict]) -> str:
    
    if not prior_sessions:
        return ""

    lines = ["## What you already know from previous conversations"]
    added = False
    for s in prior_sessions:
        domain = s.get("domain", "")
        event_title = s.get("event_title", "")
        facts = s.get("gt_memory", [])
        if not facts:
            continue
        lines.append(f"\n[{domain} — {event_title}]")
        for f in facts:
            lines.append(f"- {f['fact']}")
        added = True

    if not added:
        return ""

    lines.append("")
    return "\n".join(lines) + "\n"


FACT_REVEALED_PROMPT = """\
You are checking whether a specific fact about the user has come up in a conversation.

Fact to check:
"{fact}"

Conversation so far (user turns only):
{user_turns}

Has the essence of this fact been expressed by the user in the conversation above?
Answer YES if the user has communicated the core meaning — even if worded differently.
Answer NO if the user has not yet expressed this.

Reply with only YES or NO.
"""


def facts_revealed_so_far(
    dialogue: list[dict],
    gt_memory: list[dict],
    llm: UnifiedLLM,
) -> set[int]:
    """
    Use an LLM to check whether each gt_memory fact has surfaced in the user turns.
    Returns set of indices of facts considered revealed.
    """
    user_turns = "\n".join(
        f"User: {t['content']}" for t in dialogue if t["role"] == "user"
    )
    if not user_turns.strip():
        return set()

    revealed = set()
    for i, item in enumerate(gt_memory):
        prompt = FACT_REVEALED_PROMPT.format(
            fact=item["fact"],
            user_turns=user_turns,
        )
        try:
            answer = llm.chat(prompt).strip().upper()
            if answer.startswith("YES"):
                revealed.add(i)
        except Exception:
            pass
    return revealed


def build_unrevealed_facts_section(gt_memory: list[dict], revealed_indices: set[int]) -> str:
    """Build bullet lists of revealed and unrevealed facts."""
    unrevealed = []
    revealed = []
    for i, item in enumerate(gt_memory):
        label = "(about yourself)" if item["type"] == "user_profile" else "(about your situation)"
        if i not in revealed_indices:
            unrevealed.append(f"- {label} {item['fact']}")
        else:
            revealed.append(f"- {label} {item['fact']}")

    parts = []
    if unrevealed:
        parts.append("Not yet surfaced — find a natural moment to let these come through:")
        parts.extend(unrevealed)
    else:
        parts.append("(All key aspects have already come up.)")
    if revealed:
        parts.append("")
        parts.append("Already came up — do NOT repeat these again:")
        parts.extend(revealed)
    return "\n".join(parts)


def generate_session(
    llm_user: UnifiedLLM,
    llm_agent: UnifiedLLM,
    user_system: str,
    domain_name: str,
    event_description: str,
    gt_memory: list[dict],
    max_turns: int,
    min_turns: int = 3,
    prior_sessions: list[dict] | None = None,
) -> list[dict]:
    
    dialogue = []
    gt_facts_block = build_gt_facts_block(gt_memory)
    prior_context_section = build_prior_context_section(prior_sessions or [])

    # --- User opening ---
    opening_prompt = USER_OPENING_PROMPT.format(
        domain_name=domain_name,
        event_description=event_description,
        prior_context_section=prior_context_section,
        gt_facts_block=gt_facts_block,
    )
    first_user = llm_user.chat_messages(
        messages=[{"role": "user", "content": opening_prompt}],
        system=user_system,
    )
    dialogue.append({"role": "user", "content": first_user})

    pairs_completed = 0

    while pairs_completed < max_turns:
        # --- Agent reply ---
        agent_reply = llm_agent.chat_messages(
            messages=dialogue,
            system=AGENT_SYSTEM,
            tools=[{"type": "web_search"}],
        )
        dialogue.append({"role": "assistant", "content": agent_reply})
        pairs_completed += 1

        if pairs_completed >= max_turns:
            break

        if agent_reply.strip().startswith("[CANNOT]"):
            break

        # --- Revealed facts tracking (skipped for oneoff: gt_memory=[]) ---
        if gt_memory:
            revealed = facts_revealed_so_far(dialogue, gt_memory, llm_user)
            all_revealed = len(revealed) >= len(gt_memory)
            if all_revealed and pairs_completed >= min_turns:
                unrevealed_section = "(All key aspects have already come up — wrap up naturally if you have nothing more to ask.)"
            else:
                unrevealed_section = build_unrevealed_facts_section(gt_memory, revealed)
        else:
            # Oneoff session: no facts to track, nudge toward natural close
            unrevealed_section = "(This is a one-time request. Wrap up naturally once your need is met.)"

        # --- User next turn ---
        continue_prompt = USER_CONTINUE_PROMPT.format(
            unrevealed_facts=unrevealed_section,
        )
        next_user = llm_user.chat_messages(
            messages=dialogue + [{"role": "user", "content": continue_prompt}],
            system=user_system,
        )
        if "[END]" in next_user:
            break
        dialogue.append({"role": "user", "content": next_user})

    return strip_cannot_token(dialogue)


def process_timeline_file(
    filepath: str,
    llm_user: UnifiedLLM,
    llm_agent: UnifiedLLM,
    output_dir: str,
    max_turns: int,
    max_turns_oneoff: int,
    overwrite: bool,
) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    uuid = data["uuid"]
    persona = data["persona"]
    domain_skeletons = data.get("domain_skeletons", [])
    sessions = data.get("timeline", {}).get("sessions", [])

    if not sessions:
        return {"uuid": uuid, "skipped": True, "reason": "no sessions"}

    persona_text = format_persona(persona)
    user_system = USER_SYSTEM_TEMPLATE.format(persona=persona_text)

    out_dir = Path(output_dir) / uuid
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"uuid": uuid, "total": len(sessions), "success": 0, "skipped": 0, "errors": []}

    for session_meta in tqdm(sessions, desc=f"  {uuid[:8]}", leave=False):
        session_id = session_meta["session_id"]
        domain = session_meta["domain"]
        event_id = session_meta["event_id"]
        month = session_meta["month"]
        memory_required = session_meta.get("memory_required", True)

        out_path = out_dir / f"session_{session_id:04d}.json"
        if not overwrite and out_path.exists():
            results["skipped"] += 1
            continue

        # ── Branch: memory_required=True vs False ──────────────────────────
        if memory_required:
            # Existing path: look up event from domain skeleton
            project_id = session_meta["project_id"]
            event = lookup_event(domain_skeletons, domain, project_id, event_id)
            project = lookup_project(domain_skeletons, domain, project_id)
            if not event:
                results["errors"].append(
                    f"session {session_id}: event not found "
                    f"({domain} P{project_id}E{event_id})"
                )
                continue
            event_description = event.get("description", "")
            event_title = event.get("title", "")
            project_title = project.get("title", "") if project else ""
            gt_memory = event.get("gt_memory", [])
            turns = max_turns
        else:
            # Oneoff path: event_description is embedded directly in the timeline
            project_id = None
            project_title = ""
            event_description = session_meta.get("event_description", "")
            event_title = session_meta.get("event_title", "")
            gt_memory = []
            turns = max_turns_oneoff
        # ───────────────────────────────────────────────────────────────────

        # Load prior sessions for user LLM context (oneoff gt_memory=[] auto-skipped)
        prior_sessions = []
        for prev_id in range(1, session_id):
            prev_path = out_dir / f"session_{prev_id:04d}.json"
            if prev_path.exists():
                try:
                    with open(prev_path, "r", encoding="utf-8") as pf:
                        prev_data = json.load(pf)
                    prior_sessions.append({
                        "domain": prev_data.get("domain", ""),
                        "event_title": prev_data.get("event_title", ""),
                        "gt_memory": prev_data.get("gt_memory", []),
                    })
                except Exception:
                    pass

        try:
            dialogue = generate_session(
                llm_user=llm_user,
                llm_agent=llm_agent,
                user_system=user_system,
                domain_name=domain,
                event_description=event_description,
                gt_memory=gt_memory,
                max_turns=turns,
                prior_sessions=prior_sessions,
            )

            output = {
                "uuid": uuid,
                "session_id": session_id,
                "month": month,
                "domain": domain,
                "memory_required": memory_required,
                "project_id": project_id,
                "project_title": project_title,
                "event_id": event_id,
                "event_title": event_title,
                "event_description": event_description,
                "gt_memory": gt_memory,
                "dialogue": dialogue,
                "anchor_life_event": session_meta.get("anchor_life_event"),
                "cross_domain_links": session_meta.get("cross_domain_links", []),
            }

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            results["success"] += 1

        except Exception as e:
            results["errors"].append(f"session {session_id}: {e}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate dialogue sessions from integrated timelines"
    )
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to a single timeline JSON file")
    parser.add_argument("--input_dir", type=str, default='life_timelines_merged',
                        help="Directory containing timeline JSON files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of files to process")
    parser.add_argument("--output_dir", type=str, default="./skeleton_dialogues",
                        help="Output directory (default: ./skeleton_dialogues_gpt)")
    parser.add_argument("--provider", type=str, default="openai",
                        help="LLM provider for both user and agent: openai | claude | together | gemini")
    parser.add_argument("--model", type=str, default='gpt-5-mini',
                        help="Model name override")
    parser.add_argument("--user_provider", type=str, default=None,
                        help="Override provider for user LLM (default: same as --provider)")
    parser.add_argument("--user_model", type=str, default=None,
                        help="Override model for user LLM")
    parser.add_argument("--agent_provider", type=str, default=None,
                        help="Override provider for agent LLM (default: same as --provider)")
    parser.add_argument("--agent_model", type=str, default=None,
                        help="Override model for agent LLM")
    parser.add_argument("--max_turns", type=int, default=10,
                        help="Max turn pairs per memory_required=True session (default: 10)")
    parser.add_argument("--max_turns_oneoff", type=int, default=5,
                        help="Max turn pairs per memory_required=False one-off session (default: 5)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing session files")
    args = parser.parse_args()

    if not args.input_file and not args.input_dir:
        parser.error("Provide --input_file or --input_dir")

    user_provider = args.user_provider or args.provider
    user_model = args.user_model or args.model
    agent_provider = args.agent_provider or args.provider
    agent_model = args.agent_model or args.model

    llm_user = UnifiedLLM(user_provider, user_model)
    llm_agent = UnifiedLLM(agent_provider, agent_model)

    # Share llm_user for converting ongoing_state facts to open topics
    import sys as _sys
    _mod = _sys.modules[__name__]
    _mod._llm_for_topic = llm_user

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_file:
        files = [args.input_file]
    else:
        files = sorted(glob(os.path.join(args.input_dir, "*.json")))
        files = [f for f in files if not os.path.basename(f).startswith("_")]

    if args.limit:
        files = files[:args.limit]

    print(f"User  LLM        : {user_provider} / {user_model or 'default'}")
    print(f"Agent LLM        : {agent_provider} / {agent_model or 'default'}")
    print(f"Max turns (True) : {args.max_turns}")
    print(f"Max turns (False): {args.max_turns_oneoff}")
    print(f"Files            : {len(files)}")
    print(f"Output           : {args.output_dir}\n")

    total_success = total_skipped = total_errors = 0

    for filepath in tqdm(files, desc="Personas"):
        try:
            result = process_timeline_file(
                filepath=filepath,
                llm_user=llm_user,
                llm_agent=llm_agent,
                output_dir=args.output_dir,
                max_turns=args.max_turns,
                max_turns_oneoff=args.max_turns_oneoff,
                overwrite=args.overwrite,
            )
            if result.get("skipped") and "reason" in result:
                print(f"  skip {result['uuid'][:8]}: {result.get('reason')}")
                continue

            s = result.get("success", 0)
            sk = result.get("skipped", 0)
            errs = result.get("errors", [])
            total_success += s
            total_skipped += sk
            total_errors += len(errs)

            print(f"  ok   {result['uuid'][:8]}: {s} sessions, {sk} skipped, {len(errs)} errors")
            for e in errs[:3]:
                print(f"    - {e}")

        except Exception as e:
            total_errors += 1
            print(f"  err  {filepath}: {e}")

    print(f"\nDone — sessions: {total_success} generated, {total_skipped} skipped, {total_errors} errors")


if __name__ == "__main__":
    main()
