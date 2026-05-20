"""
memory_policy.py

Memory Policy 구현.

- UniversalHeuristicPolicy: 모든 turn 무조건 저장 (raw text), oldest-first 삭제, consolidation 없음
- UniversalLLMPolicy: LLM이 저장 여부 판단, raw or key_fact (hyperparameter),
                      importance 기반 삭제, consolidation 있음
                      ※ LLM에게 domain/persona 정보 제공 안 함 — 대화 내용만 제공
- OraclePolicy: memory_required=True인 domain만 저장 (메타데이터 oracle 사용)
                저장 방식 및 삭제/consolidation은 UniversalLLMPolicy와 동일
"""

import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional

from memory_bank import MemoryBank, MemoryEntry, count_tokens


# ========================
# Prompts (UniversalLLMPolicy)
# ========================

STORAGE_DECISION_PROMPT = '''You are a memory management system for an AI assistant.
Below is a single turn from a conversation between a user and an AI agent.

## Turn
User: {user_content}
Agent: {agent_content}

## Task
Decide whether this turn contains information worth storing in long-term memory.
Store information that: contains specific personal facts, preferences, goals, constraints, or context
that would be useful to recall in a FUTURE conversation.
Do NOT store: generic questions, chit-chat, clarifications, or information with no future relevance.

Respond ONLY with a JSON object:
{{"store": true/false, "reason": "one sentence"}}
'''

KEY_FACT_EXTRACTION_PROMPT = '''Extract the key facts and important information from this conversation turn.

## Turn
User: {user_content}
Agent: {agent_content}

## Task
Summarize only the essential information worth remembering into 1-3 concise sentences.
Focus on: facts, preferences, goals, constraints, decisions made.
Ignore: greetings, filler phrases, generic advice not specific to this user.

Respond ONLY with a JSON object:
{{"content": "concise summary of key facts", "keywords": ["keyword1", "keyword2", ...]}}
'''

IMPORTANCE_SCORING_PROMPT = '''Rate the importance of storing this memory for future conversations.

## Memory Content
{content}

## Task
Score the importance from 0.0 to 1.0:
- 1.0: Critical personal information (e.g., medical condition, major life goal, strong preference)
- 0.7: Useful context (e.g., ongoing project details, recurring needs)
- 0.4: Mildly useful (e.g., one-time task details, passing mentions)
- 0.1: Low value (e.g., generic exchanges with no personal specifics)

Respond ONLY with a JSON object:
{{"importance_score": <float 0.0-1.0>, "reason": "one sentence"}}
'''

KEYWORD_EXTRACTION_PROMPT = '''Extract 3-7 keywords from this memory content.

Content: {content}

Respond ONLY with a JSON array of strings:
["keyword1", "keyword2", ...]
'''

CONSOLIDATION_PROMPT = '''You are managing a memory bank. Below are memory entries that may be related.

## Memory Entries
{entries_block}

## Task
Decide if any of these memories should be:
1. UPDATED: one memory supersedes another (newer information replaces older)
2. MERGED: multiple memories are about the same topic and can be consolidated
3. KEPT: memories are distinct enough to keep separately

Respond ONLY with a JSON object:
{{
  "action": "keep" | "update" | "merge",
  "target_ids": ["entry_id to keep/merge into"],
  "remove_ids": ["entry_ids to remove"],
  "merged_content": "new consolidated content (only if action=merge)",
  "merged_keywords": ["keyword1", ...],
  "merged_importance": <float 0.0-1.0>,
  "reason": "one sentence"
}}
'''


# ========================
# Helper
# ========================

def parse_llm_json(response: str) -> dict:
    clean = response.replace('```json', '').replace('```', '').strip()
    return json.loads(clean)


def format_turn_content(turn: dict) -> tuple[str, str]:
    """dialogue turn → (user_content, agent_content)"""
    return turn.get('user', ''), turn.get('agent', '')


def build_turn_raw_text(user_content: str, agent_content: str) -> str:
    return f"User: {user_content}\nAgent: {agent_content}"


# ========================
# Base
# ========================

class BaseMemoryPolicy(ABC):
    """모든 policy의 base class."""

    def __init__(self, deletion_strategy: str = 'oldest_first'):
        self.deletion_strategy = deletion_strategy

    @abstractmethod
    def process_session(
        self,
        session: dict,
        memory_bank: MemoryBank,
    ) -> list[MemoryEntry]:
        """
        세션 전체를 처리해 memory_bank에 저장.
        반환: 새로 추가된 MemoryEntry 목록
        """
        pass

    def post_session(self, memory_bank: MemoryBank) -> dict:
        """
        세션 종료 후 후처리 (consolidation, token limit 적용).
        반환: 처리 결과 통계
        """
        deleted = memory_bank.enforce_token_limit(strategy=self.deletion_strategy)
        return {"deleted_for_token_limit": deleted}

    def _extract_dialogue_turns(self, dialogue: list[dict]) -> list[tuple[int, str, str]]:
        """
        dialogue → [(turn_idx, user_content, agent_content), ...]
        user-agent pair를 하나의 turn으로 묶음 (turn_idx는 0-based pair index)
        """
        turns = []
        pair_idx = 0
        i = 0
        while i < len(dialogue):
            if dialogue[i]['role'] == 'user':
                user_content = dialogue[i]['content']
                agent_content = ''
                if i + 1 < len(dialogue) and dialogue[i + 1]['role'] == 'assistant':
                    agent_content = dialogue[i + 1]['content']
                    i += 2
                else:
                    i += 1
                turns.append((pair_idx, user_content, agent_content))
                pair_idx += 1
            else:
                i += 1
        return turns


# ========================
# Universal Heuristic Policy
# ========================

class UniversalHeuristicPolicy(BaseMemoryPolicy):
    """
    모든 turn을 무조건 raw text로 저장.
    삭제: oldest-first
    Consolidation: 없음
    Importance: 모두 0.5 (균일)
    """

    def __init__(self):
        super().__init__(deletion_strategy='oldest_first')

    def process_session(
        self,
        session: dict,
        memory_bank: MemoryBank,
    ) -> list[MemoryEntry]:
        dialogue = session.get('dialogue', [])
        session_file = session.get('_filename', 'unknown')
        session_idx = session.get('session_idx', -1)  # cold_start은 -1
        domain_name = session.get('domain_name', '')

        turns = self._extract_dialogue_turns(dialogue)
        entries = []
        for turn_idx, user_content, agent_content in turns:
            if not user_content and not agent_content:
                continue
            content = build_turn_raw_text(user_content, agent_content)
            # 키워드: 간단히 user 발화에서 상위 단어 추출 (heuristic)
            keywords = _simple_keywords(user_content, n=5)
            entry = MemoryEntry(
                entry_id=str(uuid.uuid4()),
                session_idx=session_idx,
                session_file=session_file,
                turn_idx=turn_idx,
                domain_name=domain_name,
                content=content,
                keywords=keywords,
                importance_score=0.5,
                timestamp=time.time(),
            )
            entries.append(entry)

        memory_bank.add_batch(entries)
        return entries

    def post_session(self, memory_bank: MemoryBank) -> dict:
        # consolidation 없음, token limit만 적용
        deleted = memory_bank.enforce_token_limit(strategy='oldest_first')
        return {"deleted_for_token_limit": deleted, "consolidation": False}


# ========================
# Universal LLM Policy
# ========================

class UniversalLLMPolicy(BaseMemoryPolicy):
    """
    LLM이 저장 여부 판단 (대화 내용만 보고 판단 — domain/persona 정보 없음).
    content 형식: 'raw' or 'key_fact' (hyperparameter)
    삭제: importance 기반
    Consolidation: 매 세션 종료 후
    """

    def __init__(
        self,
        llm,                          # UnifiedLLM instance
        content_mode: str = 'key_fact',  # 'raw' or 'key_fact'
        consolidation_top_k: int = 5,    # consolidation 시 retrieve top_k
    ):
        super().__init__(deletion_strategy='importance_based')
        self.llm = llm
        self.content_mode = content_mode
        self.consolidation_top_k = consolidation_top_k

    def _decide_store(self, user_content: str, agent_content: str) -> bool:
        prompt = STORAGE_DECISION_PROMPT.format(
            user_content=user_content,
            agent_content=agent_content[:500],  # truncate for cost
        )
        try:
            response = self.llm.chat(prompt)
            result = parse_llm_json(response)
            return bool(result.get('store', False))
        except Exception as e:
            print(f"  [WARN] storage decision failed: {e}")
            return False

    def _extract_key_fact(self, user_content: str, agent_content: str) -> tuple[str, list[str]]:
        prompt = KEY_FACT_EXTRACTION_PROMPT.format(
            user_content=user_content,
            agent_content=agent_content[:500],
        )
        try:
            response = self.llm.chat(prompt)
            result = parse_llm_json(response)
            content = result.get('content', '').strip()
            keywords = result.get('keywords', [])
            return content, keywords
        except Exception as e:
            print(f"  [WARN] key fact extraction failed: {e}")
            return build_turn_raw_text(user_content, agent_content), []

    def _score_importance(self, content: str) -> float:
        prompt = IMPORTANCE_SCORING_PROMPT.format(content=content[:500])
        try:
            response = self.llm.chat(prompt)
            result = parse_llm_json(response)
            score = float(result.get('importance_score', 0.5))
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"  [WARN] importance scoring failed: {e}")
            return 0.5

    def _extract_keywords(self, content: str) -> list[str]:
        prompt = KEYWORD_EXTRACTION_PROMPT.format(content=content[:400])
        try:
            response = self.llm.chat(prompt)
            result = json.loads(response.replace('```json', '').replace('```', '').strip())
            if isinstance(result, list):
                return [str(k) for k in result[:7]]
        except Exception:
            pass
        return _simple_keywords(content, n=5)

    def process_session(
        self,
        session: dict,
        memory_bank: MemoryBank,
    ) -> list[MemoryEntry]:
        dialogue = session.get('dialogue', [])
        session_file = session.get('_filename', 'unknown')
        session_idx = session.get('session_idx', -1)
        domain_name = session.get('domain_name', '')

        turns = self._extract_dialogue_turns(dialogue)
        entries = []
        for turn_idx, user_content, agent_content in turns:
            if not user_content:
                continue

            # Step 1: 저장 여부 판단
            should_store = self._decide_store(user_content, agent_content)
            if not should_store:
                continue

            # Step 2: content 생성
            if self.content_mode == 'key_fact':
                content, keywords = self._extract_key_fact(user_content, agent_content)
            else:  # raw
                content = build_turn_raw_text(user_content, agent_content)
                keywords = self._extract_keywords(content)

            if not content.strip():
                continue

            # Step 3: importance scoring
            importance = self._score_importance(content)

            entry = MemoryEntry(
                entry_id=str(uuid.uuid4()),
                session_idx=session_idx,
                session_file=session_file,
                turn_idx=turn_idx,
                domain_name=domain_name,
                content=content,
                keywords=keywords,
                importance_score=importance,
                timestamp=time.time(),
            )
            entries.append(entry)

        memory_bank.add_batch(entries)
        return entries

    def post_session(self, memory_bank: MemoryBank) -> dict:
        """세션 종료 후: consolidation → token limit 적용"""
        consolidated = self._run_consolidation(memory_bank)
        deleted = memory_bank.enforce_token_limit(strategy='importance_based')
        return {
            "consolidation_actions": consolidated,
            "deleted_for_token_limit": deleted,
        }

    def _run_consolidation(self, memory_bank: MemoryBank) -> list[dict]:
        """
        최근 추가된 entry들에 대해 관련 메모리를 retrieve해서 consolidation 실행.
        """
        if memory_bank.size < 2:
            return []

        actions = []
        processed_ids = set()

        # 최근 5개 entry에 대해 관련 메모리 확인
        recent = memory_bank.entries[-5:]
        for entry in recent:
            if entry.entry_id in processed_ids:
                continue

            # 관련 메모리 retrieve
            related = memory_bank.retrieve(entry.content, top_k=self.consolidation_top_k)
            # 자기 자신 제외, 이미 처리된 것 제외
            related = [
                e for e in related
                if e.entry_id != entry.entry_id and e.entry_id not in processed_ids
            ]
            if not related:
                continue

            # 상위 관련도 1개와 consolidation 시도
            candidate = related[0]
            entries_block = (
                f"Entry 1 (ID: {entry.entry_id}):\n{entry.content}\n\n"
                f"Entry 2 (ID: {candidate.entry_id}):\n{candidate.content}"
            )
            prompt = CONSOLIDATION_PROMPT.format(entries_block=entries_block)
            try:
                response = self.llm.chat(prompt)
                result = parse_llm_json(response)
                action = result.get('action', 'keep')

                if action == 'merge':
                    merged_content = result.get('merged_content', '')
                    merged_keywords = result.get('merged_keywords', [])
                    merged_importance = float(result.get('merged_importance', 0.5))
                    if merged_content:
                        memory_bank.merge_entries(
                            [entry.entry_id, candidate.entry_id],
                            merged_content, merged_keywords, merged_importance
                        )
                        processed_ids.update([entry.entry_id, candidate.entry_id])
                        actions.append({
                            "action": "merge",
                            "ids": [entry.entry_id, candidate.entry_id],
                        })

                elif action == 'update':
                    remove_ids = result.get('remove_ids', [])
                    for rid in remove_ids:
                        memory_bank.delete(rid)
                        processed_ids.add(rid)
                    actions.append({
                        "action": "update",
                        "removed": remove_ids,
                    })

            except Exception as e:
                print(f"  [WARN] consolidation failed: {e}")
                continue

        return actions


# ========================
# Simple keyword helper
# ========================

def _simple_keywords(text: str, n: int = 5) -> list[str]:
    """간단한 heuristic 키워드 추출 (stopword 제거 + 빈도순)"""
    STOPWORDS = {
        'i', 'me', 'my', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by', 'from',
        'and', 'or', 'but', 'not', 'it', 'its', 'this', 'that', 'what', 'how',
        'can', 'just', 'so', 'if', 'you', 'your', 'we', 'our', 'they', 'their',
        'user', 'agent', 'yes', 'no', 'ok', 'okay', 'great', 'thanks', 'sure',
    }
    import re
    from collections import Counter
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    return [w for w, _ in Counter(filtered).most_common(n)]


# ========================
# Oracle Policy (LLM-based)
# ========================

class OraclePolicy(UniversalLLMPolicy):
    """
    Personalized Oracle Policy.

    memory_required=True인 domain의 세션만 저장.
    memory_required=False이면 해당 세션 전체를 스킵.

    저장 방식, importance scoring, 삭제(importance 기반), consolidation은
    UniversalLLMPolicy와 완전히 동일 — 유일한 차이는 domain 필터링 여부.

    ※ session의 'memory_required' 필드를 그대로 사용 (메타데이터 oracle).
    """

    def __init__(self, llm, content_mode: str = 'key_fact', consolidation_top_k: int = 5):
        super().__init__(llm=llm, content_mode=content_mode, consolidation_top_k=consolidation_top_k)

    def process_session(self, session: dict, memory_bank: MemoryBank) -> list[MemoryEntry]:
        if not session.get('memory_required', False):
            return []
        return super().process_session(session, memory_bank)


# ========================
# Oracle Heuristic Policy
# ========================

class OracleHeuristicPolicy(UniversalHeuristicPolicy):
    """
    Oracle 필터 + Heuristic 저장 방식.

    memory_required=True인 domain의 세션만 저장하되,
    저장 방식은 UniversalHeuristicPolicy와 동일:
    - 모든 turn 무조건 raw text 저장
    - importance=0.5 균일
    - oldest-first 삭제
    - consolidation 없음

    비교 목적:
      OracleHeuristicPolicy vs OraclePolicy(LLM)
        → oracle 필터 유지하면서 저장 방식(heuristic vs LLM)의 효과만 분리
      OracleHeuristicPolicy vs UniversalHeuristicPolicy
        → 저장 방식 고정하고 oracle 필터의 효과만 분리
    """

    def __init__(self):
        super().__init__()

    def process_session(self, session: dict, memory_bank: MemoryBank) -> list[MemoryEntry]:
        if not session.get('memory_required', False):
            return []
        return super().process_session(session, memory_bank)