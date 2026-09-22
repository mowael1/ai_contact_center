"""Short-term conversation memory.

An agent asks follow-ups: "ازاي أجدد باقتي؟" then "وكام مدتها؟". The second
question is meaningless on its own - neither retrieval nor generation can
resolve "مدتها" without the previous turn.

Two distinct uses, kept separate on purpose:

* **Retrieval** needs a self-contained query. Pronouns are resolved by
  stitching the recent turns' key terms onto the follow-up, deterministically -
  no extra LLM round trip, which would add ~700 ms to every question.
* **Generation** gets the recent turns verbatim, so the answer reads as part of
  a conversation.

Memory is deliberately short and in-process: it is a UX aid, not a record.
Anything worth keeping belongs in the database.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from rag.config import settings
from rag.logging_utils import get_logger

logger = get_logger(__name__)

# A follow-up is short and leans on the previous turn.
_REFERENTIAL = re.compile(
    r"\b(it|its|that|this|they|them|those|these|he|she|there)\b"
    r"|(?:^|\s)(?:و|ال)?(?:ده|دي|دول|هو|هي|هم|كده|بتاعها|بتاعه|بتاعهم|ليها|ليه|فيها|فيه|عنها|عنه)(?:\s|$|؟)",
    re.IGNORECASE,
)
_WORD = re.compile(r"[\w؀-ۿ]+", re.UNICODE)


@dataclass(slots=True)
class Turn:
    question: str
    answer: str
    at: float = field(default_factory=time.time)


@dataclass(slots=True)
class Conversation:
    turns: list[Turn] = field(default_factory=list)
    updated: float = field(default_factory=time.time)

    def add(self, question: str, answer: str, max_turns: int) -> None:
        self.turns.append(Turn(question, answer))
        del self.turns[:-max_turns]
        self.updated = time.time()

    def recent(self, limit: int) -> list[Turn]:
        return self.turns[-limit:]


class ConversationStore:
    """In-process, per-conversation history with a size and age bound.

    Not shared between workers and not persisted - restarting the API clears
    it. That is acceptable for a UX aid and avoids pretending it is a durable
    transcript.
    """

    def __init__(self, max_conversations: int = 500) -> None:
        self._data: "OrderedDict[str, Conversation]" = OrderedDict()
        self._max = max_conversations

    @staticmethod
    def key(company_id: int, conversation_id: str) -> str:
        # Scoped by company so a conversation id can never reach another tenant.
        return f"{company_id}:{conversation_id}"

    def get(self, company_id: int, conversation_id: Optional[str]) -> Conversation:
        if not conversation_id:
            return Conversation()
        key = self.key(company_id, conversation_id)
        conversation = self._data.get(key)
        if conversation is None:
            return Conversation()
        if time.time() - conversation.updated > settings.CHAT_MEMORY_TTL_SECONDS:
            self._data.pop(key, None)
            return Conversation()
        self._data.move_to_end(key)
        return conversation

    def append(
        self, company_id: int, conversation_id: Optional[str], question: str, answer: str
    ) -> None:
        if not conversation_id:
            return
        key = self.key(company_id, conversation_id)
        conversation = self._data.get(key) or Conversation()
        conversation.add(question, answer, settings.CHAT_MEMORY_TURNS)
        self._data[key] = conversation
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def clear(self, company_id: int, conversation_id: str) -> bool:
        return self._data.pop(self.key(company_id, conversation_id), None) is not None

    def stats(self) -> dict:
        return {"conversations": len(self._data), "max": self._max}


#: One store per process.
conversations = ConversationStore()


def is_follow_up(question: str) -> bool:
    """Whether a question leans on what came before."""
    words = _WORD.findall(question)
    if len(words) <= 3:
        return True
    return bool(_REFERENTIAL.search(question))


def contextual_query(question: str, conversation: Conversation) -> str:
    """A self-contained query for retrieval.

    Built by prepending the previous question's distinctive terms rather than
    by asking a model to rewrite it: retrieval runs on every message and an
    extra LLM call would dominate the latency budget.
    """
    if not conversation.turns or not is_follow_up(question):
        return question

    previous = conversation.turns[-1].question
    existing = {w.lower() for w in _WORD.findall(question)}
    carried = [
        w for w in _WORD.findall(previous)
        if len(w) > 2 and w.lower() not in existing
    ]
    if not carried:
        return question
    enriched = f"{question} {' '.join(carried[:12])}"
    logger.info("Follow-up expanded for retrieval: %r -> %r", question, enriched)
    return enriched


def format_history(conversation: Conversation, limit: int) -> str:
    """Recent turns for the generation prompt."""
    turns = conversation.recent(limit)
    if not turns:
        return ""
    lines = []
    for turn in turns:
        lines.append(f"User: {turn.question}")
        lines.append(f"Assistant: {turn.answer}")
    return "\n".join(lines)
