"""
core/prompts/history_builder.py

HistoryBuilder transforms raw multi-turn conversation logs into concise,
semantically deduplicated rolling summaries and recent verbatim dialogue turns.
"""

from typing import List, Dict, Any, Optional


class HistoryBuilder:
    """
    Constructs rolling conversation summaries, eliminates repetitive operator turns,
    and formats recent dialogue turns within context budget constraints.
    """

    def __init__(self, recent_turns_limit: int = 3):
        self.recent_turns_limit = recent_turns_limit

    def deduplicate(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Remove consecutive duplicate user queries or identical assistant responses.
        """
        if not history:
            return []

        deduped = []
        last_role = None
        last_msg = None

        for turn in history:
            role = turn.get("role", "").strip()
            msg = turn.get("message", turn.get("content", "")).strip()

            # Skip consecutive identical turns from the same role
            if role == last_role and msg == last_msg:
                continue

            deduped.append({"role": role, "message": msg})
            last_role = role
            last_msg = msg

        return deduped

    def compress_older_turns(self, older_turns: List[Dict[str, str]]) -> List[str]:
        """
        Compress older historical turns into compact semantic bullet points.
        """
        summary_bullets = []
        i = 0
        while i < len(older_turns):
            curr = older_turns[i]
            role = curr.get("role", "")
            msg = curr.get("message", "")

            # If user turn followed by assistant turn, collapse into a single transaction
            if role in ("user", "human") and i + 1 < len(older_turns):
                next_turn = older_turns[i + 1]
                if next_turn.get("role") in ("assistant", "ai", "model"):
                    u_brief = msg.split("\n")[0][:60]
                    a_brief = next_turn.get("message", "").split("\n")[0][:60]
                    summary_bullets.append(f"- User inquired: '{u_brief}' -> Co-Pilot guided: '{a_brief}'")
                    i += 2
                    continue

            brief = msg.split("\n")[0][:75]
            summary_bullets.append(f"- {role.capitalize()}: {brief}")
            i += 1

        return summary_bullets

    def build(
        self,
        history: Optional[List[Dict[str, str]]] = None,
        max_chars: int = 800
    ) -> str:
        """
        Build rolling history context formatted within max_chars.
        """
        if not history:
            return "(first turn)"

        clean_history = self.deduplicate(history)
        if not clean_history:
            return "(first turn)"

        # Partition into older turns (to be summarized) and recent turns (verbatim)
        if len(clean_history) <= self.recent_turns_limit:
            older_turns = []
            recent_turns = clean_history
        else:
            older_turns = clean_history[:-self.recent_turns_limit]
            recent_turns = clean_history[-self.recent_turns_limit:]

        blocks = []

        # 1. Rolling summary of older conversation
        if older_turns:
            bullets = self.compress_older_turns(older_turns)
            if bullets:
                blocks.append("Prior Context Summary:\n" + "\n".join(bullets))

        # 2. Recent turns verbatim
        recent_lines = []
        for turn in recent_turns:
            role = turn.get("role", "unknown").capitalize()
            msg = turn.get("message", "")[:200].strip()
            recent_lines.append(f"{role}: {msg}")

        if recent_lines:
            blocks.append("Recent Turns:\n" + "\n".join(recent_lines))

        result = "\n\n".join(blocks).strip()

        if len(result) > max_chars:
            result = result[:max_chars].rstrip() + "... [history truncated]"

        return result
