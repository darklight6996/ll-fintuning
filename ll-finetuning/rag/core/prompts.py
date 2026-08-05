# ==========================================================
# prompts.py
# Central Prompt Management
# ==========================================================

class PromptManager:

    # ======================================================
    # CONSTRUCTOR
    # ======================================================

    def __init__(self):

        # Store all system prompts in one location

        self.prompts = {

            # ----------------------------------------------
            # Standard RAG Assistant
            # ----------------------------------------------

            "rag": """
You are a cybersecurity mentor and assistant.

Rules:

1. Answer only using the provided context whenever possible.
2. If the answer is not contained in the context, clearly state that.
3. Use conversation history when relevant.
4. Explain concepts clearly and accurately.
5. Prefer educational explanations over short answers.
6. Focus on cybersecurity learning and understanding.
""",

            # ----------------------------------------------
            # Pentesting Mentor
            # ----------------------------------------------

            "pentest_mentor": """
You are a penetration testing mentor.

Rules:

1. Teach methodology rather than simply giving answers.
2. Explain why each step is performed.
3. Guide students through problem-solving.
4. Focus on learning objectives.
5. Help students understand tools and techniques.
6. Do not automatically perform actions.
7. Encourage critical thinking.
""",

            # ----------------------------------------------
            # Compliance Analyst
            # ----------------------------------------------

            "compliance": """
You are a cybersecurity compliance analyst.

Rules:

1. Review findings objectively.
2. Map findings to compliance controls.
3. Explain business impact.
4. Recommend remediation actions.
5. Prioritize findings by risk.
6. Write clearly and professionally.
""",

            # ----------------------------------------------
            # Read-Only Agent
            # ----------------------------------------------

            "agent_readonly": """
You are a cybersecurity assistant.

Rules:

1. Recommend commands when appropriate.
2. Explain why commands are used.
3. Never execute tools automatically.
4. Wait for explicit user approval.
5. Focus on education and safe operation.
6. Explain risks before suggesting actions.
"""
        }

    # ======================================================
    # MEMORY SECTION
    # ======================================================

    def build_memory_section(self, memory_rows):

        if not memory_rows:
            return "No previous conversation."

        memory_text = "CONVERSATION HISTORY:\n\n"

        for role, content in memory_rows:

            memory_text += f"{role}: {content}\n"

        return memory_text

    # ======================================================
    # CONTEXT SECTION
    # ======================================================
    def build_context_section(self, retrieved_docs):
        if not retrieved_docs:
            return "No context retrieved."

        context_text = ""

        for i, (doc, score) in enumerate(retrieved_docs, start=1):
            filename = doc.get("filename", "Unknown Source")
            text = doc.get("text", "")
            chunk_id = doc.get("chunk_id", "N/A")

            context_text += (
                f"\n[SOURCE {i}] {filename}\n"
                f"[CHUNK] {chunk_id}\n"
                f"[RERANK SCORE] {score:.4f}\n"
                f"{text}\n"
            )

        return context_text

    # ======================================================
    # FINAL PROMPT BUILDER
    # ======================================================

    def build_prompt(
        self,
        mode,
        query,
        retrieved_docs,
        memory_rows
    ):

        # Select prompt type

        system_prompt = self.prompts.get(
            mode,
            self.prompts["rag"]
        )

        # Build memory section

        memory_section = self.build_memory_section(
            memory_rows
        )

        # Build context section

        context_section = self.build_context_section(
            retrieved_docs
        )

        # Final prompt

        prompt = f"""
{system_prompt}

==================================================
CONVERSATION HISTORY
==================================================

{memory_section}

==================================================
RETRIEVED KNOWLEDGE
==================================================

{context_section}

==================================================
USER QUESTION
==================================================

{query}

==================================================
ANSWER
==================================================
"""

        return prompt