import os
import sys

# Ensure parent directory (rag/) is in sys.path so 'from core...' imports resolve cleanly
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(CORE_DIR)
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

from core.rag_engine import RAGEngine
from core.memory import ChatMemory
from core.prompts import PromptManager
from core.model_router import ModelRouter

from core.config import (
    INDEX_PATH,
    META_PATH,
    DATABASE_PATH,
    TOP_K,
    FINAL_K,
    MEMORY_WINDOW,
    CPU_MODEL_NAME,
    CPU_MAX_INPUT_TOKENS,
    CPU_MAX_NEW_TOKENS,
    CPU_SESSION_ID
)


def main():
    print("=" * 60)
    print("CPU RAG Pipeline")
    print("=" * 60)

    print("Loading RAG Engine...")
    rag = RAGEngine(
        index_path=INDEX_PATH,
        meta_path=META_PATH,
        top_k=TOP_K,
        final_k=FINAL_K
    )

    print("Loading Memory...")
    memory = ChatMemory(
        session_id=CPU_SESSION_ID,
        db_path=DATABASE_PATH
    )

    print("Loading Prompt Manager...")
    prompt_manager = PromptManager()

    print("Loading CPU model...")
    llm = ModelRouter(
        model_name=CPU_MODEL_NAME,
        mode="cpu",
        max_input_tokens=CPU_MAX_INPUT_TOKENS,
        max_new_tokens=CPU_MAX_NEW_TOKENS
    )

    print("System Ready")

    while True:
        try:
            query = input("\nEnter query (exit/reset/history): ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue

        if query.lower() == "exit":
            memory.close()
            print("Exiting CPU pipeline.")
            break

        if query.lower() == "reset":
            memory.reset_session()
            print("Memory cleared.")
            continue

        if query.lower() == "history":
            history = memory.get_full_history()
            print("\n=== FULL HISTORY ===")
            for role, content in history:
                print(f"{role}: {content}")
            print("====================")
            continue

        recent_memory = memory.get_recent(MEMORY_WINDOW)
        retrieved_docs = rag.retrieve(query)

        prompt = prompt_manager.build_prompt(
            mode="rag",
            query=query,
            retrieved_docs=retrieved_docs,
            memory_rows=recent_memory
        )

        answer = llm.generate(prompt)

        memory.add_turn(
            user_message=query,
            assistant_message=answer
        )

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    main()