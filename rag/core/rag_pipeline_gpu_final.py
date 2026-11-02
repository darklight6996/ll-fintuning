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
    GPU_MODEL_NAME,
    GPU_MAX_INPUT_TOKENS,
    GPU_MAX_NEW_TOKENS,
    GPU_SESSION_ID
)


def main():
    print("=" * 60)
    print("GPU RAG Pipeline")
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
        session_id=GPU_SESSION_ID,
        db_path=DATABASE_PATH
    )

    print("Loading Prompt Manager...")
    prompt_manager = PromptManager()

    print("Loading GPU model...")
    llm = ModelRouter(
        model_name=GPU_MODEL_NAME,
        mode="gpu",
        max_input_tokens=GPU_MAX_INPUT_TOKENS,
        max_new_tokens=GPU_MAX_NEW_TOKENS
    )

    print("System Ready")

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue

        if query.lower() == "exit":
            memory.close()
            print("Exiting GPU pipeline.")
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

        ranked_docs = rag.retrieve(query)
        memory_rows = memory.get_recent(MEMORY_WINDOW)

        prompt = prompt_manager.build_prompt(
            mode="rag",
            query=query,
            retrieved_docs=ranked_docs,
            memory_rows=memory_rows
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