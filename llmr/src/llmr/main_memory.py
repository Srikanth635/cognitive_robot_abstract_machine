"""
main_cram.py - Entry point for CRAM generation workflow
"""

from llmr.workflows.graphs.enhanced_ad_graph import run_with_cache

def generate_cram_plan(instruction: str, user_id: str = "default_user", thread_id: str = "cram_001"):
    """
    Generate a CRAM plan for a given instruction.
    Uses semantic cache + long-term memory.
    """

    print(f"\n{'='*60}")
    print(f"  CRAM Generator | user={user_id} thread={thread_id}")
    print(f"{'='*60}")
    print(f"\nInstruction: {instruction}\n")

    # Use the cache wrapper
    result = run_with_cache(instruction, user_id=user_id, thread_id=thread_id)

    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Action cores: {result.get('action_core', [])}")
    print(f"\nCRAM plans:")
    for i, plan in enumerate(result.get('cram_plan_response', []), 1):
        print(f"\nPlan {i}:")
        print(plan)

    return result


if __name__ == "__main__":
    # Example usage
    user = "user_123"

    # Three runs to see all behaviors:

    print("\n" + "=" * 70)
    print("TEST 1: First instruction (cache miss + no memories)")
    print("=" * 70)
    res1 = generate_cram_plan(instruction="Pick up the red mug", user_id="user_123",thread_id= "t1")

    input("\nPress Enter...")

    print("\n" + "=" * 70)
    print("TEST 2: Similar instruction (cache hit + graph skipped)")
    print("=" * 70)
    res2 = generate_cram_plan(instruction="Pick up the red mug", user_id="user_123",thread_id= "t2")  # ← Same/similar

    input("\nPress Enter...")

    print("\n" + "=" * 70)
    print("TEST 3: Different instruction (cache miss + memories loaded!)")
    print("=" * 70)
    res3 = generate_cram_plan(instruction="Pour milk into bowl", user_id="user_123",thread_id= "t3")  # ← Different

    input("\nPress Enter...")

    print("\n" + "=" * 70)
    print("TEST 3: Different instruction (cache hit + graph skipped)")
    print("=" * 70)
    res4 = generate_cram_plan(instruction="Pick up the jar from the counter table", user_id="user_123", thread_id="t4")  # ← Same/similar

    # # Similar instruction - cache hit, instant
    # result2 = generate_cram_plan(
    #     instruction="add milk to cup",
    #     user_id="user_123",
    #     thread_id="session_002"
    # )