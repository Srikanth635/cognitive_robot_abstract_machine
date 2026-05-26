"""Verification script for the Agentic LLM ReAct loop."""

import os

# Make iai_pr2_description resolvable via both ROS and ament index lookups.
# Required when running outside the Jupyter kernel (which sources these via kernel_launcher.sh).
_IAI_PR2_INSTALL = "/home/malineni/ros_packages/iai_pr2_install"
_IAI_PR2_SRC = "/home/malineni/ros_packages/iai_pr2/iai_pr2_description"
os.environ["AMENT_PREFIX_PATH"] = _IAI_PR2_INSTALL + ":" + os.environ.get("AMENT_PREFIX_PATH", "")
os.environ["ROS_PACKAGE_PATH"] = _IAI_PR2_SRC + ":" + os.environ.get("ROS_PACKAGE_PATH", "")

from langchain_openai import ChatOpenAI
from agentic_llmr.backend import AgenticLLMBackend
from krrood.entity_query_language.query.match import Match
from agentic_llmr.integrations.world_manager import set_active_world


os.environ["OPENAI_API_KEY"] = "REPLACE_WITH_OPENAI_API_KEY"


class DummyActionClass:
    pass

def run_verification():
    try:
        from uniworld import load_pr2_apartment_world
        print("Loading real PR2 apartment world...")
        world, robot_view, context = load_pr2_apartment_world()
        set_active_world(world, robot_view)
        print("Apartment world loaded successfully.")
    except ImportError:
        print("Warning: Could not import 'uniworld'. Proceeding without a real world.")
        set_active_world(None, None)
    except Exception as e:
        print(f"Warning: Apartment world unavailable ({type(e).__name__}: {e}).")
        print("Falling back to simple PR2 world (table + milk + cereal)...")
        from uniworld import load_pr2_simple_world
        world, robot_view, context = load_pr2_simple_world()
        set_active_world(world, robot_view)
        print("Simple world loaded successfully.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    backend = AgenticLLMBackend(llm=llm)
    
    print("=== TEST 1: Raw String Input ===")
    string_input = "Pick up the milk from the table."
    results_1 = list(backend._evaluate(string_input))
    
    if results_1:
        action_instance = results_1[0]
        print(f"\nFinal Executable Instance:")
        print(f"Type: {type(action_instance)}")
        print(f"Object Designator: {getattr(action_instance, 'object_designator', None)}")
        print(f"Arm: {getattr(action_instance, 'arm', None)}")
        
        # If it's a real PyCRAM object, we can theoretically execute it!
        # action_instance.execute()
    else:
        print("\nNo result returned.")
    
    # print("=== TEST 2: KRROOD Match Input ===")
    # dummy_match = Match(DummyActionClass) # type: ignore
    # results_2 = list(backend._evaluate(dummy_match))
    # print(f"\nFinal Verified Result (Match): {results_2}\n")

if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set OPENAI_API_KEY environment variable to run this test.")
    else:
        run_verification()
