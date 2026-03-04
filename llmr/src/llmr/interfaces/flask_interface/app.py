"""
Flask Application for Interactive Data Visualization
A split-view interface for data generation and visualization
"""

from flask import Flask, render_template, request, jsonify
import json
import os
import signal
import subprocess
import time
from llmr.workflows.graphs.enhanced_ad_graph import run_with_cache
from llmr.workflows.graphs.models_graph import model_reasoning_graph
from llmr.interfaces.flask_interface.data_integrator import DynamicActionIntegrator
from llmr.interfaces.flask_interface.enrichment import enrich_action_sequence, extract_action_sequence
from llmr.interfaces.flask_interface.ontology_refs import ActionSequence
from llmr.workflows.agents.pycram_mapper import pycram_mapper_graph
import requests

app = Flask(__name__)

ins = ""
final_out = []
myout = []

# Global variables for processes
rviz_process = None
vnc_process = None
novnc_process = None

def runners(input_instruction : str):
    out = run_with_cache(input_instruction, user_id="flask_user", thread_id="flask_session")

    global myout
    myout.append(out)

    config = {"configurable" : {"thread_id" : 1}}
    action_core = out['action_core']
    action_core_attributes = out['action_core_attributes']
    enriched_action_core_attributes = out['enriched_action_core_attributes']
    cram_plan = out['cram_plan_response']
    intents = out['intents']['instructions']
    print("cram_plan to MODEL GRAPH : ", cram_plan)
    atomic_instructions = []
    for intent in intents:
        atomic_instructions.append(intent['atomic_instruction'])

    print(type(action_core), type(action_core_attributes), type(enriched_action_core_attributes), type(cram_plan), type(intents))
    if list(action_core):
        for ai, ac, eca, cplan in zip(atomic_instructions, action_core_attributes, enriched_action_core_attributes, cram_plan):
            final_graph_state = model_reasoning_graph.invoke({"instruction": ai,
                                                    "action_core": ac,
                                                    "enriched_action_core_attributes": eca,
                                                    "cram_plan_response": cplan, "intents": intents}, config=config, stream_mode="updates")

            print("Final Graph State :", final_graph_state)
            flanagan = model_reasoning_graph.get_state(config).values["flanagan"]
            framenet_model = model_reasoning_graph.get_state(config).values["framenet_model"]
            new_out = {}
            global final_out
            try:
                flanagan_json = json.loads(flanagan)
                framenet_model_json = json.loads(framenet_model)
                print("Parsed models output normally")
                new_out = {
                    "framenet": framenet_model_json,
                    "flanagan": flanagan_json
                }
                final_out.append(new_out)
            except:
                print("Parsed models output with strings")
                new_out = {
                    "framenet": framenet_model,
                    "flanagan": flanagan,
                }
                final_out.append(new_out)



@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')


@app.route('/generate_all', methods=['POST'])
def generate_all():
    """
    API endpoint to generate both data and ontology based on user instruction
    Expects JSON payload: {"instruction": "user input"}
    Returns: {"success": bool, "visualizer_data": dict, "ontology_data": dict, "error": str}
    """
    global final_out
    global myout

    # Reset globals
    final_out = []
    myout = []

    try:
        payload = request.get_json()
        instruction = payload.get('instruction', '')

        if not instruction:
            return jsonify({
                'success': False,
                'error': 'No instruction provided'
            }), 400
        global ins
        ins = instruction

        # Step 1: Run the main data generation
        print("Step 1: Generating visualizer data...")
        runners(input_instruction=instruction)
        integrator = DynamicActionIntegrator()
        actions = integrator.integrate(final_out, myout)
        print(f"Created {len(actions)} actions")

        file_path = "dynamic_actions.json"
        integrator.export_to_json(file_path)

        visualizer_data = None
        with open(f"{file_path}", 'r') as json_file:
            visualizer_data = json.load(json_file)

        # Step 2: Generate ontology data using the results from step 1
        print("Step 2: Generating ontology data...")
        SYSTEM_PROMPT = """
        You are an ontology-grounded action extractor.

        You MUST output a single JSON object that conforms exactly
        to the provided JSON Schema.

        Output rules:
        - Output JSON only. No explanations.
        - Use references only: {id, ontology_class, label}.
        - Participants must be RoleAssignments:
            { participant, role, related_to? }.
        - Use RoleRef as:
            { "id": "role-<n>", "ontology_class": "Role", "label": "<n>" }
          where <n> is one of:
            agent, theme, source, destination, instrument.

        Identifier and label conventions (IMPORTANT):

        - Every referenced entity MUST have:
            * label: a single lowercase word
              - use snake_case if needed (e.g., kitchen_counter, milk_bottle)
              - no spaces, no punctuation

            * id: derived from the label, in the form:
              <label>_<number>

        - Numbering rules:
            * Start numbering at 1 for each distinct label.
            * Increment only when a NEW entity with the same label is introduced.
            * Reuse the SAME id when the same entity is referred to again
              (including via pronouns like "it", "them", "there").

        Examples:
        - First table mentioned      -> label: "table", id: "table_1"
        - Second table mentioned     -> label: "table", id: "table_2"
        - Milk bottle reused later   -> label: "milk_bottle", id: "milk_bottle_1"
        - Generic agent              -> label: "agent", id: "agent_1"


        Action splitting:
        - If the instruction contains multiple actions joined by "and", "then",
          or similar connectors, output an ActionSequence with one PhysicalAction
          per action, in the order described.

        Task mapping:
        - Map verbs to task classifiers when appropriate.
          For example:
            * "pick up"            -> PickingUp
            * "put down / set down"-> PuttingDown
        - Use the most appropriate task classifier based on the verb phrase.

        Role assignment rules (GENERIC, APPLY TO ALL ACTIONS):
        - Always include:
            * agent: the entity performing the action
              (create a generic PhysicalAgent if none is mentioned)
            * theme: the primary object being acted upon

        - Prepositional phrases create additional RoleAssignments:
            * "from <X>"            -> participant <X> with role "source"
            * "to / into / onto / on <X>" -> participant <X> with role "destination"
            * "with / using <X>"    -> participant <X> with role "instrument"

        - If a prepositional phrase is NOT explicitly present in the instruction,
          do NOT invent that role.
          (Omit it entirely; absence represents null.)

        Representation guidance:
        - Prefer creating a separate RoleAssignment for location entities
          (source / destination / instrument) rather than encoding them only
          via theme.related_to.
        - related_to is optional and should only be used if it adds clarity;
          do not duplicate information unnecessarily.

        Theme linking across actions:
        - The same physical object mentioned across multiple actions
          (including via pronouns like "it", "them") MUST reuse the same id
          and appear as role "theme" in each relevant action.

        """
        SEQUENCE_SCHEMA = ActionSequence.model_json_schema()
        action = extract_action_sequence(instruction=instruction, SYSTEM_PROMPT=SYSTEM_PROMPT,
                                         SEQUENCE_SCHEMA=SEQUENCE_SCHEMA)
        print("ACTION:", action)
        print("myout :", myout)

        ontology_data = enrich_action_sequence(
            action_sequence=action,
            enriched_action_core_attributes=myout[0]['enriched_action_core_attributes'],
        )
        print("ONTOLOGY DATA:", ontology_data)

        # Convert Pydantic models to dict for JSON serialization
        ontology_dict = ontology_data.copy()
        if 'ontology_spine' in ontology_dict and hasattr(ontology_dict['ontology_spine'], 'model_dump'):
            ontology_dict['ontology_spine'] = ontology_dict['ontology_spine'].model_dump()
        elif 'ontology_spine' in ontology_dict and hasattr(ontology_dict['ontology_spine'], 'dict'):
            ontology_dict['ontology_spine'] = ontology_dict['ontology_spine'].dict()

        return jsonify({
            'success': True,
            'visualizer_data': visualizer_data,
            'ontology_data': ontology_dict,
            'instruction': instruction
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/grounding', methods=['POST'])
def grounding():
    """
    API endpoint to take in the belief state and perform grounding.
    """
    try:
        global ins
        global myout

        # Map Symbolic Entities to their Belief state entities
        print("DEBUG : Grounding Context : ",type(myout), myout)
        first_parse = myout[0]
        print("DEBUG : Grounding Context first_parse : ", first_parse)
        parsed_instructions = first_parse['intents']['instructions']
        print("DEBUG : Grounding Context parsed_instructions : ", parsed_instructions)
        atomics = []
        for ins in parsed_instructions:
            atomics.append(ins['atomic_instruction'])

        cram_plans = first_parse['cram_plan_response']
        print("DEBUG : Grounding Context cram_plans : ", cram_plans)

        pycram_graph_output = pycram_mapper_graph.invoke({'atomics': str(atomics), 'cram_plans': str(cram_plans)})

        print({
            'atomics': pycram_graph_output['atomics'],
            'cram_plans': pycram_graph_output['grounded_cram_plans'],
            'action_names': pycram_graph_output['action_names']
        })
        # print(pycram_graph_output)

        return jsonify({
            'success': True,
            'atomics': pycram_graph_output['atomics'],
            'cram_plans': pycram_graph_output['grounded_cram_plans'],
            'action_names': pycram_graph_output['action_names']
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


@app.route('/start_rviz', methods=['POST'])
def start_rviz():
    """Start RViz2 with VNC server and noVNC"""
    global rviz_process, vnc_process, novnc_process

    try:
        # Set display
        display = ":99"

        # Create environment with ROS2 sourced
        env = os.environ.copy()
        env['DISPLAY'] = display

        # Source ROS2 Jazzy
        ros2_setup = '/opt/ros/jazzy/setup.bash'

        # Start Xvfb (virtual display) with larger resolution
        try:
            subprocess.run(['killall', 'Xvfb'], stderr=subprocess.DEVNULL)
        except:
            pass

        xvfb_process = subprocess.Popen([
            'Xvfb', display,
            '-screen', '0', '1600x900x24',  # Changed from 1920x1080 to better fit split screen
            '-ac', '+extension', 'GLX', '+render', '-noreset'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(2)

        # Start RViz2 with sourced ROS2 Jazzy environment
        if rviz_process is None or rviz_process.poll() is not None:
            # Use bash to source ROS2 and then run rviz2
            rviz_command = f'source {ros2_setup} && rviz2'

            rviz_process = subprocess.Popen(
                rviz_command,
                shell=True,
                executable='/bin/bash',
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(3)

        # Start x11vnc
        if vnc_process is None or vnc_process.poll() is not None:
            vnc_process = subprocess.Popen([
                'x11vnc',
                '-display', display,
                '-forever',
                '-shared',
                '-rfbport', '5900',
                '-nopw',
                '-quiet'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)

        # Start noVNC websockify
        if novnc_process is None or novnc_process.poll() is not None:
            novnc_process = subprocess.Popen([
                'websockify',
                '--web=/usr/share/novnc',
                '6080',
                'localhost:5900'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)

        return jsonify({
            'success': True,
            'message': 'RViz2 started with VNC access',
            'vnc_url': 'http://localhost:6080/vnc.html'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/stop_rviz', methods=['POST'])
def stop_rviz():
    """Stop RViz2 and VNC services"""
    global rviz_process, vnc_process, novnc_process

    try:
        # Stop RViz2
        if rviz_process is not None:
            rviz_process.terminate()
            rviz_process.wait(timeout=5)
            rviz_process = None

        # Stop x11vnc
        if vnc_process is not None:
            vnc_process.terminate()
            vnc_process.wait(timeout=5)
            vnc_process = None

        # Stop noVNC
        if novnc_process is not None:
            novnc_process.terminate()
            novnc_process.wait(timeout=5)
            novnc_process = None

        # Kill any remaining processes
        subprocess.run(['killall', 'rviz2'], stderr=subprocess.DEVNULL)
        subprocess.run(['killall', 'x11vnc'], stderr=subprocess.DEVNULL)
        subprocess.run(['killall', 'websockify'], stderr=subprocess.DEVNULL)
        subprocess.run(['killall', 'Xvfb'], stderr=subprocess.DEVNULL)

        return jsonify({'success': True, 'message': 'RViz2 and VNC stopped'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/rviz_status', methods=['GET'])
def rviz_status():
    """Check if RViz2 is running"""
    global rviz_process, vnc_process, novnc_process

    rviz_running = rviz_process is not None and rviz_process.poll() is None
    vnc_running = vnc_process is not None and vnc_process.poll() is None
    novnc_running = novnc_process is not None and novnc_process.poll() is None

    return jsonify({
        'rviz_running': rviz_running,
        'vnc_running': vnc_running,
        'novnc_running': novnc_running,
        'ready': rviz_running and vnc_running and novnc_running
    })


@app.route('/api/tabs')
def get_tabs():
    """Get available tabs configuration"""
    return jsonify({
        'tabs': [
            {'id': 'visualizer', 'name': 'Action Representation', 'active': True},
            {'id': 'ontology', 'name': 'Ontology Tree', 'active': False},
            {'id': 'pycram', 'name': 'Grounded Actions & Simulation', 'active': False}
        ]
    })


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)