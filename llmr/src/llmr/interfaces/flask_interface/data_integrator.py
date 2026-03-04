# action_integration_dynamic.py (UPDATED)

from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
import json


# ============================================================================
# FLEXIBLE BASE CLASS (FIXED)
# ============================================================================

class FlexibleDataContainer:
    """Base class that can hold any attributes dynamically"""

    def __init__(self, data: Dict[str, Any] = None, **kwargs):
        """Initialize with dictionary or keyword arguments"""
        self._data = {}
        self._metadata = {
            'source_type': None,
            'original_keys': []
        }

        if data:
            self._load_from_dict(data)
        if kwargs:
            self._load_from_dict(kwargs)

    def _load_from_dict(self, data: Dict[str, Any]):
        """Load attributes from dictionary"""
        for key, value in data.items():
            self._data[key] = value
            self._metadata['original_keys'].append(key)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access"""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any):
        """Allow attribute-style setting"""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def get(self, key: str, default=None) -> Any:
        """Dictionary-style get"""
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        """Dictionary-style set"""
        self._data[key] = value
        if key not in self._metadata['original_keys']:
            self._metadata['original_keys'].append(key)

    def has(self, key: str) -> bool:
        """Check if key exists"""
        return key in self._data

    def keys(self) -> List[str]:
        """Get all keys"""
        return list(self._data.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - FIXED VERSION"""
        result = {}
        for key, value in self._data.items():
            result[key] = self._convert_to_serializable(value)
        return result

    def _convert_to_serializable(self, value: Any) -> Any:
        """Recursively convert values to JSON-serializable format"""
        if isinstance(value, FlexibleDataContainer):
            return value.to_dict()
        elif isinstance(value, dict):
            return {k: self._convert_to_serializable(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._convert_to_serializable(item) for item in value]
        elif isinstance(value, tuple):
            return tuple(self._convert_to_serializable(item) for item in value)
        elif isinstance(value, set):
            return list(value)
        elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, type(None))):
            # Handle custom objects
            try:
                return {k: self._convert_to_serializable(v) for k, v in value.__dict__.items() if not k.startswith('_')}
            except:
                return str(value)
        else:
            # Primitive types or None
            return value

    def __repr__(self):
        return f"{self.__class__.__name__}({list(self._data.keys())})"


# ============================================================================
# DYNAMIC LAYER CLASSES (No changes needed, but included for completeness)
# ============================================================================

class DynamicSemanticLayer(FlexibleDataContainer):
    """Semantic layer with flexible attributes"""

    @classmethod
    def from_json_string(cls, json_str: str) -> 'DynamicSemanticLayer':
        """Parse from JSON string"""
        try:
            data = json.loads(json_str) if isinstance(json_str, str) else json_str
            instance = cls(data)
            instance._metadata['source_type'] = 'framenet'
            return instance
        except Exception as e:
            print(f"Warning: Failed to parse semantic data: {e}")
            return cls()

    def get_core_element(self, element: str, default=None):
        """Get core element flexibly"""
        if self.has('core'):
            core = self.get('core', {})
            if isinstance(core, dict):
                return core.get(element, default)
        if self.has('core_elements'):
            return self.get('core_elements', {}).get(element, default)
        return self.get(element, default)

    def get_peripheral_element(self, element: str, default=None):
        """Get peripheral element flexibly"""
        if self.has('peripheral'):
            peripheral = self.get('peripheral', {})
            if isinstance(peripheral, dict):
                return peripheral.get(element, default)
        if self.has('peripheral_elements'):
            return self.get('peripheral_elements', {}).get(element, default)
        return self.get(element, default)


class DynamicIntentLayer(FlexibleDataContainer):
    """Intent layer with flexible attributes"""

    @classmethod
    def from_intent_dict(cls, intent_dict: Dict) -> 'DynamicIntentLayer':
        """Parse from intent dictionary"""
        instance = cls(intent_dict)
        instance._metadata['source_type'] = 'intent'
        return instance

    def get_role(self, role: str, default=None):
        """Get role flexibly"""
        if self.has('roles'):
            roles = self.get('roles', {})
            if isinstance(roles, dict):
                return roles.get(role, default)
        return self.get(role, default)

    def get_parameter(self, param: str, default=None):
        """Get parameter flexibly"""
        if self.has('parameters'):
            params = self.get('parameters', {})
            if isinstance(params, dict):
                return params.get(param, default)
        return self.get(param, default)


class DynamicObject(FlexibleDataContainer):
    """Object with flexible properties"""

    @classmethod
    def from_json_string(cls, json_str: str) -> 'DynamicObject':
        """Parse from JSON string"""
        try:
            data = json.loads(json_str) if isinstance(json_str, str) else json_str
            instance = cls()

            # Extract name flexibly
            name = (data.get('obj_to_be_grabbed') or
                    data.get('obj_to_be_put') or
                    data.get('object_name') or
                    data.get('name') or
                    data.get('object') or
                    'unknown')
            instance.set('name', name)

            # Extract properties flexibly
            props = (data.get('obj_to_be_grabbed_props') or
                     data.get('obj_to_be_put_props') or
                     data.get('properties') or
                     data.get('props') or
                     {})
            instance.set('properties', props)

            # Store any other attributes
            for key, value in data.items():
                if key not in ['obj_to_be_grabbed', 'obj_to_be_put',
                               'obj_to_be_grabbed_props', 'obj_to_be_put_props']:
                    instance.set(key, value)

            instance._metadata['source_type'] = 'object'
            return instance
        except Exception as e:
            print(f"Warning: Failed to parse object data: {e}")
            return cls()

    def get_property(self, prop: str, default=None):
        """Get property flexibly"""
        if self.has('properties'):
            props = self.get('properties', {})
            if isinstance(props, dict):
                return props.get(prop, default)
        return default


class DynamicPhase(FlexibleDataContainer):
    """Phase with flexible attributes"""

    @classmethod
    def from_phase_dict(cls, phase_dict: Dict, phase_id: str = None) -> 'DynamicPhase':
        """Parse from phase dictionary"""
        instance = cls(phase_dict)

        # Set phase_id if provided
        if phase_id:
            instance.set('phase_id', phase_id)
        elif not instance.has('phase_id'):
            instance.set('phase_id', f"phase_{id(instance)}")

        # Normalize common attribute names
        if not instance.has('phase_name') and instance.has('phase'):
            instance.set('phase_name', instance.get('phase'))

        instance._metadata['source_type'] = 'phase'
        return instance

    def check_preconditions(self, current_state: Dict[str, Any]) -> bool:
        """Verify preconditions flexibly"""
        preconditions = self.get('preconditions', {})
        if not isinstance(preconditions, dict):
            return True

        for key, required_value in preconditions.items():
            if current_state.get(key) != required_value:
                return False
        return True

    def get_expected_force_range(self) -> Optional[List[float]]:
        """Extract force range flexibly"""
        force_dynamics = self.get('force_dynamics', {})
        if isinstance(force_dynamics, dict):
            force_profile = force_dynamics.get('force_profile', {})
            if isinstance(force_profile, dict):
                return (force_profile.get('expected_range_N') or
                        force_profile.get('expected_range') or
                        force_profile.get('force_range'))
        return None

    def get_duration(self) -> Optional[float]:
        """Get duration flexibly"""
        temporal = self.get('temporal_constraints', {})
        if isinstance(temporal, dict):
            return (temporal.get('max_duration_sec') or
                    temporal.get('duration_sec') or
                    temporal.get('max_duration') or
                    temporal.get('duration'))
        return self.get('duration')


# ============================================================================
# DYNAMIC UNIFIED ACTION (UPDATED to_dict removed since inherited)
# ============================================================================

class DynamicUnifiedAction(FlexibleDataContainer):
    """Unified action with flexible attributes"""

    def __init__(self, action_id: str = None, **kwargs):
        super().__init__(**kwargs)

        # Core identifiers
        if action_id:
            self.set('action_id', action_id)

        # Initialize containers
        if not self.has('layers'):
            self.set('layers', {})
        if not self.has('phases'):
            self.set('phases', [])
        if not self.has('objects'):
            self.set('objects', {})
        if not self.has('dependencies'):
            self.set('dependencies', {'requires': [], 'enables': []})

    def add_layer(self, layer_name: str, layer_data: FlexibleDataContainer):
        """Add a representation layer"""
        layers = self.get('layers', {})
        layers[layer_name] = layer_data
        self.set('layers', layers)

    def get_layer(self, layer_name: str) -> Optional[FlexibleDataContainer]:
        """Get a representation layer"""
        layers = self.get('layers', {})
        return layers.get(layer_name)

    def add_phase(self, phase: DynamicPhase):
        """Add a phase"""
        phases = self.get('phases', [])
        phases.append(phase)
        self.set('phases', phases)

    def get_phase(self, phase_identifier: Union[str, int]) -> Optional[DynamicPhase]:
        """Get phase by name, id, or index"""
        phases = self.get('phases', [])

        if isinstance(phase_identifier, int):
            if 0 <= phase_identifier < len(phases):
                return phases[phase_identifier]
        else:
            for phase in phases:
                if (phase.get('phase_name') == phase_identifier or
                        phase.get('phase_id') == phase_identifier or
                        phase.get('phase') == phase_identifier):
                    return phase
        return None

    def add_object(self, key: str, obj: DynamicObject):
        """Add an object"""
        objects = self.get('objects', {})
        objects[key] = obj
        self.set('objects', objects)

    def get_object(self, key: str = 'primary') -> Optional[DynamicObject]:
        """Get object"""
        objects = self.get('objects', {})
        return objects.get(key)

    def get_all_phases(self) -> List[DynamicPhase]:
        """Get all phases"""
        return self.get('phases', [])

    def get_phase_count(self) -> int:
        """Get number of phases"""
        return len(self.get('phases', []))

    def get_total_duration(self) -> float:
        """Calculate total duration"""
        total = 0.0
        for phase in self.get('phases', []):
            duration = phase.get_duration()
            if duration:
                total += duration
        return total

    def validate_consistency(self) -> Dict[str, Any]:
        """Validate consistency across layers"""
        issues = []

        # Check object consistency across layers
        layers = self.get('layers', {})
        object_references = []

        for layer_name, layer in layers.items():
            refs = self._extract_object_references(layer)
            object_references.extend([(layer_name, ref) for ref in refs])

        # Check if object references are consistent
        if len(set([ref for _, ref in object_references])) > 1:
            unique_refs = set([ref for _, ref in object_references])
            if len(unique_refs) > 1:
                normalized = set([self._normalize_name(ref) for ref in unique_refs])
                if len(normalized) > 1:
                    issues.append(f"Object reference mismatch across layers: {object_references}")

        # Check phase continuity
        phases = self.get('phases', [])
        for i in range(len(phases) - 1):
            current_phase = phases[i]
            next_phase = phases[i + 1]

            goal_state = current_phase.get('goal_state', {})
            preconditions = next_phase.get('preconditions', {})

            if isinstance(goal_state, dict) and isinstance(preconditions, dict):
                for key in preconditions:
                    if key in goal_state:
                        if goal_state[key] != preconditions[key]:
                            issues.append(
                                f"State mismatch: {current_phase.get('phase_name')} -> "
                                f"{next_phase.get('phase_name')}: {key}"
                            )

        return {
            'valid': len(issues) == 0,
            'issues': issues
        }

    def _extract_object_references(self, layer: FlexibleDataContainer) -> List[str]:
        """Extract object references from a layer"""
        refs = []

        possible_keys = [
            'theme', 'patient', 'object', 'target_object',
            'obj_to_be_grabbed', 'obj_to_be_put', 'object_name'
        ]

        for key in possible_keys:
            value = layer.get(key)
            if value and isinstance(value, str):
                refs.append(value)

            for nested_key in ['core', 'core_elements', 'roles']:
                nested = layer.get(nested_key, {})
                if isinstance(nested, dict):
                    nested_value = nested.get(key)
                    if nested_value and isinstance(nested_value, str):
                        refs.append(nested_value)

        return refs

    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison"""
        return name.replace('_', ' ').replace('-', ' ').lower().strip()


# ============================================================================
# DYNAMIC INTEGRATOR (No changes needed)
# ============================================================================

class DynamicActionIntegrator:
    """Integrator that adapts to changing data structures"""

    def __init__(self):
        self.actions: List[DynamicUnifiedAction] = []
        self.config = {
            'object_key_patterns': [
                'obj_to_be_grabbed', 'obj_to_be_put', 'object_name',
                'name', 'object', 'target_object', 'theme', 'patient'
            ],
            'instruction_key_patterns': [
                'instruction', 'atomic_instruction', 'command',
                'task', 'description'
            ],
            'action_type_keywords': {
                'PickingUp': ['pick', 'grab', 'get', 'grasp', 'take'],
                'Placing': ['place', 'put', 'set', 'position'],
                'Moving': ['move', 'transport', 'carry'],
                'Pushing': ['push', 'press'],
                'Pulling': ['pull', 'drag'],
            }
        }

    def integrate(
            self,
            primary_data: List[Dict],
            secondary_data: List[Dict] = None,
            auto_match: bool = True
    ) -> List[DynamicUnifiedAction]:
        """
        Integrate data sources flexibly - FIXED to handle atomic instructions
        """
        self.actions = []

        # Build lookup for secondary data at ATOMIC instruction level
        secondary_lookup = {}
        if secondary_data and auto_match:
            secondary_lookup = self._build_atomic_lookup(secondary_data)

        # Process primary data
        for item in primary_data:
            action = self._process_item(item, secondary_lookup)
            if action:
                self.actions.append(action)

        # Establish dependencies
        self._establish_dependencies()

        return self.actions

    def _build_atomic_lookup(self, data: List[Dict]) -> Dict[str, Dict]:
        """
        Build lookup dictionary at ATOMIC instruction level
        """
        lookup = {}

        for item in data:
            if 'intents' in item:
                intents_data = item['intents']
                if isinstance(intents_data, dict) and 'instructions' in intents_data:
                    instructions = intents_data['instructions']
                    if isinstance(instructions, list):
                        for intent_dict in instructions:
                            atomic_inst = intent_dict.get('atomic_instruction')
                            if atomic_inst:
                                normalized = self._normalize_instruction(atomic_inst)
                                lookup[normalized] = {
                                    'parent_item': item,
                                    'intent_dict': intent_dict
                                }
            else:
                instruction = self._extract_instruction(item)
                if instruction:
                    lookup[self._normalize_instruction(instruction)] = {
                        'parent_item': item,
                        'intent_dict': None
                    }

        return lookup

    def _extract_instruction(self, item: Dict) -> Optional[str]:
        """Extract instruction flexibly"""
        for key in self.config['instruction_key_patterns']:
            if key in item:
                value = item[key]
                if isinstance(value, str):
                    return value

        for nested_key in ['flanagan', 'intent', 'intents']:
            if nested_key in item:
                nested = item[nested_key]
                if isinstance(nested, dict):
                    for key in self.config['instruction_key_patterns']:
                        if key in nested:
                            value = nested[key]
                            if isinstance(value, str):
                                return value

        return None

    def _normalize_instruction(self, instruction: str) -> str:
        """Normalize instruction for matching"""
        normalized = instruction.lower().strip()
        for article in ['the ', 'a ', 'an ']:
            if normalized.startswith(article):
                normalized = normalized[len(article):]
        return normalized

    def _process_item(
            self,
            item: Dict,
            secondary_lookup: Dict[str, Dict]
    ) -> Optional[DynamicUnifiedAction]:
        """Process a single item into unified action"""

        action_id = self._extract_action_id(item)
        action = DynamicUnifiedAction(action_id=action_id)

        instruction = self._extract_instruction(item)
        if instruction:
            action.set('atomic_instruction', instruction)
            action.set('instruction', instruction)

        action_type = self._infer_action_type(item, instruction)
        action.set('action_type', action_type)

        # Process primary data
        self._process_framenet(item, action)
        self._process_flanagan(item, action)
        self._process_intent(item, action)
        self._process_objects(item, action)
        self._process_cram(item, action)

        # Try to match with secondary data
        if instruction and secondary_lookup:
            normalized_instruction = self._normalize_instruction(instruction)

            if normalized_instruction in secondary_lookup:
                match_data = secondary_lookup[normalized_instruction]
                self._merge_secondary_data(action, match_data)
            else:
                for lookup_key, match_data in secondary_lookup.items():
                    if self._instructions_match(normalized_instruction, lookup_key):
                        self._merge_secondary_data(action, match_data)
                        break

        return action if action.keys() else None

    def _instructions_match(self, inst1: str, inst2: str) -> bool:
        """Check if two instructions refer to the same action"""
        if inst1 == inst2:
            return True

        if inst1 in inst2 or inst2 in inst1:
            return True

        words1 = set(inst1.split())
        words2 = set(inst2.split())

        common_words = {'the', 'a', 'an', 'on', 'from', 'to', 'it'}
        words1 -= common_words
        words2 -= common_words

        if words1 and words2:
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            similarity = intersection / union
            return similarity > 0.7

        return False

    def _merge_secondary_data(self, action: DynamicUnifiedAction, match_data: Dict):
        """Merge secondary data into action"""
        parent_item = match_data['parent_item']
        intent_dict = match_data['intent_dict']

        if intent_dict:
            if 'action_id' in intent_dict:
                action.set('action_id', intent_dict['action_id'])

            intent = DynamicIntentLayer.from_intent_dict(intent_dict)
            action.add_layer('intent', intent)

        self._process_objects_for_action(parent_item, action, intent_dict)
        self._process_cram_for_action(parent_item, action)

    def _process_objects_for_action(
            self,
            parent_item: Dict,
            action: DynamicUnifiedAction,
            intent_dict: Optional[Dict]
    ):
        """Process objects matching this specific action"""
        action_type = action.get('action_type', '')

        enriched = parent_item.get('enriched_action_core_attributes', [])
        if enriched:
            for idx, attr_json in enumerate(enriched):
                try:
                    data = json.loads(attr_json) if isinstance(attr_json, str) else attr_json

                    if action_type == 'PickingUp' and 'obj_to_be_grabbed' in data:
                        obj = DynamicObject.from_json_string(attr_json)
                        action.add_object('primary', obj)
                        break
                    elif action_type == 'Placing' and 'obj_to_be_put' in data:
                        obj = DynamicObject.from_json_string(attr_json)
                        action.add_object('primary', obj)
                        break
                except Exception as e:
                    print(f"Warning: Failed to parse object: {e}")

        if not action.get('objects'):
            core_attrs = parent_item.get('action_core_attributes', [])
            for idx, attr_json in enumerate(core_attrs):
                try:
                    data = json.loads(attr_json) if isinstance(attr_json, str) else attr_json

                    if action_type == 'PickingUp' and 'obj_to_be_grabbed' in data:
                        obj = DynamicObject.from_json_string(attr_json)
                        action.add_object('primary', obj)
                        break
                    elif action_type == 'Placing' and 'obj_to_be_put' in data:
                        obj = DynamicObject.from_json_string(attr_json)
                        action.add_object('primary', obj)
                        break
                except Exception as e:
                    print(f"Warning: Failed to parse object: {e}")

    def _process_cram_for_action(self, parent_item: Dict, action: DynamicUnifiedAction):
        """Process CRAM plan matching this specific action"""
        cram_data = parent_item.get('cram_plan_response') or parent_item.get('cram_plan')
        if cram_data and isinstance(cram_data, list):
            action_type = action.get('action_type', '')
            for cram_item in cram_data:
                if action_type in str(cram_item):
                    action.set('cram_plan', cram_item)
                    break

    def _extract_action_id(self, item: Dict) -> str:
        """Extract action ID flexibly"""
        if 'action_id' in item:
            return str(item['action_id'])

        for nested_key in ['intent', 'intents', 'metadata']:
            if nested_key in item:
                nested = item[nested_key]
                if isinstance(nested, dict) and 'action_id' in nested:
                    return str(nested['action_id'])
                elif isinstance(nested, dict) and 'instructions' in nested:
                    instructions = nested['instructions']
                    if isinstance(instructions, list) and len(instructions) > 0:
                        first = instructions[0]
                        if isinstance(first, dict) and 'action_id' in first:
                            return str(first['action_id'])

        return f"action_{len(self.actions) + 1:08d}"

    def _infer_action_type(self, item: Dict, instruction: str = None) -> str:
        """Infer action type flexibly"""
        if 'action_type' in item:
            return str(item['action_type'])

        for nested_key in ['intent', 'intents', 'semantic']:
            if nested_key in item:
                nested = item[nested_key]
                if isinstance(nested, dict):
                    if 'intent' in nested:
                        intent_str = str(nested['intent'])
                        if 'PICK' in intent_str.upper():
                            return 'PickingUp'
                        elif 'PLACE' in intent_str.upper():
                            return 'Placing'
                        return intent_str.replace('IntentType.', '')
                    if 'frame' in nested:
                        frame = str(nested['frame']).lower()
                        if 'get' in frame or 'pick' in frame:
                            return 'PickingUp'
                        elif 'plac' in frame or 'put' in frame:
                            return 'Placing'

        if instruction:
            instruction_lower = instruction.lower()
            for action_type, keywords in self.config['action_type_keywords'].items():
                for keyword in keywords:
                    if keyword in instruction_lower:
                        return action_type

        return 'Unknown'

    def _process_framenet(self, item: Dict, action: DynamicUnifiedAction):
        """Process FrameNet data flexibly"""
        framenet_data = item.get('framenet')
        if framenet_data:
            semantic = DynamicSemanticLayer.from_json_string(framenet_data)
            action.add_layer('semantic', semantic)

    def _process_flanagan(self, item: Dict, action: DynamicUnifiedAction):
        """Process Flanagan phase data flexibly"""
        flanagan_data = item.get('flanagan', {})

        phases = flanagan_data.get('phases', [])
        for idx, phase_dict in enumerate(phases):
            phase_id = f"phase_{action.get_phase_count() + 1:03d}"
            phase = DynamicPhase.from_phase_dict(phase_dict, phase_id)
            action.add_phase(phase)

        if flanagan_data and len(flanagan_data) > 1:
            flanagan_layer = FlexibleDataContainer(flanagan_data)
            action.add_layer('flanagan', flanagan_layer)

    def _process_intent(self, item: Dict, action: DynamicUnifiedAction):
        """Process intent data flexibly - only for direct items, not lookups"""
        if 'intent' in item and not 'intents' in item:
            intent = DynamicIntentLayer.from_intent_dict(item)
            action.add_layer('intent', intent)

    def _process_objects(self, item: Dict, action: DynamicUnifiedAction):
        """Process object data flexibly - for primary data only"""
        pass

    def _process_cram(self, item: Dict, action: DynamicUnifiedAction):
        """Process CRAM plan data flexibly - for primary data only"""
        pass

    def _establish_dependencies(self):
        """Establish dependencies between actions"""
        for i in range(len(self.actions) - 1):
            current_action = self.actions[i]
            next_action = self.actions[i + 1]

            current_deps = current_action.get('dependencies', {'requires': [], 'enables': []})
            next_deps = next_action.get('dependencies', {'requires': [], 'enables': []})

            current_deps['enables'].append(next_action.get('action_id'))
            next_deps['requires'].append(current_action.get('action_id'))

            current_action.set('dependencies', current_deps)
            next_action.set('dependencies', next_deps)

    def get_action_by_id(self, action_id: str) -> Optional[DynamicUnifiedAction]:
        """Retrieve action by ID"""
        for action in self.actions:
            if action.get('action_id') == action_id:
                return action
        return None

    def export_to_json(self, filepath: str):
        """Export all actions to JSON"""
        data = {
            'actions': [action.to_dict() for action in self.actions]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully exported {len(self.actions)} actions to {filepath}")

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        total_phases = sum(action.get_phase_count() for action in self.actions)
        total_duration = sum(action.get_total_duration() for action in self.actions)

        objects = set()
        for action in self.actions:
            for obj_key, obj in action.get('objects', {}).items():
                if hasattr(obj, 'get'):
                    name = obj.get('name')
                    if name:
                        objects.add(name)

        return {
            'total_actions': len(self.actions),
            'total_phases': total_phases,
            'estimated_duration_sec': total_duration,
            'objects_involved': list(objects),
            'action_sequence': [
                {
                    'action_id': a.get('action_id'),
                    'type': a.get('action_type'),
                    'instruction': a.get('atomic_instruction') or a.get('instruction')
                } for a in self.actions
            ]
        }


# ============================================================================
# DYNAMIC QUERY INTERFACE (No changes needed)
# ============================================================================

class DynamicQueryInterface:
    """Query interface for dynamic actions"""

    def __init__(self, integrator: DynamicActionIntegrator):
        self.integrator = integrator

    def query(self, action_id: str, path: str, default=None) -> Any:
        """
        Generic query using dot notation path

        Examples:
            query('action_001', 'layers.semantic.frame')
            query('action_001', 'phases.0.force_dynamics.force_profile')
            query('action_001', 'objects.primary.properties.material')
        """
        action = self.integrator.get_action_by_id(action_id)
        if not action:
            return default

        parts = path.split('.')
        current = action

        for part in parts:
            if current is None:
                return default

            if part.isdigit() and isinstance(current, list):
                idx = int(part)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return default
            elif hasattr(current, 'get'):
                current = current.get(part, default)
            elif isinstance(current, dict):
                current = current.get(part, default)
            elif isinstance(current, list):
                return default
            else:
                return default

        return current

    def find_actions_with(self, **criteria) -> List[DynamicUnifiedAction]:
        """
        Find actions matching criteria

        Examples:
            find_actions_with(action_type='PickingUp')
            find_actions_with(has_phase='Grasp')
        """
        results = []
        for action in self.integrator.actions:
            matches = True
            for key, value in criteria.items():
                if key == 'has_phase':
                    if not action.get_phase(value):
                        matches = False
                        break
                elif key == 'has_object':
                    objects = action.get('objects', {})
                    found = False
                    for obj in objects.values():
                        if hasattr(obj, 'get') and obj.get('name') == value:
                            found = True
                            break
                    if not found:
                        matches = False
                        break
                else:
                    if action.get(key) != value:
                        matches = False
                        break

            if matches:
                results.append(action)

        return results

    def get_all_object_names(self) -> List[str]:
        """Get all unique object names across actions"""
        objects = set()
        for action in self.integrator.actions:
            for obj_key, obj in action.get('objects', {}).items():
                if hasattr(obj, 'get'):
                    name = obj.get('name')
                    if name:
                        objects.add(name)
        return sorted(list(objects))

    def get_phase_statistics(self) -> Dict[str, Any]:
        """Get statistics about phases"""
        phase_counts = defaultdict(int)
        total_phases = 0

        for action in self.integrator.actions:
            for phase in action.get_all_phases():
                phase_name = phase.get('phase_name') or phase.get('phase', 'unknown')
                phase_counts[phase_name] += 1
                total_phases += 1

        return {
            'total_phases': total_phases,
            'phase_distribution': dict(phase_counts),
            'unique_phase_types': len(phase_counts)
        }