"""Static inventory only: no imports, robot execution, or runtime ABC claims.

python tools/extract_coraplex_inventory.py /path/to/pinned/repository output.json
Scope is canonical coraplex/robot_plans actions and motions; external mixins
are named but not resolved. 'concrete_candidate' means no abstract methods
remain in the locally inspected inheritance closure, not runtime eligibility.
"""
import ast
import hashlib
import json
import sys
from pathlib import Path


def extract(repository):
    repository = Path(repository)
    root = repository / 'coraplex/src/coraplex/robot_plans'
    classes = {}
    for category in ['actions', 'motions']:
        for path in sorted((root / category).rglob('*.py')):
            for node in ast.parse(path.read_text()).body:
                if not isinstance(node, ast.ClassDef):
                    continue
                name = node.name
                if name in classes:
                    raise ValueError('Ambiguous class name: ' + name)
                methods = {m.name: any('abstractmethod' in ast.unparse(d) for d in m.decorator_list)
                           for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
                fields = [{'name': n.target.id, 'annotation': ast.unparse(n.annotation),
                           'default_source': ast.unparse(n.value) if n.value else None}
                          for n in node.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
                classes[name] = {'name': name, 'category': category,
                    'path': str(path.relative_to(repository)), 'line': node.lineno,
                    'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'bases': [ast.unparse(b).split('.')[-1] for b in node.bases],
                    'methods': methods, 'declared_fields': fields,
                    'local_calls': sorted({ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)})}

    def ancestry(name, trail=()):
        if name in trail:
            raise ValueError('Inheritance cycle')
        return set(classes[name]['bases']) | set().union(*(ancestry(b, trail + (name,)) for b in classes[name]['bases'] if b in classes))

    def abstract(name):
        pending = set().union(*(abstract(b) for b in classes[name]['bases'] if b in classes))
        for method, is_abstract in classes[name]['methods'].items():
            if is_abstract:
                pending.add(method)
            else:
                pending.discard(method)
        return pending

    def fields(name):
        found = {}
        for base in reversed(classes[name]['bases']):
            if base in classes:
                found.update({f['name']: f for f in fields(base)})
        found.update({f['name']: f for f in classes[name]['declared_fields']})
        return list(found.values())

    for name, item in classes.items():
        family = 'ActionDescription' if item['category'] == 'actions' else 'BaseMotion'
        item['family_member'] = family in ancestry(name)
        item['unimplemented_local_abstract_methods'] = sorted(abstract(name))
        item['concrete_candidate'] = item['family_member'] and not abstract(name)
        item['fields_in_inspected_inheritance'] = fields(name)
    return sorted(classes.values(), key=lambda c: (c['category'], c['path'], c['line']))


if __name__ == '__main__':
    result = extract(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'static_action_candidates': sum(c['concrete_candidate'] and c['category'] == 'actions' for c in result),
                      'static_motion_candidates': sum(c['concrete_candidate'] and c['category'] == 'motions' for c in result)}))
