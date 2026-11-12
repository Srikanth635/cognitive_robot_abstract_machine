"""Bounded offline HermiT classification. Requires owlready2 0.51 and Java.

Preserves all asserted candidate files; writes only validation reports.
"""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORKER = '''
from pathlib import Path
import json
import owlready2 as o
import owlready2.reasoning as r
r.JAVA_MEMORY = 2048
w=o.World()
path=Path(__import__('sys').argv[1])
w.get_ontology(path.as_uri()).load()
try:
    o.sync_reasoner(w,debug=1)
    result={'status':'completed','consistent':True,'unsatisfiable_classes':sorted(str(c.iri) for c in w.inconsistent_classes())}
except o.OwlReadyInconsistentOntologyError:
    result={'status':'completed','consistent':False}
print('RESULT_JSON:'+json.dumps(result))
'''

if __name__ == '__main__':
    start=time.monotonic()
    try:
        p=subprocess.Popen([sys.executable,'-c',WORKER,str(ROOT/'candidates/aicor-unified-dul-offline.owl')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        p.stdout, p.stderr = p.communicate(timeout=90)
        (ROOT/'reports/hermit.log').write_text(p.stdout+'\n'+p.stderr)
        lines=[s for s in p.stdout.splitlines() if s.startswith('RESULT_JSON:')]
        result=json.loads(lines[-1][12:]) if lines else {'status':'error','returncode':p.returncode,'see':'hermit.log'}
    except subprocess.TimeoutExpired as e:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        result={'status':'timeout','limit_seconds':90,'consistent':None}
        (ROOT/'reports/hermit.log').write_text(str(e.stdout or '')+'\n'+str(e.stderr or ''))
    result['elapsed_seconds']=round(time.monotonic()-start,2)
    result['scope']='Offline candidate plus supplied DUL; no SWRL, simulation or code-alignment validation.'
    (ROOT/'reports/hermit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
