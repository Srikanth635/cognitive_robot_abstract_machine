"""Check ontology consistency only, not satisfiability of every named class."""
from pathlib import Path
import json
import subprocess
import time
import owlready2.reasoning as reasoning
ROOT = Path(__file__).resolve().parents[1]
start = time.monotonic()
try:
    p = subprocess.run(['java', '-Xmx2048M', '-cp', reasoning._HERMIT_CLASSPATH,
                        'org.semanticweb.HermiT.cli.CommandLine', '-k',
                        (ROOT/'candidates/aicor-unified-dul-offline.owl').as_uri()],
                       capture_output=True, text=True, timeout=90)
    result = {'status':'completed', 'returncode':p.returncode,
              'stdout':p.stdout, 'stderr':p.stderr}
except subprocess.TimeoutExpired:
    result = {'status':'timeout', 'limit_seconds':90}
result['elapsed_seconds'] = round(time.monotonic()-start, 2)
(ROOT/'reports/hermit_consistency.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
