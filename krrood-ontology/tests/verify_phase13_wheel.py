"""Verify wheel-only imports, fresh compilation and candidate worker construction."""

import importlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def main():
    wheel = Path(sys.argv[1]).resolve()
    source = Path(sys.argv[2]).resolve()
    with TemporaryDirectory(prefix="krrood-wheel-") as temporary:
        root = Path(temporary)
        runtime = root / "runtime"
        with ZipFile(wheel) as archive:
            archive.extractall(runtime)
        sys.path.insert(0, str(runtime))

        import krrood_ontology
        from krrood_ontology import CompilationOptions, OntologyCompiler, OntologyInput
        from krrood_ontology.construction import CandidateConstructor
        from krrood_ontology.contracts import (
            CheckStage,
            CheckStatus,
            CreateEntity,
            Proposal,
        )
        from krrood_ontology.preparation import CandidatePreparer
        from krrood_ontology.sessions import OntologySystem

        assert Path(krrood_ontology.__file__).is_relative_to(runtime)
        directory = root / "generated"
        OntologyCompiler().compile(
            OntologyInput(source), CompilationOptions(directory, "wheel_model")
        )
        sys.path.insert(0, str(directory))
        model = importlib.import_module("wheel_model").MODEL
        preparer = CandidatePreparer.from_manifest(
            model,
            directory / "manifest.json",
            schema_id="urn:wheel:schema",
            base_revision="r0",
        )
        constructor = CandidateConstructor(preparer, directory, "wheel_model")
        proposal = Proposal(
            (CreateEntity("create", "entity", (next(iter(sorted(model.classes))),)),)
        )
        session = OntologySystem(preparer.context).begin_candidate(proposal)
        result = constructor.construct(session.snapshot)
        construction = next(
            r for r in result.report.results if r.stage is CheckStage.CONSTRUCTION
        )
        assert construction.status is CheckStatus.PASS, result.report.to_dict()
        assert result.artifact.roundtrip_equal is True
        print(
            json.dumps(
                {
                    "wheel_import": "passed",
                    "fresh_compilation": "passed",
                    "isolated_construction": construction.status.value,
                    "worker_objects": len(result.artifact.objects),
                    "compiler_phase": json.loads(
                        (directory / "manifest.json").read_text()
                    )["phase"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
