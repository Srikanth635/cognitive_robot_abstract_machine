"""Verify wheel-only imports and a real HermiT candidate review after fresh compilation."""

import importlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def main():
    wheel, source = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    with TemporaryDirectory(prefix="krrood-wheel14-") as temporary:
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
        from krrood_ontology.reasoning import (
            CandidateReasoner,
            IsolatedReasoner,
            ReasonerKind,
        )
        from krrood_ontology.sessions import OntologySystem

        assert Path(krrood_ontology.__file__).is_relative_to(runtime)
        directory = root / "generated"
        OntologyCompiler().compile(
            OntologyInput(source), CompilationOptions(directory, "wheel14_model")
        )
        sys.path.insert(0, str(directory))
        model = importlib.import_module("wheel14_model").MODEL
        preparer = CandidatePreparer.from_manifest(
            model,
            directory / "manifest.json",
            schema_id="urn:wheel:schema",
            base_revision="r0",
        )
        constructor = CandidateConstructor(preparer, directory, "wheel14_model")
        checker = CandidateReasoner(constructor, IsolatedReasoner(ReasonerKind.HERMIT))
        proposal = Proposal(
            (CreateEntity("create", "entity", (next(iter(sorted(model.classes))),)),)
        )
        session = OntologySystem(preparer.context).begin_candidate(proposal)
        result = checker.review(session.snapshot)
        stage = next(r for r in result.report.results if r.stage is CheckStage.REASONER)
        assert stage.status is CheckStatus.PASS, result.report.to_dict()
        assert result.evidence.consistent is True
        assert result.construction.roundtrip_equal is True
        print(
            json.dumps(
                {
                    "wheel_import": "passed",
                    "fresh_compilation": "passed",
                    "native_construction": "passed",
                    "hermit_consistency": "pass",
                    "consistency": True,
                    "compiler_phase": json.loads(
                        (directory / "manifest.json").read_text()
                    )["phase"],
                    "engine_version": result.evidence.engine_version,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
