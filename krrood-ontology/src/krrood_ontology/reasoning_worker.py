"""Run existing OWL engines in an ephemeral interpreter; never mutate accepted RDF."""

import hashlib
import importlib.metadata
import importlib.util
import os
import pickle
import sys
from pathlib import Path


def owlrl_review(job):
    from owlrl import OWLRL_Semantics

    from krrood_ontology.contracts import (
        CheckStatus,
        Feedback,
        FeedbackCategory,
        Severity,
    )
    from krrood_ontology.reasoning import OWL_RL_CAPABILITIES, ReasonerEvidence

    graph = job.prepared.graph()
    engine = OWLRL_Semantics(graph, axioms=False, daxioms=False)
    engine.closure()
    messages = tuple(sorted(set(engine.error_messages)))
    if messages:
        status, consistent = CheckStatus.FAIL, False
        feedback = tuple(
            Feedback(FeedbackCategory.REASONER, Severity.ERROR, "owlrl_clash", message)
            for message in messages
        )
    else:
        status, consistent = CheckStatus.UNKNOWN, None
        feedback = (
            Feedback(
                FeedbackCategory.REASONER,
                Severity.WARNING,
                "owlrl_no_clashes",
                "OWL RL rules reported no clashes; this diagnostic profile cannot certify ontology consistency",
            ),
        )
    return ReasonerEvidence(
        job.prepared,
        OWL_RL_CAPABILITIES,
        status,
        consistent,
        feedback,
        engine_version=importlib.metadata.version("owlrl"),
        worker_pid=os.getpid(),
    )


def hermit_review(job, temporary: Path):
    from krrood_ontology.contracts import (
        CheckStatus,
        Feedback,
        FeedbackCategory,
        Severity,
    )
    from krrood_ontology.reasoning import (
        HERMIT_CAPABILITIES,
        ProfileFinding,
        ReasonerEvidence,
    )

    try:
        import jpype

        bundle = importlib.util.find_spec("owlready2")
        if bundle is None:
            raise ModuleNotFoundError("owlready2")
    except ModuleNotFoundError:
        return ReasonerEvidence(
            job.prepared,
            HERMIT_CAPABILITIES,
            CheckStatus.UNSUPPORTED,
            None,
            (
                Feedback(
                    FeedbackCategory.UNSUPPORTED,
                    Severity.ERROR,
                    "reasoner_dependency_missing",
                    "Install krrood-ontology[reasoner] to enable the HermiT backend",
                ),
            ),
            worker_pid=os.getpid(),
        )
    jar = Path(bundle.origin).parent / "hermit/HermiT.jar"
    if not jar.is_file():
        return ReasonerEvidence(
            job.prepared,
            HERMIT_CAPABILITIES,
            CheckStatus.UNSUPPORTED,
            None,
            (
                Feedback(
                    FeedbackCategory.UNSUPPORTED,
                    Severity.ERROR,
                    "reasoner_bundle_missing",
                    "The installed Owlready2 bundle has no HermiT engine",
                ),
            ),
            worker_pid=os.getpid(),
        )
    version = f"Owlready2 {importlib.metadata.version('owlready2')}; JPype1 {importlib.metadata.version('JPype1')}"
    digest = hashlib.sha256(jar.read_bytes()).hexdigest()
    try:
        jvm = job.jvm_path or jpype.getDefaultJVMPath()
        if not Path(jvm).is_file():
            raise jpype.JVMNotFoundException("Configured JVM library is unavailable")
    except jpype.JVMNotFoundException:
        return ReasonerEvidence(
            job.prepared,
            HERMIT_CAPABILITIES,
            CheckStatus.UNSUPPORTED,
            None,
            (
                Feedback(
                    FeedbackCategory.UNSUPPORTED,
                    Severity.ERROR,
                    "reasoner_java_missing",
                    "HermiT requires an available Java runtime/JVM shared library",
                ),
            ),
            engine_version=version,
            engine_sha256=digest,
            worker_pid=os.getpid(),
        )
    # Java is embedded in this worker. The parent owns the complete process deadline.
    jpype.startJVM(
        jvm, f"-Xmx{job.max_heap_mb}m", classpath=[str(jar)], convertStrings=True
    )
    J = jpype.JClass
    version += f"; Java {J('java.lang.System').getProperty('java.version')}"
    candidate = temporary / "candidate.rdf"
    job.prepared.graph().serialize(destination=candidate, format="xml")
    manager = J(
        "org.semanticweb.owlapi.apibinding.OWLManager"
    ).createOWLOntologyManager()
    ontology = manager.loadOntologyFromOntologyDocument(
        J("java.io.File")(str(candidate))
    )
    profile = J("org.semanticweb.owlapi.profiles.OWL2DLProfile")().checkOntology(
        ontology
    )
    axiom_count = int(ontology.getAxiomCount())
    metadata = manager.getOntologyFormat(ontology).getOntologyLoaderMetaData()
    if metadata is None:
        from krrood_ontology.contracts import ContractError

        raise ContractError(
            "reasoner_parser_metadata", "OWLAPI did not report RDF parsing coverage"
        )
    unparsed = tuple(sorted(str(t) for t in metadata.getUnparsedTriples()))
    findings = []
    for violation in profile.getViolations():
        code = str(violation.getClass().getSimpleName())
        axiom = violation.getAxiom()
        text = str(axiom) if axiom is not None else ""
        identity = (
            (
                "urn:krrood-ontology:owlapi-axiom:"
                + hashlib.sha256(text.encode()).hexdigest()
            )
            if axiom is not None
            else None
        )
        findings.append(ProfileFinding(code, text, identity))
    findings = tuple(sorted(set(findings), key=lambda f: (f.code, f.axiom_text)))
    if not profile.isInProfile() or unparsed:
        feedback = tuple(
            Feedback(
                FeedbackCategory.UNSUPPORTED,
                Severity.ERROR,
                "owl2_dl_profile",
                f"OWL 2 DL profile violation: {f.code}",
                axiom_id=f.axiom_id,
            )
            for f in findings
        )
        if unparsed:
            feedback += (
                Feedback(
                    FeedbackCategory.UNSUPPORTED,
                    Severity.ERROR,
                    "reasoner_unparsed_rdf",
                    f"OWLAPI left {len(unparsed)} RDF triples unparsed; the adapter cannot certify this candidate",
                ),
            )
        return ReasonerEvidence(
            job.prepared,
            HERMIT_CAPABILITIES,
            CheckStatus.UNSUPPORTED,
            None,
            feedback,
            version,
            digest,
            os.getpid(),
            axiom_count,
            findings,
            unparsed,
        )
    config = J("org.semanticweb.HermiT.Configuration")()
    config.ignoreUnsupportedDatatypes = False
    reasoner = None
    try:
        reasoner = J("org.semanticweb.HermiT.Reasoner")(config, ontology)
        consistent = bool(reasoner.isConsistent())
    except J("org.semanticweb.HermiT.datatypes.UnsupportedDatatypeException"):
        return ReasonerEvidence(
            job.prepared,
            HERMIT_CAPABILITIES,
            CheckStatus.UNSUPPORTED,
            None,
            (
                Feedback(
                    FeedbackCategory.UNSUPPORTED,
                    Severity.ERROR,
                    "reasoner_datatype_unsupported",
                    "HermiT cannot evaluate a datatype in this candidate; none were ignored",
                ),
            ),
            version,
            digest,
            os.getpid(),
            axiom_count,
        )
    finally:
        if reasoner is not None:
            reasoner.dispose()
        manager.removeOntology(ontology)
    if hashlib.sha256(jar.read_bytes()).hexdigest() != digest:
        from krrood_ontology.contracts import ContractError

        raise ContractError(
            "reasoner_engine_changed", "HermiT engine bytes changed during reasoning"
        )
    feedback = (
        ()
        if consistent
        else (
            Feedback(
                FeedbackCategory.REASONER,
                Severity.ERROR,
                "ontology_inconsistent",
                "HermiT reports that the candidate ontology is inconsistent; an explanation and request attribution are not available in this profile",
            ),
        )
    )
    return ReasonerEvidence(
        job.prepared,
        HERMIT_CAPABILITIES,
        CheckStatus.PASS if consistent else CheckStatus.FAIL,
        consistent,
        feedback,
        version,
        digest,
        os.getpid(),
        axiom_count,
    )


def main():
    runtime, request, response = sys.argv[1:]
    sys.path.insert(0, runtime)
    from krrood_ontology.construction import _dump_worker_record
    from krrood_ontology.contracts import (
        CheckStatus,
        ContractError,
        Feedback,
        FeedbackCategory,
        Severity,
    )
    from krrood_ontology.reasoning import (
        HERMIT_CAPABILITIES,
        OWL_RL_CAPABILITIES,
        ReasonerEvidence,
        ReasonerKind,
        _ReasonerJob,
    )

    job = pickle.loads(Path(request).read_bytes())
    if not isinstance(job, _ReasonerJob):
        raise ContractError("reasoner_protocol", "Expected a trusted reasoner job")
    capabilities = (
        HERMIT_CAPABILITIES if job.kind is ReasonerKind.HERMIT else OWL_RL_CAPABILITIES
    )
    try:
        evidence = (
            hermit_review(job, Path(response).parent)
            if job.kind is ReasonerKind.HERMIT
            else owlrl_review(job)
        )
    except ContractError as error:
        evidence = ReasonerEvidence(
            job.prepared,
            capabilities,
            CheckStatus.ERROR,
            None,
            (
                Feedback(
                    FeedbackCategory.EXECUTION, Severity.ERROR, error.code, str(error)
                ),
            ),
            worker_pid=os.getpid(),
        )
    except Exception as error:
        evidence = ReasonerEvidence(
            job.prepared,
            capabilities,
            CheckStatus.ERROR,
            None,
            (
                Feedback(
                    FeedbackCategory.EXECUTION,
                    Severity.ERROR,
                    "reasoner_failed",
                    f"Ontology reasoner failed ({type(error).__name__}); no consistency conclusion was produced",
                ),
            ),
            worker_pid=os.getpid(),
        )
    Path(response).write_bytes(_dump_worker_record(evidence))


if __name__ == "__main__":
    main()
