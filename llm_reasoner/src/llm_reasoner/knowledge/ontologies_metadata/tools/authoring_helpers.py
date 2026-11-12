"""Small synthetic-fixture authoring helpers; never evidence acquisition or acceptance."""
from rdflib import RDF,RDFS,URIRef,Literal,XSD
from aspect_runtime import A,C,P

def clone_description(source,out,root,new_root,namespace,prefix,overrides=None):
    oldns=str(root).split('#')[0]+'#';seen=set();todo=[root]
    while todo:
        n=todo.pop()
        if n in seen:continue
        seen.add(n);todo.extend(o for o in source.objects(n) if isinstance(o,URIRef) and str(o).startswith(oldns))
    overrides={} if overrides is None else dict(overrides);overrides[root]=new_root
    mapping={n:overrides.get(n,namespace[prefix+str(n).split('#')[-1]]) for n in seen}
    for n in seen:
        for p,o in source.predicate_objects(n):out.add((mapping[n],p,mapping.get(o,o)))
    return mapping

class Fixture:
    def __init__(self,graph,namespace):self.g=graph;self.ns=namespace
    def node(self,key,typ):
        n=self.ns[key];self.g.add((n,RDF.type,typ));return n
    def add(self,s,p,o):self.g.add((s,p,o));return s
    def parameter(self,key,unit):
        n=self.node(key,A.ParameterSpecification);self.add(n,A.parameterKey,Literal(key,datatype=XSD.string));self.add(n,A.unitIRI,Literal('http://qudt.org/vocab/unit/'+unit,datatype=XSD.anyURI));return n
    def atom(self,key,relation,args,scope):
        local=str(relation).split('#')[-1];rel_ns=str(relation).split('#')[0]+'#';pat=self.node(key+'Pattern',P.RelationPattern);self.add(pat,P.usesRelation,relation)
        for i,arg in enumerate(args,1):
            binding=self.node(key+'Binding'+str(i),P.ArgumentBinding);self.add(pat,P.hasArgumentBinding,binding);self.add(binding,P.bindsArgument,URIRef(rel_ns+local+'Argument'+str(i)));self.add(binding,P.bindsEntity,arg)
        e=self.node(key,C.AtomicCondition);self.add(e,C.hasPattern,pat);self.add(e,C.inScope,scope);return e
    def combine(self,key,typ,parts):
        if len(parts)==1:return parts[0]
        e=self.node(key,typ)
        for p in parts:self.add(e,C.hasOperand,p)
        return e
    def condition(self,key,typ,expression):return self.add(self.node(key,typ),C.hasExpression,expression)
    def snapshot(self,root,key,version='synthetic-1'):
        seen=set();todo=[root]
        while todo:
            n=todo.pop()
            if n in seen:continue
            seen.add(n);todo.extend(o for o in self.g.objects(n) if isinstance(o,URIRef) and str(o).startswith(str(self.ns)))
        triples={(s,p,o) for s in seen for p,o in self.g.predicate_objects(s) if p not in [RDF.type,RDFS.label,RDFS.comment]}
        record=self.node(key+'Record',A.KnowledgeRecord);ann=self.node(key+'Annotation',A.AnnotationRecord);evidence=self.node(key+'Evidence',A.EvidenceRecord);bundle=self.node(key+'ClaimBundle',A.ActionComponent)
        for p,o in [(A.expressesDescription,root),(A.recordVersion,Literal(version,datatype=XSD.string)),(A.hasAnnotation,ann)]:self.add(record,p,o)
        self.add(ann,A.annotatesDescription,root);self.add(ann,A.producerIdentifier,Literal('synthetic-schema-authoring',datatype=XSD.string));self.add(evidence,A.evidenceKind,A.CuratedExample);self.add(evidence,A.evidenceText,Literal('Synthetic Proposed engineering example, not a measured outcome, accepted observation or executable action.',datatype=XSD.string))
        for i,(s,p,o) in enumerate(sorted(triples,key=lambda t:tuple(map(str,t)))):
            claim=self.node(key+'Claim'+str(i),A.KnowledgeClaim);st=self.node(key+'Statement'+str(i),A.Statement)
            for x,pred,value in [(record,A.hasClaim,claim),(ann,A.contributesClaim,claim),(claim,A.aboutComponent,bundle),(claim,A.hasStatus,A.Proposed),(claim,A.hasEvidence,evidence),(claim,A.hasStatement,st),(st,A.statementSubject,s),(st,A.statementPredicateIRI,Literal(str(p),datatype=XSD.anyURI)),(st,A.statementLiteral if isinstance(o,Literal) else A.statementObject,o)]:self.add(x,pred,value)
        return record
