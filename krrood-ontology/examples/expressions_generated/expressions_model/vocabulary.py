"""Full IRIs of explicitly declared terms; no OWL reasoning is applied."""

SOURCE_SHA256 = "453eabf7d3ac439555012e1ff1a8675b2c8580876f0cdb26270ed26eec2a4f12"
CLASS_IRIS = (
    "https://example.org/expressions#AcmeWorker",
    "https://example.org/expressions#Company",
    "https://example.org/expressions#EmployerOfAlice",
    "https://example.org/expressions#EmployerOfManager",
    "https://example.org/expressions#EmployerOfWorker",
    "https://example.org/expressions#Manager",
    "https://example.org/expressions#NamedAlice",
    "https://example.org/expressions#Person",
    "https://example.org/expressions#PersonOrCompany",
    "https://example.org/expressions#SelectedPerson",
    "https://example.org/expressions#WorkingOrSelected",
    "https://example.org/expressions#WorkingPerson",
)
OBJECT_PROPERTY_IRIS = (
    "https://example.org/expressions#hasEmployee",
    "https://example.org/expressions#worksFor",
)
DATATYPE_PROPERTY_IRIS = ("https://example.org/expressions#name",)
ANNOTATION_PROPERTY_IRIS = ()
INDIVIDUAL_IRIS = (
    "https://example.org/expressions#Acme",
    "https://example.org/expressions#Alice",
    "https://example.org/expressions#Bob",
    "https://example.org/expressions#Charlie",
    "https://example.org/expressions#EmptyCompany",
    "https://example.org/expressions#ExplicitWorker",
)
ONTOLOGY_IMPORT_IRIS = ()
