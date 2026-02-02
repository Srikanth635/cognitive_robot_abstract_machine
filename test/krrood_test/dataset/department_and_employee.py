from dataclasses import dataclass

from krrood.entity_query_language.predicate import Symbol


@dataclass(unsafe_hash=True)
class Department(Symbol):
    name: str


@dataclass
class Employee(Symbol):
    name: str
    department: Department
    salary: float
