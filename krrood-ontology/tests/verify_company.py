"""Verify inherited relations, mutually referring classes, and datatype hints."""

from dataclasses import fields
from datetime import date, datetime, time
from decimal import Decimal

from company_model import (
    CLASS_BY_IRI,
    PROPERTY_FIELD_NAMES,
    Department,
    Employee,
    Entity,
    Manager,
)
from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.symbol_graph.symbol_graph import SymbolGraph

base = "https://example.org/company#"
assert CLASS_BY_IRI[base + "Employee"] is Employee
assert issubclass(Manager, Entity)
hints = get_type_hints_of_object(Manager)
assert hints["name"] == list[str]
assert hints["class_"] == list[str]
assert hints["active"] == list[bool]
assert hints["level"] == list[int]
assert hints["salary"] == list[Decimal]
assert hints["score"] == list[float]
assert hints["ratio"] == list[float]
assert hints["birthday"] == list[date]
assert hints["started"] == list[datetime]
assert hints["shift"] == list[time]
assert hints["reports_to"] == list[Employee]
assert hints["department"] == list[Department]
assert PROPERTY_FIELD_NAMES[base + "class"] == "class_"
assert len([item for item in fields(Manager) if item.name == "name"]) == 1

manager = Manager(name=["Pat"], salary=[Decimal("100.00")], class_=["permanent"])
employee = Employee(name=["Sam"])
department = Department()
manager.reports_to.append(employee)
employee.department.append(department)
department.employee.append(employee)
assert list(manager.reports_to) == [employee]
assert list(employee.reports_to) == []
assert list(employee.department) == [department]
assert list(department.employee) == [employee]
assert any(
    relation.source.instance is manager and relation.target.instance is employee
    for relation in SymbolGraph().relations()
)
print(
    "Company model passed: inherited relations, cross references, and all supported datatypes"
)
