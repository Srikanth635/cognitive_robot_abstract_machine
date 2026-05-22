# Snapshot: pycram.plans.designator.Designator (full class)

- **source:** `pycram/src/pycram/plans/designator.py`, lines 19–82
- **commit:** `0528d8cf3f93170978915b17e96d311162c5af85`
- **cited by:** [[pycram.plans.Designator]], [[concept.designator]]

```python
@dataclass
class Designator:
    """
    Abstract base class for designators.
    Designators are objects that can be executed and are managed by a plan node.
    """

    plan_node: Optional[PlanNode] = field(
        kw_only=True, default=None, repr=False, init=False
    )
    """
    The plan node that manages the designator.
    """

    @property
    def plan(self) -> Plan:
        if self.plan_node is None:
            raise ContextIsUnavailable(self)
        return self.plan_node.plan

    @property
    def robot(self) -> AbstractRobot:
        if self.plan_node is None:
            raise ContextIsUnavailable(self)
        return self.plan.robot

    @property
    def world(self) -> World:
        if self.plan_node is None:
            raise ContextIsUnavailable(self)
        return self.plan_node.plan.world

    @property
    def context(self) -> Context:
        return self.plan.context

    @classmethod
    @property
    def fields(cls) -> List[Field]:
        """
        The fields of this action, returns only the fields defined in the class and not inherit fields of parents
        """
        self_fields = list(fields(cls))
        [self_fields.remove(parent_field) for parent_field in fields(Designator)]
        type_hints = cls.get_type_hints()
        for field in self_fields:
            field.type = type_hints[field.name]
        return self_fields

    @property
    def designator_parameter(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in self.fields}

    @classmethod
    def get_type_hints(cls) -> Dict[str, Any]:
        """
        Returns the type hints of the __init__ method of this designator_description description.
        """
        global_namespace = sys.modules[cls.__module__].__dict__
        return get_type_hints(cls.__init__, globalns=global_namespace)
```
