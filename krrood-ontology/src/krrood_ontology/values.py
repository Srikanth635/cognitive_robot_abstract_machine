"""Shared converted-RDF value comparison policy, extracted from scoped validation."""

from decimal import Decimal as _Decimal

from krrood_ontology.constraints import ConstraintTranslationError


def value_key(value):
    # Boolean and number value spaces differ; Python False == 0 must not merge them.
    if type(value) in (int, float, _Decimal):
        number = _Decimal.from_float(value) if type(value) is float else _Decimal(value)
        if not number.is_finite():
            raise ConstraintTranslationError(
                "Nonfinite numeric values are outside this comparison profile"
            )
        return ("number", number)
    try:
        hash(value)
    except TypeError as error:
        raise ConstraintTranslationError(
            "Unhashable field values are outside this profile"
        ) from error
    return (type(value), value)
