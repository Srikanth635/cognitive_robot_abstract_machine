"""Atomic-only profile signatures, sharing consistency checks with other domains."""
from condition_signatures import signature

def atoms(g,expression):
 checked,comparisons=signature(g,expression)
 if comparisons:raise ValueError('Quantity comparisons outside manipulation profile')
 return checked
