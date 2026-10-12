# Draft: object qualities and the situation a hypothesis is applied to

Status: draft for review. Nothing here is loaded by the package.

## Problem

A generated hypothesis sees only the instruction text. Nothing ties its parameters to the
objects involved: two runs of "pour water from the jug into the glass" disagree on tilt
(60-110 deg vs 0-110 deg) and lift-off force (4-11 N vs 2-5 N) as much as the jug differs
from a carton. DUL supplies the pattern for describing properties (Quality, Region,
Situation) but no properties; CRAM-AICOR names `MaterialProperty`, `Affordance` and
dispositions but leaves them empty.

## Design

The scientific contract keeps the microworld at the description level: "a separate
occurrence-level model binds the generic description to the world". This draft is that
model, built on DUL's Description and Situation pattern.

```
hypothesis (dul:Description)            situation (dul:Situation)
  struct ──────────────── dul:satisfies ──── situation
  role hyp:src ◄── aicor:bindsSlot ── binding ── aicor:bindsFiller ──► jug
                                                                      │ dul:hasQuality
                                                                      ▼
                                                    capacity ── dul:hasRegion ──► 1.5 litre
```

- Qualities are DUL qualities of the world objects; their values reuse core's
  `MagnitudeRange` (an exact value has equal bounds) and `CategoricalValue`.
- Property and state stay apart, as the contract requires: capacity, opening width,
  rigidity, viscosity are stable properties; contained quantity is a state.
- Substance qualities are `aicor:MaterialProperty`, filling CRAM-AICOR's empty slot.
- The material category reuses pouring's `Liquid`/`Granular`/`Cohesive`, so the
  substance and the structure speak of the same regimes.
- Estimates in the hypothesis stay separate from the actual qualities. A prediction that
  disagrees with reality is a prediction error, so rules comparing them only warn.

## Files

| File | Contents |
|---|---|
| `situation.ttl` | situation, bindings, quality classes, and classes the reasoner infers from qualities (`LiquidSubstance`, `GranularSubstance`, `CohesiveSubstance`, `DeformableContainer`) |
| `core-additions.ttl` | proposed core changes: a density dimension and unit, and a conversion factor per unit |
| `situation-shapes.ttl` | rules relating a hypothesis to the actual qualities |
| `example-jug-glass.ttl` | illustrative situation for the stored jug hypothesis (values are not measurements) |

## Rules

| Rule | Severity | States |
|---|---|---|
| `MaterialRegimeOfSubstance` | violation | the structure's material regime is the poured substance's material category |
| `TransferWithinSourceContents` | violation | setting out to transfer more than the source holds ends with the source exhausted |
| `TransferWithinTargetSpace` | violation | never set out to transfer more than the target has room for |
| `HeldLoadPredicted` | warning | the expected mass covers the source's actual mass with contents; the load force estimate reaches its weight |

Each rule is silent when the situation does not state the quality it needs.

## What was checked

Against the stored jug hypothesis (record `7ce5be81758b4f15a8dc50ed3d622478`):

- With the example situation, only `HeldLoadPredicted` warns: the hypothesis predicts a
  load of 2-5 N for a 0.4 kg jug holding a litre of water, which weighs about 13.7 N.
- A glass already holding 200 ml: `TransferWithinTargetSpace` rejects the 150-250 ml pour.
- A jug holding 100 ml: `TransferWithinSourceContents` rejects it, since the structure
  ends with the target quantity attained.
- A granular substance: `MaterialRegimeOfSubstance` rejects the liquid regime.
- HermiT finds the ontology, the hypothesis and the situation consistent, and infers the
  water to be a `LiquidSubstance` from its quality alone.
- HermiT finds a jug capacity given in kilograms inconsistent: unit and dimension are
  checked open-world without a dedicated rule.

## Open questions

1. **Module placement.** One occurrence-level module, as drafted, or the generic object
   qualities in core and the substance qualities in pouring?
2. **Core changes.** `core-additions.ttl` extends core's closed lists of dimensions and
   units. Acceptable?
3. **Dependencies that need sources.** These are plausible but have no claim in the
   ontology to cite, so they are not encoded:
   - higher viscosity calls for a slower tilt rate and a later flow onset;
   - a deformable source container caps grip force, since squeezing it pushes the
     contents out;
   - a narrow target opening caps the flow rate;
   - a granular substance starts flowing near its angle of repose.
4. **Flow onset from fill level.** For an upright cylindrical source at least half full,
   the free surface reaches the rim when `tan(tilt) = 2 (interior height - fill height) /
   opening width`. SPARQL has no trigonometry, so this needs a computed check in Python
   rather than a shape. Worth it?
5. **Amounts in the instruction.** "Half a glass" or "200 ml" should become a fixed value
   of the transferred quantity rather than something the model guesses.

## Not yet done

- Giving the situation to the generation, so the model states its parameters knowing the
  objects, and merging it into the graph the acceptance rules read.
- Filling situations from the scene description now and from the semantic digital twin
  later.
