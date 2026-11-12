# Recommended next domain: mobility and navigation

Status: the agreed initial scope was implemented in 0.15.0. Read **MOBILITY_NAVIGATION.md** for the current representation, checks and remaining limitations. This file retains the original planning rationale.

Robot motion supplies configuration/pose/path descriptions. Object manipulation supplies object interaction and support/holding outcomes. Mobility adds the semantics of relocating a robot through an environment. It can support later transport, search and move-and-open recipes without depending on those recipes.

## Initial action families

| Family | Intended additional knowledge |
|---|---|
| Navigation | Robot/base target pose, arrival tolerances, reference frame and applicable region |
| Route/path navigation | Route segments, traversability requirements, route compliance and endpoint arrival |
| Facing | Orient the robot/base toward a target while constraining translation |
| Inter-level navigation | Source/destination levels, transition facilities, entry/exit and arrival conditions |
| Elevator navigation profile | Cabin/landing relation, door state, serving level, waiting, entry, ride and exit conditions |

These families provide initial coverage for the previously inspected NavigateAction, PathPlanningNavigateAction, FaceAtAction and ElevatorNavigation implementations. Such Python class names belong in a future adapter mapping, not in the domain ontology vocabulary.

## Boundary decisions

- Import robot motion and selected generic aspects; keep object manipulation optional. Carrying is later composed from mobility and holding requirements.
- Describe regions, routes, passages, levels and transition facilities as specifications. Supply actual geometry, robots, maps and states through injected context.
- Preserve frames, units, map/version references, observation timestamps and uncertainty. Missing route or door evidence stays unknown.
- Distinguish arriving at a pose from following an admissible route. Distinguish geometric arrival from being on the requested level.
- Represent locomotion capabilities and footprint/clearance requirements without fixing a differential-drive or holonomic robot.
- Keep viewing/detection in the future perception/attention domain, while reusing visibility or observation requirements when needed.
- Treat elevator interaction as a scoped navigation profile. General door articulation remains in the future mechanism-interaction domain.

## Competency questions for implementation

1. Which robot entity description moves, toward which target and frame?
2. Which route segments and regions must be traversed, in what order?
3. Is the route conditionally traversable for the injected robot footprint and capabilities?
4. Does completion require route compliance as well as endpoint arrival?
5. Does a facing task preserve position and achieve the requested orientation?
6. Is the robot on the intended destination level, rather than merely at similar XY coordinates?
7. Which door/cabin/landing conditions permit entry and exit?
8. Which observations are missing, stale or contradictory?
9. Which conditions must be maintained when mobility is composed with carrying?

This scope creates reusable semantic structures and hypothetical profiles. Route planning, localization, collision checking, elevator control and execution evidence remain separate adapters/models requiring validation.
