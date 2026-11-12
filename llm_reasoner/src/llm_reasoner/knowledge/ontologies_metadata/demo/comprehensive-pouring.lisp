;; Readable, abbreviated illustration of comprehensive-pouring.ttl/json.
;; These display slots are not a new parser or projection schema.
;; Every number and assumed outcome is fictional Proposed knowledge.

(an action-description
  (id pouring-demo)
  (type Pouring)
  (knowledge-status Proposed)
  (evidence CuratedExample)
  (context fictional-pouring-context)

  (entities
    (:tag bottle
      (an entity-specification
        (kind bottle) (material glass) (dry-mass 0.42 kg)
        (initial-contents (:ref water)) (initial-volume 600 mL)))
    (:tag water
      (a material-specification
        (kind water) (state liquid)
        (density 998 kg/m3) (viscosity 0.0011 Pa.s)))
    (:tag glass
      (an entity-specification (kind drinking-glass) (capacity 400 mL)))
    (:tag mouth (a working-part (part-of (:ref bottle))))
    (:tag delivered-water (a portion-of (:ref water))))

  (material-transfer
    (source (:ref bottle)) (substance (:ref water))
    (destination (:ref glass)) (portion (:ref delivered-water))
    (mode partial) (quantity-basis volume)
    (target-volume 250 mL) (spillage-tolerance 5 mL))

  (tool-interaction
    (type UsingTool) (context fictional-tool-use-context)
    (operator abstract-end-effector)
    (tool (:ref bottle)) (working-part (:ref mouth))
    (target (:ref glass)) (intended-effect flow-into-glass)
    (contact-policy unspecified)
    (hypothetical-force-limit 8 N))

  (motion-specification
    (reference-frame unbound-demo-frame)
    (pour-position (x 0.62 m) (y -0.18 m) (z 0.94 m))
    (tilt 1.15 rad) (maximum-angular-speed 0.35 rad/s)
    (maximum-acceleration 0.4 m/s2))

  (success-criteria
    (and (>= delivered-volume 250 mL)
         (<= spilled-volume 5 mL)
         (contained-in (:ref delivered-water) (:ref glass))))

  (knowledge-aspects
    (conditions (reference pourPrecondition))
    (phases prepare transfer verify complete-or-abort)
    (monitoring (reference pourControlPoint)
                (on-unknown request-evidence))
    (causality (reference pourCausalModel)
               (status Proposed) (causal-probability unknown))
    (physics
      (reference pourPhysicalModel)
      (accounting V_after = V_before - V_delivered - V_spilled)
      (assumptions incompressible all-losses-accounted)
      (physical-validity unassessed))
    (control (type FeedbackControl) (sampling-period 0.02 s)
             (state-evolution-equation unknown)
             (output-equation unknown)
             (calibrated-actuator-model unknown))
    (objectives (priority-1 minimize-spillage)
                (priority-2 minimize-completion-time))
    (capabilities (reference pourCapabilityProfile)
                  (providers-matched false))
    (strategy (reference pourExecutionStrategy))
    (diagnosis (reference spillDiagnosticAssociation)
               (status proposed-candidate-explanation))
    (recovery (reference pourRecoveryStrategy) (retry-budget 2))
    (frame-perspective (schema PouringFrame)
                       (source (:ref bottle))
                       (substance (:ref water))
                       (destination (:ref glass)))
    (force-dynamics (reference pourForcePattern)
                    (relative-strength unknown))
    (image-schemas containment source-path-goal))

  (hypothesized-outcome
    (not-an-observation true)
    (delivered-volume 252 mL) (spilled-volume 1 mL)
    (source-remaining 347 mL)
    (assumed-correct-destination true)
    (accounting 600 = 347 + 252 + 1 mL))

  (runtime-status
    (world-bindings none)
    (controller-synthesized false)
    (execution-readiness unassessed)
    (coraplex-executable-projection none)))
