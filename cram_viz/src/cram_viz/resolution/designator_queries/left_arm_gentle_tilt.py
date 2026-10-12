pouring = the(PouringAction)(
    source_container=variable(Tool),
    target_container=variable(Body),
    arm=Arms.LEFT,
    tilt_angle=...,
)
pouring.where(pouring.variable.tilt_angle >= 1.0, pouring.variable.tilt_angle <= 1.5)
