pouring = an(PouringAction)(
    source_container=variable(Tool),
    target_container=variable(Body),
    arm=...,
    tilt_angle=...,
)
pouring.expression.limit(3)
pouring
