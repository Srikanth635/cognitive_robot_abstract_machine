from dataclasses import dataclass


@dataclass
class JointProperty:
    """
    Represents the dynamic properties of a joint.
    tau = I*ddq + b*dq + f*sign(dq)
    where:
    - tau is the torque applied to the joint
    - I is the armature (inertia)
    - ddq is the joint acceleration
    - b is the damping coefficient
    - f is the dry friction coefficient
    - dq is the joint velocity
    """

    armature: float = 0.0
    """
    Additional inertia associated with movement of the joint that is not due to body mass. 
    This added inertia is usually due to a rotor (a.k.a armature) spinning faster than the joint itself due to a geared transmission.
    """

    dry_friction: float = 0.0
    """
    Dry friction coefficient of the joint.
    """

    damping: float = 0.0
    """
    Viscous friction coefficient of the joint.
    """
