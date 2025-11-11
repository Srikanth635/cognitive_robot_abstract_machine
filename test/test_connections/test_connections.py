from numpy.testing import assert_allclose
from semantic_digital_twin.world_description.connection_properties import JointProperty


class TestJointProperty:
    def test_default_values(self):
        joint_prop = JointProperty()
        assert_allclose(joint_prop.armature, 0.0)
        assert_allclose(joint_prop.dry_friction, 0.0)
        assert_allclose(joint_prop.damping, 0.0)

    def test_custom_values(self):
        armature = 1.5
        dry_friction = 0.2
        damping = 0.05
        joint_prop = JointProperty(
            armature=armature, dry_friction=dry_friction, damping=damping
        )
        assert_allclose(joint_prop.armature, armature)
        assert_allclose(joint_prop.dry_friction, dry_friction)
        assert_allclose(joint_prop.damping, damping)

        joint_prop_dict = joint_prop.__dict__
        assert_allclose(joint_prop_dict["armature"], armature)
        assert_allclose(joint_prop_dict["dry_friction"], dry_friction)
        assert_allclose(joint_prop_dict["damping"], damping)
