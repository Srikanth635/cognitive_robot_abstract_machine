import sys
import os

try:
    from uniworld import load_pr2_apartment_world
    from agentic_llmr.platform.world import set_active_world
except ImportError:
    print("Failed to import")
    sys.exit(1)

world, robot_view, context = load_pr2_apartment_world()

for body in world.bodies:
    if hasattr(body.name, "name"):
        name = str(body.name.name)
    else:
        name = str(body.name)

    if name == "milk.stl":
        print(f"Found milk! {type(body)}")
        try:
            pose = body.global_pose
            print(f"pose type: {type(pose)}")
            print(f"pose dir: {dir(pose)}")
            if hasattr(pose, "position"):
                print(f"pose.position type: {type(pose.position)}")
                print(f"pose.position dir: {dir(pose.position)}")
                print(f"x: {pose.position.x}")
        except Exception as e:
            import traceback
            traceback.print_exc()
        
        try:
            com = body.center_of_mass
            print(f"com type: {type(com)}")
            print(f"com dir: {dir(com)}")
        except Exception as e:
            import traceback
            traceback.print_exc()
        break
