#!/usr/bin/env python3

import argparse
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_RGBA = "0.7 0.7 0.7 1"

POSITION_GAINS = {
    "torso_lift_joint": 1000000.0,
    "head_pan_joint": 50.0,
    "head_tilt_joint": 100.0,
    "r_shoulder_pan_joint": 200.0,
    "r_shoulder_lift_joint": 200.0,
    "r_upper_arm_roll_joint": 100.0,
    "r_elbow_flex_joint": 100.0,
    "r_forearm_roll_joint": 50.0,
    "r_wrist_flex_joint": 50.0,
    "r_wrist_roll_joint": 50.0,
    "l_shoulder_pan_joint": 200.0,
    "l_shoulder_lift_joint": 200.0,
    "l_upper_arm_roll_joint": 100.0,
    "l_elbow_flex_joint": 100.0,
    "l_forearm_roll_joint": 50.0,
    "l_wrist_flex_joint": 50.0,
    "l_wrist_roll_joint": 50.0,
}

GRIPPER_ACTUATORS = {
    "r_gripper_l_finger_joint": "r_gripper_motor",
    "l_gripper_l_finger_joint": "l_gripper_motor",
}

BASE_ACTUATORS = {
    "base_x_joint": ("base_x_motor", 10000.0, [-100.0, 100.0], 1000000.0),
    "base_y_joint": ("base_y_motor", 10000.0, [-100.0, 100.0], 1000000.0),
    "base_yaw_joint": ("base_yaw_motor", 5000.0, [-8 * math.pi, 8 * math.pi], 1000000.0),
}


def parse_floats(value: str, expected_length: int) -> list[float]:
    values = [float(part) for part in value.split()]
    if len(values) != expected_length:
        raise ValueError(
            f"Expected {expected_length} values, got {len(values)} in {value!r}"
        )
    return values


def format_floats(values: list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def material_colors(robot: ET.Element) -> dict[str, str]:
    colors = {}
    for material in robot.findall("material"):
        color = material.find("color")
        if color is not None and color.get("rgba"):
            colors[material.get("name", "")] = color.get("rgba", DEFAULT_RGBA)
    return colors


def link_color(link: ET.Element, colors: dict[str, str]) -> str:
    material = link.find("./visual/material")
    if material is None:
        return DEFAULT_RGBA

    inline_color = material.find("color")
    if inline_color is not None and inline_color.get("rgba"):
        return inline_color.get("rgba", DEFAULT_RGBA)
    return colors.get(material.get("name", ""), DEFAULT_RGBA)


def add_origin_attributes(geom: ET.Element, source: ET.Element) -> None:
    origin = source.find("origin")
    if origin is None:
        return

    xyz = origin.get("xyz")
    if xyz:
        geom.set("pos", format_floats(parse_floats(xyz, 3)))

    rpy = origin.get("rpy")
    if rpy:
        geom.set("euler", format_floats(parse_floats(rpy, 3)))


def mesh_source_path(urdf_path: Path, filename: str) -> Path:
    relative_path = filename.removeprefix("file://./")
    if relative_path.startswith("file://"):
        return Path(relative_path.removeprefix("file://"))
    return (urdf_path.parent / relative_path).resolve()


def add_collision_geometry(
    link: ET.Element,
    body: ET.Element,
    asset: ET.Element,
    urdf_path: Path,
    mesh_output_dir: Path,
    mesh_asset_paths: dict[Path, str],
    rgba: str,
) -> int:
    added = 0
    for index, collision in enumerate(link.findall("collision")):
        geometry = collision.find("geometry")
        if geometry is None or len(geometry) != 1:
            continue

        shape = geometry[0]
        attributes = {
            "name": f"{link.get('name')}_geom_{index}",
            "class": "pr2_visual",
            "rgba": rgba,
        }

        if shape.tag == "mesh":
            filename = shape.get("filename")
            if not filename:
                continue
            source_path = mesh_source_path(urdf_path, filename)
            if source_path.suffix.lower() != ".stl":
                continue
            if not source_path.is_file():
                raise FileNotFoundError(source_path)

            mesh_name = mesh_asset_paths.get(source_path)
            if mesh_name is None:
                mesh_name = f"pr2_mesh_{len(mesh_asset_paths)}"
                mesh_asset_paths[source_path] = mesh_name
                destination = mesh_output_dir / source_path.name
                shutil.copy2(source_path, destination)

                mesh_attributes = {
                    "name": mesh_name,
                    "file": f"{mesh_output_dir.name}/{destination.name}",
                }
                scale = shape.get("scale")
                if scale:
                    mesh_attributes["scale"] = format_floats(
                        parse_floats(scale, 3)
                    )
                ET.SubElement(asset, "mesh", mesh_attributes)

            attributes.update({"type": "mesh", "mesh": mesh_name})
        elif shape.tag == "box":
            full_size = parse_floats(shape.get("size", ""), 3)
            attributes.update(
                {
                    "type": "box",
                    "size": format_floats([value / 2 for value in full_size]),
                }
            )
        elif shape.tag == "cylinder":
            attributes.update(
                {
                    "type": "cylinder",
                    "size": format_floats(
                        [
                            float(shape.get("radius", "0")),
                            float(shape.get("length", "0")) / 2,
                        ]
                    ),
                }
            )
        elif shape.tag == "sphere":
            attributes.update(
                {
                    "type": "sphere",
                    "size": format_floats([float(shape.get("radius", "0"))]),
                }
            )
        else:
            continue

        if "gripper" in link.get("name", "") and "finger_tip" in link.get("name", ""):
            attributes.update(
                {
                    "contype": "1",
                    "conaffinity": "1",
                    "friction": "2 0.05 0.01",
                    "condim": "4",
                }
            )

        geom = ET.SubElement(body, "geom", attributes)
        add_origin_attributes(geom, collision)
        added += 1

    return added


def add_position_actuators(
    mjcf_root: ET.Element, control_root: ET.Element
) -> int:
    control_joints = {
        joint.get("name"): joint for joint in control_root.findall("joint")
    }
    mjcf_joints = {
        joint.get("name"): joint
        for joint in mjcf_root.findall(".//joint")
        if joint.get("name")
    }
    motor_names = {}
    for transmission in control_root.findall("transmission"):
        if transmission.get("type") != "pr2_mechanism_model/SimpleTransmission":
            continue
        joint = transmission.find("joint")
        actuator = transmission.find("actuator")
        if joint is not None and actuator is not None:
            motor_names[joint.get("name")] = actuator.get("name")

    actuator_section = mjcf_root.find("actuator")
    if actuator_section is None:
        actuator_section = ET.Element("actuator")
        equality = mjcf_root.find("equality")
        if equality is None:
            mjcf_root.append(actuator_section)
        else:
            mjcf_root.insert(list(mjcf_root).index(equality), actuator_section)
    else:
        actuator_section.clear()

    for joint_name, kp in POSITION_GAINS.items():
        control_joint = control_joints[joint_name]
        mjcf_joint = mjcf_joints[joint_name]
        limit = control_joint.find("limit")
        dynamics = control_joint.find("dynamics")

        if dynamics is not None and dynamics.get("damping"):
            mjcf_joint.set("damping", dynamics.get("damping", "0"))

        if control_joint.get("type") == "continuous":
            control_range = [-8 * math.pi, 8 * math.pi]
        else:
            control_range = [
                float(limit.get("lower", "0")),
                float(limit.get("upper", "0")),
            ]

        attributes = {
            "name": motor_names.get(joint_name, f"{joint_name}_motor"),
            "joint": joint_name,
            "kp": format_floats([kp]),
            "ctrlrange": format_floats(control_range),
            "ctrllimited": "true",
        }
        if limit is not None and limit.get("effort"):
            effort = abs(float(limit.get("effort", "0")))
            attributes.update(
                {
                    "forcerange": format_floats([-effort, effort]),
                    "forcelimited": "true",
                }
            )
        ET.SubElement(actuator_section, "position", attributes)

    for joint_name, (actuator_name, kp, control_range, force) in BASE_ACTUATORS.items():
        ET.SubElement(
            actuator_section,
            "position",
            {
                "name": actuator_name,
                "joint": joint_name,
                "kp": format_floats([kp]),
                "ctrlrange": format_floats(control_range),
                "ctrllimited": "true",
                "forcerange": format_floats([-force, force]),
                "forcelimited": "true",
            },
        )

    for joint_name, actuator_name in GRIPPER_ACTUATORS.items():
        ET.SubElement(
            actuator_section,
            "position",
            {
                "name": actuator_name,
                "joint": joint_name,
                "kp": "100",
                "ctrlrange": "0 0.548",
                "ctrllimited": "true",
                "forcerange": "-1000 1000",
                "forcelimited": "true",
            },
        )

    return len(POSITION_GAINS) + len(BASE_ACTUATORS) + len(GRIPPER_ACTUATORS)


def generate(
    urdf_path: Path,
    control_urdf_path: Path,
    template_path: Path,
    output_path: Path,
    mesh_output_dir: Path,
) -> tuple[int, int]:
    urdf_root = ET.parse(urdf_path).getroot()
    control_root = ET.parse(control_urdf_path).getroot()
    mjcf_tree = ET.parse(template_path)
    mjcf_root = mjcf_tree.getroot()

    links = {link.get("name"): link for link in urdf_root.findall("link")}
    bodies = {
        body.get("name"): body
        for body in mjcf_root.findall(".//body")
        if body.get("name")
    }
    if links.keys() != bodies.keys():
        missing_links = sorted(bodies.keys() - links.keys())
        missing_bodies = sorted(links.keys() - bodies.keys())
        raise ValueError(
            f"URDF/MJCF names differ. Missing links: {missing_links}; "
            f"missing bodies: {missing_bodies}"
        )

    asset = mjcf_root.find("asset")
    if asset is None:
        compiler = mjcf_root.find("compiler")
        insert_at = list(mjcf_root).index(compiler) + 1 if compiler is not None else 0
        asset = ET.Element("asset")
        mjcf_root.insert(insert_at, asset)
    else:
        asset.clear()
    option = mjcf_root.find("option")
    if option is None:
        default = mjcf_root.find("default")
        insert_at = list(mjcf_root).index(default) if default is not None else 0
        option = ET.Element("option")
        mjcf_root.insert(insert_at, option)
    option.set("gravity", "0 0 0")

    for freejoint in mjcf_root.findall(".//freejoint"):
        parent = next(
            body for body in mjcf_root.findall(".//body") if freejoint in list(body)
        )
        parent.remove(freejoint)

    base = bodies["base_footprint"]
    for joint_name in BASE_ACTUATORS:
        existing = base.find(f"joint[@name='{joint_name}']")
        if existing is not None:
            base.remove(existing)

    worldbody = mjcf_root.find("worldbody")
    if worldbody is None or base not in list(worldbody):
        raise ValueError("base_footprint must be a top-level body")
    worldbody.remove(base)

    planar_x = ET.Element("body", {"name": "pr2_planar_x"})
    ET.SubElement(
        planar_x,
        "joint",
        {
            "name": "base_x_joint",
            "type": "slide",
            "axis": "1 0 0",
            "damping": "1000",
        },
    )
    planar_y = ET.SubElement(planar_x, "body", {"name": "pr2_planar_y"})
    ET.SubElement(
        planar_y,
        "joint",
        {
            "name": "base_y_joint",
            "type": "slide",
            "axis": "0 1 0",
            "damping": "1000",
        },
    )
    base.insert(
        1,
        ET.Element(
            "joint",
            {
                "name": "base_yaw_joint",
                "type": "hinge",
                "axis": "0 0 1",
                "damping": "1000",
            },
        ),
    )
    planar_y.append(base)
    worldbody.append(planar_x)

    actuator_count = add_position_actuators(mjcf_root, control_root)

    mesh_output_dir.mkdir(parents=True, exist_ok=True)
    mesh_asset_paths: dict[Path, str] = {}
    colors = material_colors(urdf_root)
    geometry_count = 0

    for body_name, body in bodies.items():
        link = links[body_name]
        geometry_count += add_collision_geometry(
            link=link,
            body=body,
            asset=asset,
            urdf_path=urdf_path,
            mesh_output_dir=mesh_output_dir,
            mesh_asset_paths=mesh_asset_paths,
            rgba=link_color(link, colors),
        )

    ET.indent(mjcf_tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mjcf_tree.write(output_path, encoding="unicode", xml_declaration=True)
    return geometry_count, len(mesh_asset_paths), actuator_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge PR2 URDF collision geometry into its kinematic MJCF."
    )
    parser.add_argument("urdf", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument(
        "--control-urdf",
        type=Path,
        help="URDF supplying joint limits, damping, transmissions, and effort limits.",
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mesh-output-dir",
        type=Path,
        help="Defaults to a pr2_meshes directory next to the output MJCF.",
    )
    args = parser.parse_args()

    output_path = args.output.resolve()
    mesh_output_dir = (
        args.mesh_output_dir.resolve()
        if args.mesh_output_dir
        else output_path.parent / "pr2_meshes"
    )
    if mesh_output_dir.parent != output_path.parent:
        raise ValueError("The mesh output directory must be next to the output MJCF.")

    control_urdf_path = (args.control_urdf or args.urdf).resolve()
    geometry_count, mesh_count, actuator_count = generate(
        urdf_path=args.urdf.resolve(),
        control_urdf_path=control_urdf_path,
        template_path=args.template.resolve(),
        output_path=output_path,
        mesh_output_dir=mesh_output_dir,
    )
    print(
        f"Wrote {output_path} with {geometry_count} geoms "
        f"{mesh_count} mesh assets, and {actuator_count} position actuators."
    )


if __name__ == "__main__":
    main()
