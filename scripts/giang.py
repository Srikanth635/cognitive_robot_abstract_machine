import time

import trimesh
from sqlalchemy import create_engine, select
from sqlalchemy.orm.session import Session

import plotly.graph_objects as go

from semantic_world.adapters.factories import supporting_surfaces
from semantic_world.geometry import TriangleMesh, BoundingBoxCollection
from semantic_world.orm.ormatic_interface import *
from semantic_world.world import World
import coacd
import datetime


engine = create_engine('mysql+pymysql://semantic_world@localhost:3306/semantic_world')

session = Session(engine)
Base.metadata.create_all(bind=session.bind)

from semantic_world.adapters.viz_marker import VizMarkerPublisher
import rclpy

rclpy.init()

query = select(WorldMappingDAO)
world_dao = session.scalars(query).all()
world: World = world_dao[7].from_dao()

body = [b for b in world.bodies if not b.name.name.startswith("PS")][0]
[world.remove_body(b) for b in world.bodies if b != body]
mesh: trimesh.Trimesh = body.collision[0].mesh
origin = body.collision[0].origin

mesh = coacd.Mesh(mesh.vertices, mesh.faces)
parts = coacd.run_coacd(mesh, apx_mode="box", mcts_iterations=1000)

new_geometry = []
for vs, fs in parts:
    new_geometry.append(TriangleMesh(mesh=trimesh.Trimesh(vs, fs), origin=origin))
print("NUMBER OF PRIMITIVES", len(new_geometry))
body.collision = new_geometry
body.vertices = new_geometry

begin_time = datetime.datetime.now()
support = supporting_surfaces(body)


go.Figure(support.region.as_bounding_box_collection().event.plot()).show()

print("AAAAA", (datetime.datetime.now() - begin_time).total_seconds())
# a list of convex hulls.

node = rclpy.create_node("viz_marker")

p = VizMarkerPublisher(world, node)
time.sleep(100)
p._stop_publishing()
rclpy.shutdown()