# Semantic World

Welcome to the semantic world: The bridge between geometric worlds and machine-understandable meaning!
The semantic world is a python package that unifies the access and manipulation of scene graphs with the assertion
of meaning.

Robots and agents don’t just need coordinates—they need context. 
Semantic World fuses geometry, kinematics, and meaning so planners, learners, 
and reasoning systems can operate on the world the way humans think about it: as actionable, relatable concepts.
Build worlds that you and your agents can understand, query, and transform—then store, share, and iterate. 
From research prototypes to data-driven pipelines, 
Semantic World turns environments into knowledge.

## Assimilated Technologies
```{raw} html
<div style="display: flex; flex-direction: row; align-items: center; gap: 30px;">
  <div style="flex: 1; max-width: 70%;">
    <img src="_static/images/assimilation.png" alt="Assimilation Icon" style="width: 100%; height: auto; object-fit: contain;">
  </div>
  <div style="flex: 2; display: flex; flex-direction: column; gap: 1em;">
    <p>🌍 <b>Model full kinematic worlds, not just meshes</b>. Define bodies, regions, connections, and degrees of freedom as primary, first-class entities within a clean, composable Python API.</p>
    <p>🤔 <b>Enhance meaning with Views.</b> Transform raw geometry into actionable concepts like drawers, handles, containers, and task-relevant regions. Express relationships and intent beyond simple shapes.</p>
    <p>💡 <b>Intelligent Querying.</b> Use a high-level entity query language to precisely locate relevant elements—e.g., "the handle attached to the drawer that is currently accessible"—to enable targeted interaction.</p>
  </div>
</div>
```

🛢️️ **Reproducible Persistence and Replay.** 
Serialize annotated worlds into a SQL format, allowing for faithful reconstruction as consistent, interactive objects. 
This facilitates reproducible experiments and robust machine learning data pipelines.

🛠️ **Effortless Composition.** 
Leverage factories and dataclasses for simple authoring of complex scenes and extending semantics. 
Share domain knowledge efficiently without reliance on fragile glue code.

📈 **Scale and Consistency.** 
The integrated kinematic tree, DoF registry, 
and robust world validation ensure model consistency and integrity from initial prototype to large-scale production deployment.

🔮 **Flexible Visualization.** 
View worlds in lightweight RViz2, explore within notebooks, or integrate with richer simulation environments. 
Quickly understand both the structural and semantic layers of your models.

🔌 **Pluggable Integration.** 
Use a multitude of adapters for seamless import, no matter if its URDF, USD, MJCF, etc. 

🦾 **Reliable Kinematics.** 
Compute forward transforms and inverse (backward) kinematics cleanly across the tree, 
providing a straightforward and robust foundation for pose queries, control, and reasoning.

👯‍ **Real-Time World Synchronization.** 
Maintain a consistent state across multiple processes and robotic agents using lightweight, 
real-time world synchronization. 
Structures can be created, merged, and updated at once, 
ensuring they are accurately reflected across all connected instances.

🚀 Get started with the [](user-guide)!

🤝 Contribute with the [](developer-guide)!

## Acknowledgements
This package originates from different developments of the [AICOR Institute for Artificial Intelligence](https://ai.uni-bremen.de/). 
Four different projects developed a very similar component for different parts of cognitive modules.
This project aims to unify it under one solution that is flexible enough for all the different applications.

