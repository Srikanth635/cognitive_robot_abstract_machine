#!/bin/bash
# Watch the robot of a physics viewer scene perform an action in an Isaac Sim window:
#   ./run.sh SCENE ACTION [--hold SECONDS] [--capture DIRECTORY]
# for example ./run.sh pr2_at_the_counter go_to_the_counter.
# Runs in the Python environment with Isaac Sim installed next to the repository's
# packages, $ISAAC_CRAM_ENV (by default ~/envs/isaac-cram-env), with this checkout's
# packages first on the path. Isaac Sim's licence must already be accepted, through
# OMNI_KIT_ACCEPT_EULA.
REPOSITORY=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${ISAAC_CRAM_ENV:-$HOME/envs/isaac-cram-env}/bin/activate"
unset CRAMERA_SCENES CRAMERA_DATA
export PYTHONPATH=$REPOSITORY:$(ls -d "$REPOSITORY"/*/src | tr '\n' ':')$PYTHONPATH
cd "$REPOSITORY"
python "$REPOSITORY/isaac_viewer/view_action.py" "$@"
