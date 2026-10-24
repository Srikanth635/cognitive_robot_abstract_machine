#!/bin/bash
# Watch or record the robot of a physics viewer scene performing an action under Isaac Sim:
#   ./run.sh view SCENE ACTION [--camera CAMERA] [--hold SECONDS] [--capture DIRECTORY]
#   ./run.sh record SCENE ACTION DIRECTORY [--fps 25] [--width 1280] [--height 720]
# for example ./run.sh view pr2_at_the_counter go_to_the_counter, or
# ./run.sh record pr2_at_the_counter pick_up_milk recordings/pick_up_milk, which writes
# one video per camera and player.html, a page that plays them.
# Runs in the Python environment with Isaac Sim installed next to the repository's
# packages, $ISAAC_CRAM_ENV (by default ~/envs/isaac-cram-env), with this checkout's
# packages first on the path. Isaac Sim's licence must already be accepted, through
# OMNI_KIT_ACCEPT_EULA.
case "$1" in
  view) SCRIPT=view_action.py ;;
  record) SCRIPT=record_action.py ;;
  *) echo "usage: $0 view|record SCENE ACTION ..." >&2; exit 2 ;;
esac
shift
REPOSITORY=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${ISAAC_CRAM_ENV:-$HOME/envs/isaac-cram-env}/bin/activate"
unset CRAMERA_SCENES CRAMERA_DATA
export PYTHONPATH=$REPOSITORY:$(ls -d "$REPOSITORY"/*/src | tr '\n' ':')$PYTHONPATH
python "$REPOSITORY/isaac_viewer/$SCRIPT" "$@"
