#!/bin/bash
source /opt/ros/jazzy/setup.bash
cd ~/semantic_world/examples
rm -rf tmp
mkdir tmp
jupytext --to notebook *.md
mv *.ipynb tmp
cd tmp
treon --thread 1 -v