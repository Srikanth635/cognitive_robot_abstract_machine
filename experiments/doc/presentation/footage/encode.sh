#!/usr/bin/env bash
# Encodes a captured clip for the deck: H.264 in an MP4 PowerPoint plays, 1920x1080,
# 30 frames per second, trimmed to the part that matters, with a poster frame beside it.
#
#   footage/encode.sh IN OUT.mp4 [START_SECONDS] [DURATION_SECONDS] [SPEED]
#
# SPEED > 1 plays the clip faster (2 = twice as fast). The poster is OUT.jpg, taken at
# POSTER_AT seconds of the output (default 0.5). CROP=w:h:x:y crops the input first.
set -euo pipefail
input=$1
output=$2
start=${3:-0}
duration=${4:-}
speed=${5:-1}
trim=()
[ -n "$duration" ] && trim=(-t "$duration")
crop=${CROP:+crop=$CROP,}
filter="${crop}setpts=PTS/${speed},scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=white,fps=30,format=yuv420p"
ffmpeg -v error -y -ss "$start" -i "$input" "${trim[@]}" -an -vf "$filter" -c:v libx264 -preset slow -crf 20 -movflags +faststart "$output"
ffmpeg -v error -y -ss "${POSTER_AT:-0.5}" -i "$output" -frames:v 1 -q:v 2 "${output%.mp4}.jpg"
ffprobe -v error -show_entries format=duration:stream=width,height,r_frame_rate -of default=nw=1 "$output"
