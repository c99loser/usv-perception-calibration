#!/usr/bin/env bash
# Training-only. Run from the repository root in a Gazebo Harmonic environment.
set -euo pipefail

log_file="${TMPDIR:-/tmp}/fisheye_segmentation_preflight.log"
gz sim -s -r --headless-rendering tools/simulation/fisheye_segmentation_preflight.sdf >"$log_file" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT

sleep 5
echo "Topics:"
gz topic -l | grep '^/preflight/'
echo "RGB metadata:"
timeout 5s gz topic -e -n 1 -t /preflight/rgb | grep -E '^(width|height|pixel_format_type):' || true
echo "Semantic label-map metadata:"
timeout 5s gz topic -e -n 1 -t /preflight/semantic/labels_map | grep -E '^(width|height|pixel_format_type):' || true
