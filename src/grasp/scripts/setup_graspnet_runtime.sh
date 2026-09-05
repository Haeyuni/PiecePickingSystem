#!/usr/bin/env bash
# Test-PC-only setup. It never moves the robot.
set -euo pipefail

image="${GRASPNET_IMAGE:-piece-picking-graspnet-baseline:0.1.0}"
assets_dir="${PIECE_PICKING_ASSETS_DIR:?Set PIECE_PICKING_ASSETS_DIR first.}"
archive="${assets_dir}/models/graspnet/checkpoint-rs.tar"
checkpoint="${assets_dir}/models/graspnet/checkpoint.tar"
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
docker_dir="${script_dir}/../docker/graspnet_baseline"

if ! command -v docker >/dev/null; then
  printf 'docker command was not found. Install Docker and NVIDIA Container Toolkit first.\n' >&2
  exit 1
fi

if [[ ! -f "$checkpoint" ]]; then
  if [[ ! -f "$archive" ]]; then
    printf 'Missing final checkpoint (%s) and source archive (%s)\n' "$checkpoint" "$archive" >&2
    exit 1
  fi
  temporary_dir="$(mktemp -d)"
  trap 'rm -rf "$temporary_dir"' EXIT
  tar -xf "$archive" -C "$temporary_dir"
  extracted_checkpoint="$(find "$temporary_dir" -type f -name checkpoint.tar -print -quit)"
  if [[ -z "$extracted_checkpoint" ]]; then
    printf 'Archive does not contain checkpoint.tar: %s\n' "$archive" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$checkpoint")"
  cp "$extracted_checkpoint" "$checkpoint"
fi

if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build --tag "$image" --file "$docker_dir/Dockerfile" "$docker_dir"
fi

docker run --rm --gpus all --entrypoint python "$image" -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
docker run --rm -v "$(realpath "$checkpoint"):/checkpoint.tar:ro" \
  "$image" --check-checkpoint /checkpoint.tar
printf 'GraspNet runtime is ready: %s\n' "$image"
