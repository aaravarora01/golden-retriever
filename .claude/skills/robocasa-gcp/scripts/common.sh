#!/usr/bin/env bash
# Shared settings and helpers. Source this; do not run it.
#
# Every value here can be overridden from the environment, so a caller can
# retarget zone, machine type or image without editing a script.

set -euo pipefail

: "${RC_ZONE:=us-central1-a}"
: "${RC_VM:=robocasa-cpu}"
: "${RC_IMAGE_FAMILY:=robocasa-cpu}"
# Physics is 8 ms/step and osmesa is ~45 ms per camera per step, so cores buy
# parallel episodes. 16 vCPU comfortably runs the 6 workers that took a local
# sweep from 7.4 h to 75 min.
: "${RC_MACHINE:=n2-standard-16}"
: "${RC_DISK_GB:=120}"
: "${RC_BUILD_VM:=robocasa-cpu-build}"
: "${RC_BUILD_MACHINE:=n2-standard-8}"

RC_PROJECT="${RC_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
: "${RC_BUCKET:=gs://${RC_PROJECT}-robocasa-runs}"

say()  { printf '  %s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
die()  { printf '  ERROR: %s\n' "$*" >&2; exit 1; }

require_project() {
  [ -n "$RC_PROJECT" ] || die "no GCP project; run 'gcloud config set project PROJECT'"
}

vm_exists() {
  gcloud compute instances describe "$1" --zone="$RC_ZONE" \
    --format="value(name)" >/dev/null 2>&1
}

# Run a command on the VM. Keeps the ssh invocation in one place so the quiet
# flags and the zone cannot drift between scripts.
on_vm() {
  gcloud compute ssh "$RC_VM" --zone="$RC_ZONE" --quiet --command="$1"
}

ensure_bucket() {
  if ! gcloud storage buckets describe "$RC_BUCKET" >/dev/null 2>&1; then
    say "creating results bucket $RC_BUCKET"
    gcloud storage buckets create "$RC_BUCKET" --location=US --quiet
  fi
}

# The environment every remote command needs. osmesa alone is not enough:
# PyOpenGL needs its own platform variable or it fails with a NoneType error
# that names neither.
remote_env() {
  printf '%s' 'export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE'
}
