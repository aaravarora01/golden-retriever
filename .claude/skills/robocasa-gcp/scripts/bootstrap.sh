#!/usr/bin/env bash
# Prepare a fresh Ubuntu VM to run RoboCasa headlessly. Runs ON the VM.
#
# This is the one place the Linux-only requirements live. Each was found by a
# failure that did not name its own cause, so each is asserted here rather than
# assumed, and the script is safe to re-run.

set -euo pipefail
say() { printf '  %s\n' "$*"; }

say "== apt: software GL and the headers robosuite's evdev needs"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  libosmesa6 libosmesa6-dev libglu1-mesa \
  linux-libc-dev build-essential \
  curl git ca-certificates
# Conda's mesalib does not ship libOSMesa; only the system package does.
[ -e /usr/lib/x86_64-linux-gnu/libOSMesa.so ] \
  || { echo "libOSMesa missing after apt" >&2; exit 1; }
# robosuite's evdev compiles against this header.
[ -e /usr/include/linux/input.h ] \
  || { echo "linux/input.h missing after apt" >&2; exit 1; }
say "apt ok: libOSMesa and linux/input.h present"

say "== pixi"
if ! command -v pixi >/dev/null 2>&1; then
  curl -fsSL https://pixi.sh/install.sh | bash >/dev/null
fi
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version | sed 's/^/  /'

say "== solve the robocasa environment for linux-64"
cd "$HOME/golden-retriever"
# The repo pins robosuite and robocasa to git revisions, so this resolves the
# same code the local checkout runs.
pixi install -e robocasa

say "== robosuite and robocasa macros"
pixi run -e robocasa robosuite-setup-macros || say "robosuite macros already set"
pixi run -e robocasa robocasa-setup-macros  || say "robocasa macros already set"

say "== kitchen assets (~13 GB; the reason this is baked into an image)"
pixi run -e robocasa robocasa-download-assets

say "== bake the GL variables into every future shell"
grep -q MUJOCO_GL "$HOME/.profile" 2>/dev/null || cat >> "$HOME/.profile" <<'PROFILE'
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PATH="$HOME/.pixi/bin:$PATH"
PROFILE
say "bootstrap complete"
