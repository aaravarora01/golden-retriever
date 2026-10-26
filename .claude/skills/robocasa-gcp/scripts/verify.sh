#!/usr/bin/env bash
# Assert the environment is actually usable before any long run.
#
# Each check corresponds to a failure that produced valid-looking but useless
# behaviour. The last is the repo's own preflight, whose "image is not blank"
# check is the headless-render test: a policy handed blank frames looks exactly
# like a policy that cannot act.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/common.sh"
require_project
vm_exists "$RC_VM" || die "$RC_VM does not exist; run provision.sh"

step "1/4 software GL is present and selected"
on_vm "$(remote_env); \
  test -e /usr/lib/x86_64-linux-gnu/libOSMesa.so && echo '  libOSMesa: present' || { echo '  libOSMesa: MISSING'; exit 1; }; \
  test -e /usr/include/linux/input.h && echo '  linux/input.h: present' || { echo '  linux/input.h: MISSING'; exit 1; }; \
  echo \"  MUJOCO_GL=\$MUJOCO_GL PYOPENGL_PLATFORM=\$PYOPENGL_PLATFORM\"; \
  test \"\$MUJOCO_GL\" = osmesa -a \"\$PYOPENGL_PLATFORM\" = osmesa"

step "2/4 mujoco renders offscreen (not a blank frame)"
on_vm "$(remote_env); cd ~/golden-retriever && pixi run -e robocasa python -c '
import numpy as np, mujoco
m = mujoco.MjModel.from_xml_string(\"\"\"<mujoco><worldbody>
  <light pos=\"0 0 3\"/><geom type=\"plane\" size=\"2 2 .1\"/>
  <body pos=\"0 0 .3\"><geom type=\"box\" size=\".2 .2 .2\" rgba=\"1 .2 .2 1\"/></body>
  <camera name=\"c\" pos=\"1.2 -1.2 1\" xyaxes=\"0.7 0.7 0 -0.4 0.4 0.8\"/>
</worldbody></mujoco>\"\"\")
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 128, 128); r.update_scene(d, camera=\"c\")
img = r.render(); print(\"  render std %.1f\" % img.std())
assert img.std() > 1.0, \"blank frame: software GL is not working\"
' 2>&1 | grep -aE 'render std|Error|assert'"

step "3/4 the robocasa task suite imports and a kitchen builds"
on_vm "$(remote_env); cd ~/golden-retriever && pixi run -e robocasa python -c '
import warnings; warnings.filterwarnings(\"ignore\")
from examples.advanced.robocasa.study_tasks import ATOMIC_SEEN_TASKS, official_horizon
print(\"  atomic-seen tasks:\", len(ATOMIC_SEEN_TASKS))
print(\"  horizons:\", min(official_horizon(t) for t in ATOMIC_SEEN_TASKS), \"to\",
      max(official_horizon(t) for t in ATOMIC_SEEN_TASKS))
' 2>&1 | grep -aE 'atomic-seen|horizons|Error'"

step "4/4 the repo's own preflight (policy and environment agree)"
on_vm "$(remote_env); cd ~/golden-retriever && pixi run -e robocasa python -m \
  examples.advanced.robocasa.study_app --methods model \
  --task SlideDishwasherRack --preflight" | tail -4 | sed 's/^/  /'
say "verify passed; run.sh is safe to start"
