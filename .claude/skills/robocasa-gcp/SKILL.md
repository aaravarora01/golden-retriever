---
name: robocasa-gcp
description: Run RoboCasa simulation tasks on a CPU-only Google Cloud VM — provision from a prepared image, shard tasks across workers, collect results to GCS, destroy. Use when a RoboCasa sweep is too slow locally, or when an API-model policy run needs parallelism. No GPU.
---

# RoboCasa on Google Cloud, CPU only

RoboCasa's simulator is MuJoCo on CPU, so a cloud run needs vCPUs and software
GL — never an accelerator. Asking for a GPU here is asking for the wrong thing:
it would sit idle while `osmesa` renders on the CPU.

## The measured facts this is built on

Taken from a probe on an `e2-standard-4` spot VM, not from documentation.

| | ms per control step |
| --- | --- |
| physics only | **8.0** |
| 1 camera @128 | 48.3 |
| 2 cameras @128 | 98.1 |
| 2 cameras @256 | **108.2** |

Software rendering is about **92% of the step cost** — roughly 45 ms per
camera — and it scales with the *number* of cameras far more than with
resolution (@128 to @256 cost only 10 ms more). Two consequences shape every
choice below:

* **Physics is cheap on a modest VM.** 8 ms/step beats a local M-series Mac's
  33 ms, which included rendering. The CPU is not the bottleneck; `osmesa` is.
* **Neither matters much for a model-backed run.** The atomic-seen split is
  11,850 control steps, so about **21 minutes of simulation**, against roughly
  **7 hours of API latency**. Optimise for parallel API calls, not for render
  speed. Prefer fewer, larger cameras over more, smaller ones.

## Three Linux-only requirements

None of these are visible from a macOS checkout, and each fails in a way that
does not name its own cause.

1. **`libosmesa6 libosmesa6-dev libglu1-mesa` from apt.** Conda's `mesalib`
   does *not* provide `libOSMesa`; only the system package does.
2. **`MUJOCO_GL=osmesa` *and* `PYOPENGL_PLATFORM=osmesa`.** Setting only the
   first fails with `AttributeError: 'NoneType' object has no attribute
   'glGetError'`, which reads like a MuJoCo bug and is a missing environment
   variable.
3. **`linux-libc-dev build-essential`.** robosuite depends on `evdev`, which
   compiles against `linux/input.h`. Without the headers the robosuite wheel
   build fails and nothing downstream installs.

`verify` asserts all three rather than trusting them.

## Shape: a prepared image, then create and destroy

Preparing the environment means a pixi solve plus about **13 GB of kitchen
assets**, which is far too slow to repeat per run. So it is done **once** into a
reusable disk image; every run boots from that image, works, and is destroyed.
That keeps create-and-destroy semantics without paying the download each time.

```
build-image   once, or when pins change   ~30-45 min
provision     per run                     ~1 min
verify        per run                     ~1 min
run           per run                     the actual work
collect       per run                     results to GCS and back
teardown      per run                     always, even on failure
```

## Commands

Scripts live in `scripts/`. Every one is idempotent and prints what it will do
before doing it.

```bash
S=.claude/skills/robocasa-gcp/scripts

# Once: prepare the environment and capture it as an image.
bash $S/build_image.sh                      # creates image robocasa-cpu-<date>

# Per run:
bash $S/provision.sh  --workers 6           # spot VM sized to the worker count
bash $S/verify.sh                           # the three Linux checks + preflight
bash $S/run.sh --atomic-seen --methods model --seeds 0
bash $S/collect.sh                          # pull checkpoints, aggregate
bash $S/teardown.sh                         # delete the VM
```

## Rules this skill follows

* **Spot instances, with `--instance-termination-action=DELETE`.** A forgotten
  VM is the realistic way to burn a research credit, so every provision carries
  a termination action and `teardown.sh` is safe to run twice.
* **Checkpoints go to GCS as they are written.** Spot VMs are preemptible and
  local disk dies with them. The sweep already checkpoints per episode; this
  mirrors each one off-box so a preemption costs one episode, not a run.
* **No key files.** The VM's attached service account supplies Application
  Default Credentials, so a Vertex-backed policy authenticates with nothing
  written to disk. The VM needs the `cloud-platform` scope and the service
  account needs `roles/aiplatform.user`.
* **The working tree is shipped, not cloned.** Work in progress is usually on an
  unpushed branch, so `provision.sh` sends a tarball of the tree (excluding
  `.pixi`, `.git`, `out`, `node_modules`). A `git clone` would silently run
  different code.
* **`verify` gates `run`.** It ends with the repo's own `study_preflight`, whose
  "image is not blank" check is exactly the headless-render test — a policy
  handed blank frames looks like a policy that cannot act.

## What this does not do

GPUs, accelerators, local VLA checkpoints, Isaac Sim or RoboLab-120. RoboLab
needs a GPU for the *simulator*, so no amount of CPU parallelism reaches it.
