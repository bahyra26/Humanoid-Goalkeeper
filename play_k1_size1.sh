#!/usr/bin/env bash
# Jalankan dari environment goalkeeper. Pakai --check untuk memeriksa aset tanpa GPU.
set -euo pipefail
cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
export PYTHONPATH="$PWD/legged_gym:$PWD/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n ${CONDA_PREFIX:-} ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ ${1:-} != --check ]]; then
    set -- --gui --exptid k1_v9_s1_landfix --checkpoint 5350 --num_envs 1 "$@"
fi

# python -c membiarkan stdin tersedia untuk kontrol interaktif viewer.
exec python -c '
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

radius = 0.075
mass = 0.200
inertia = 2 * mass * radius**2 / 3  # Model kulit bola tipis seperti aset K1 asli.
ball = Path("legged_gym/resources/gymassets/urdf/ball_k1.urdf").resolve()
link = ET.parse(ball).find("link")
assert all(float(link.find(f"{s}/geometry/sphere").get("radius")) == radius for s in ("visual", "collision"))
assert float(link.find("inertial/mass").get("value")) == mass
assert all(float(link.find("inertial/inertia").get(a)) == inertia for a in ("ixx", "iyy", "izz"))
print(f"K1 Size 1 | massa {mass:g} kg | radius {radius:g} m | inersia {inertia:g} kg m²", flush=True)
if sys.argv[1:] == ["--check"]:
    sys.exit(0)

import isaacgym
from legged_gym.scripts.play import play
from legged_gym.utils import get_args

args = get_args()
args.task = "k1"
play(args)
' "$@"
