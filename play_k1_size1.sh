#!/usr/bin/env bash
# Jalankan dari environment goalkeeper. Pakai --check untuk memeriksa aset tanpa GPU.
set -euo pipefail
cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
export PYTHONPATH="$PWD/legged_gym:$PWD/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n ${CONDA_PREFIX:-} ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [[ ${1:-} != --check ]]; then
    has_walk=0
    [[ " $* " == *" --walk_ckpt "* ]] && has_walk=1
    if [[ " $* " == *" --legacy "* ]]; then
        extra=()
        if [[ $has_walk -eq 0 && -f legged_gym/logs/k1_walk/k1_walk_v4/model_2400.pt ]]; then
            extra=(--walk_ckpt legged_gym/logs/k1_walk/k1_walk_v4/model_2400.pt)
        fi
        set -- --gui --exptid k1_v9_s1_landfix --checkpoint 5350 --num_envs 1 "${extra[@]}" "$@"
    elif [[ -f legged_gym/logs/k1/k1_v11_balanced_p2/model_6000.pt ]]; then
        extra=()
        if [[ $has_walk -eq 0 && -f legged_gym/logs/k1_walk/k1_walk_v5/model_2700.pt ]]; then
            extra=(--walk_ckpt legged_gym/logs/k1_walk/k1_walk_v5/model_2700.pt)
        fi
        set -- --gui --exptid k1_v11_balanced_p2 --checkpoint 6000 --num_envs 1 "${extra[@]}" "$@"
    else
        set -- --gui --legacy --exptid k1_v9_s1_landfix --checkpoint 5350 --num_envs 1 "$@"
    fi
fi

# python -c membiarkan stdin tersedia untuk kontrol interaktif viewer.
exec python -c '
import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

radius = 0.075
mass = 0.200
inertia = 2 * mass * radius**2 / 3  # Model kulit bola tipis seperti aset K1 asli.
ball = Path("legged_gym/resources/gymassets/urdf/ball_k1.urdf").resolve()
link = ET.parse(ball).find("link")
assert all(math.isclose(float(link.find(f"{s}/geometry/sphere").get("radius")), radius, rel_tol=1e-9) for s in ("visual", "collision"))
assert math.isclose(float(link.find("inertial/mass").get("value")), mass, rel_tol=1e-9)
assert all(math.isclose(float(link.find("inertial/inertia").get(a)), inertia, rel_tol=1e-9) for a in ("ixx", "iyy", "izz"))
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
