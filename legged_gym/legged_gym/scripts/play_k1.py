import os
import sys

import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.scripts.play import play

if __name__ == '__main__':
    args = get_args()
    args.task = 'k1'
    args.headless = False

    # Deteksi eksperimen k1 yang tersedia
    k1_logs_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'k1')
    if args.exptid is None:
        if os.path.exists(k1_logs_dir):
            runs = [
                d for d in os.listdir(k1_logs_dir)
                if os.path.isdir(os.path.join(k1_logs_dir, d)) and d != 'exported'
            ]
            if runs:
                runs.sort(key=lambda d: os.path.getmtime(os.path.join(k1_logs_dir, d)))
                args.exptid = runs[-1]
            else:
                args.exptid = 'test'
        else:
            args.exptid = 'test'

    print(f"\n" + "="*70)
    print(f"  VISUALISASI KIPER HUMANID BOOSTER K1 (Run: logs/k1/{args.exptid})")
    print("="*70)
    print("  Wilayah Sasaran Bola:")
    print("    Region 0: Samping Kiri Robot (Dada/Sedang)")
    print("    Region 1: Samping Kanan Robot (Dada/Sedang)")
    print("    Region 2: Pojok Atas Kiri Robot (Area Mistar / Lompat)")
    print("    Region 3: Pojok Atas Kanan Robot (Area Mistar / Lompat)")
    print("    Region 4: Bawah Kiri Robot (Menyusur Tanah)")
    print("    Region 5: Bawah Kanan Robot (Menyusur Tanah)")
    print("  ----------------------------------------------------------------------")
    print("  Contoh Opsi Custom Arah Bola:")
    print("    --region 2                  (Tembak ke Pojok Atas Kiri)")
    print("    --target_y 0.8 --target_z 1.1   (Tembak presisi koordinat Y=0.8m, Z=1.1m)")
    print("    --target_y 0.0 --target_z 0.5   (Tembak lurus ke tengah dada)")
    print("    --num_envs 1                (Hanya tampilkan 1 robot tunggal)")
    print("    --flight_time 0.5           (Tembakan kencang 0.5 detik)")
    print("="*70 + "\n")
    play(args)
