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

    # Default ke model utama landfix_5350 (atau v9_5000) bila tidak ditentukan
    k1_logs_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'k1')
    if args.exptid is None:
        if os.path.isdir(os.path.join(k1_logs_dir, 'k1_v9_s1_landfix')):
            args.exptid = 'k1_v9_s1_landfix'
            if args.checkpoint is None:
                args.checkpoint = 5350
        elif os.path.isdir(os.path.join(k1_logs_dir, 'k1_v9')):
            args.exptid = 'k1_v9'
            if args.checkpoint is None:
                args.checkpoint = 5000
        elif os.path.exists(k1_logs_dir):
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
    print("  Kontrol Interaktif Saat Simulasi Berjalan (Jendela 3D):")
    print("    Tombol 0 - 5      : Pilih Region 0-5 & langsung tembak")
    print("    Tombol 6 / C      : Mode Acak Semua Region")
    print("    Panah Kiri/Kanan  : Geser target Y (+0.10m kiri / -0.10m kanan) & tembak")
    print("    Panah Atas/Bawah  : Geser tinggi Z (+0.10m atas / -0.10m bawah) & tembak")
    print("    Tombol = / - (F/G): Tambah / kurangi kecepatan bola (+0.5 / -0.5 m/s)")
    print("    Tombol ] / [ (E/Q): Jauhkan / dekatkan jarak tendangan (+0.5 / -0.5 m)")
    print("    Tombol R / Enter  : Tembak ulang sekarang")
    print("    Tombol Space      : Pause / lanjut simulasi")
    print("  ----------------------------------------------------------------------")
    print("  Input Langsung di Terminal (ketik lalu tekan Enter kapan saja):")
    print("    v 12              -> Set kecepatan bola = 12 m/s & tembak")
    print("    x 4.5             -> Set jarak tendangan (start_x) = 4.5 m & tembak")
    print("    vx 12 4.5         -> Set kecepatan = 12 m/s & jarak = 4.5 m sekaligus")
    print("    0 .. 5            -> Tembak ke Region 0..5")
    print("    0.5 0.7           -> Tembak presisi ke Y=+0.5m, Z=0.7m")
    print("    -0.5 0.8 12 4.5   -> Tembak ke Y=-0.5m, Z=0.8m, speed=12m/s, jarak=4.5m")
    print("    acak / reset      -> Acak region / reset semua ke bawaan")
    print("=" * 70 + "\n")
    play(args)
