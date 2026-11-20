# Kalibrasi awal reward K1 — 16 November 2026

## Keputusan

Dua pilot GPU sudah dijalankan, masing-masing 100 iterasi dari checkpoint 3400. Keduanya belum lolos sebagai pengganti model utama. Kandidat B sedikit mengurangi waktu kedua tangan terangkat pada dua shot atas samping, tetapi kebobolan satu kali dan masih sering melewati ambang postur. Belum ada bukti gerak manusiawi atau peningkatan performa umum.

Checkpoint asli `legged_gym/logs/k1/k1_v2/model_3400.pt` tidak berubah. SHA-256 sebelum dan sesudah kedua training:

```text
ff0c6651a89326e061e69a252938c958a574bf3e0448963b285f78f41ba3d14c
```

Run eksperimen disimpan di `legged_gym/logs/k1_calibration/`, terpisah dari direktori pencarian otomatis `play_k1_size1.sh`. Jangan menyebut checkpoint eksperimen sebagai model final.

## Protokol

- Pilot A dan B mulai dari bobot 3400 yang sama, bukan B melanjutkan A.
- Masing-masing 384 environment, seed 42, 100 iterasi tambahan, 64 langkah rollout per iterasi: 2.457.600 transisi per pilot.
- Optimizer baru, learning rate tetap 0,0001. Randomisasi domain training tetap memakai konfigurasi K1.
- Horizon training diperpanjang dari 1,8 menjadi 5 detik agar ada waktu untuk escape bola dan pemulihan stabil.
- Evaluasi memakai sembilan shot `k1_motion_shots.csv`, seed startup 42 dan seed shot 0. Semua menggunakan horizon 5 detik dan observasi pascakontak yang baru.
- Baseline di sini adalah 3400 pada kode observasi baru. Angkanya tidak boleh disamakan dengan audit lama yang menyembunyikan bola setelah kontak.
- Gangguan gaya bola yang tersisa masih aktif, meskipun noise observasi dan opsi randomisasi domain evaluator dimatikan. GPU tidak dijamin deterministik secara bitwise.
- Evaluator sekarang menyimpan `episode_rewards`, yaitu jumlah reward berbobot per komponen selama satu episode, dipulihkan dari rate logger pada langkah terminal. Data ini terpisah dari metrik logger training yang masih dapat menghitung catatan episode lama berulang kali.

## Kandidat yang diuji

| Parameter | Pilot A | Pilot B |
|---|---:|---:|
| Reach | 26 | 6,5 |
| Penalti kecepatan sendi | -0,0002 | -0,002 |
| Penalti perubahan action tingkat dua | -0,012 | -0,12 |
| Bonus save terminal | +40 | +40 |
| Bonus pemulihan terminal | +5 | +5 |
| Penalti gol terminal | -220 | -220 |
| Penalti penghentian postur per kejadian, dt=0,02 | -4,4 | -220 |

Pilot A menunjukkan rata-rata reward reach sekitar 44,38 per episode pada sembilan shot, sedangkan rata-rata bonus save hanya 4,44. Kandidat B mengurangi reach agar shaping tidak mendominasi hasil bola; rata-rata reach pada sampel evaluasinya menjadi 9,39. Sampel perilakunya berbeda, sehingga angka ini bukan efek kausal bobot saja.

Penalti penghentian postur sebelumnya dikalikan dt seperti reward kontinu. Pada K1, penalti itu sekarang dibagi dt sebelum perkalian skala, seperti reward event lain. Ini menutup keuntungan mengakhiri episode dini dengan biaya kecil setelah menerima shaping reward. Tidak ada penalti khusus yang memaksa satu tangan diam.

Nilai B merupakan konfigurasi kandidat saat ini, bukan parameter yang sudah terbukti optimal. Reward dan definisi label evaluasi tetap harus dibedakan.

## Hasil sembilan shot

| Ukuran | 3400, observasi baru | Pilot A | Pilot B |
|---|---:|---:|---:|
| Save menurut label evaluasi lama | 9 | 7 | 8 |
| Gol | 0 | 0 | 1 |
| Unresolved | 0 | 2 | 0 |
| Penghentian sebelum timeout karena ambang postur | 9 | 7 | 7 |
| Timeout dengan bola aman menurut syarat reward baru | 0 | 1 | 1 |
| Episode mendapat bonus pemulihan | Tidak direkam | 1 | 1 |
| Rata-rata total perpindahan 8 sendi lengan, detik pertama (rad) | 24,921 | 23,859 | 24,926 |
| Puncak kecepatan sendi lengan, detik pertama (rad/s) | 18,900 | 19,884 | 19,175 |
| Rata-rata norma perubahan action lengan per langkah | 0,3490 | 0,3196 | 0,3337 |

Label save lama dapat muncul bersamaan dengan penghentian karena postur. Label itu tidak membuktikan robot menyelesaikan pemulihan atau menerima bonus save terminal baru. Melewati ambang postur juga bukan bukti langsung robot jatuh ke lantai.

Untuk shot tinggi samping, proporsi frame ketika **kedua tangan berada di atas 0,65 m**, selama detik pertama:

| Shot | 3400 | Pilot A | Pilot B |
|---|---:|---:|---:|
| 202, atas kanan robot | 72% | 72% | 58% |
| 206, atas kiri robot | 64% | 74% | 58% |

Ambang 0,65 m adalah deskripsi trace, bukan skor manusiawi. Dua tangan tetap sering naik bersama. Tidak ada referensi mocap K1 atau penilaian visual manusia yang digunakan untuk menyimpulkan naturalitas.

## Berkas bukti

- Baseline: `k1_review_20261116_160304_886439.jsonl` beserta `.motion.jsonl`.
- Pilot A: `k1_review_20261116_160919_696764.jsonl` beserta `.motion.jsonl`.
- Pilot B: `k1_review_20261116_162226_687112.jsonl` beserta `.motion.jsonl`.
- Ringkasan terhitung: `k1_recovery_calibration/comparison.json`.
- Konfigurasi, hash model, dan penolakan promosi: `k1_recovery_calibration/pilot_a_config.json` dan `pilot_b_config.json`.
- Kandidat A: `legged_gym/logs/k1_calibration/k1_recovery_pilot_20261116_a/model_3499.pt`.
- Kandidat B: `legged_gym/logs/k1_calibration/k1_recovery_pilot_20261116_b/model_3499.pt`.
- Log training lengkap berada dalam masing-masing folder run sebagai `training.log`.

Nomor 3499 berasal dari penghitung runner saat ini. Kedua pilot menjalankan tepat 100 update tambahan; nomor berkas bukan bukti riwayat training checkpoint awal.

## Mengulang

Dari root repository, aktifkan environment `goalkeeper`. Gunakan nama run baru; skrip menolak direktori tujuan yang sudah ada.

```sh
python evaluation/k1_recovery_calibration/pilot.py --task k1 --exptid k1_recovery_pilot_repeat --num_envs 384 --seed 42 --max_iterations 100 --headless
python legged_gym/legged_gym/scripts/evaluate_k1.py --task k1 --experiment_name k1_calibration --exptid k1_recovery_pilot_repeat --checkpoint 3499 --seed 42 --headless --trace_motion --shot_file evaluation/k1_motion_shots.csv
```

Skrip memakai konfigurasi sumber saat dijalankan. Konfigurasi B tersedia di sumber saat ini; detail A disimpan dalam artefak konfigurasi dan patch. Nomor checkpoint keluaran harus diperiksa jika metadata checkpoint awal atau jumlah iterasi berubah.

## Batas dan langkah berikutnya

38 unit tests lolos, termasuk pembayaran event yang tidak bergantung dt, pembatalan bonus ketika gol, pelacakan setelah kontak, reset penghitung stabil, dan penalti penghentian dini. Evaluasi GPU membuktikan checkpoint tetap dapat dimuat dan reward baru benar-benar tercatat.

Pilot belum memenuhi syarat untuk menggantikan baseline: keberhasilan bola harus tidak turun, penghentian karena postur harus berkurang, dan kecepatan sendi tidak boleh naik sambil pola dua tangan diperbaiki. Karena pilot sudah gagal pada sembilan shot, belum dilakukan evaluasi penuh 180 shot, training panjang, atau pengujian hardware. Uji berikutnya perlu mengisolasi perubahan observasi pascakontak dan membandingkan durasi training yang lebih panjang sebelum menambah aturan gerak tangan.
