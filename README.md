# 📷 Smart CCTV – YOLO + Pose (Deteksi Manusia & Hewan)

## ▶️ Cara Menggunakan (WAJIB DIBACA)

1. Pastikan **Python 3.8 – 3.11** sudah terinstall
2. Install library yang dibutuhkan:

   ```bash
   pip install ultralytics opencv-python playsound requests
   ```
3. Siapkan file berikut dalam satu folder:

   ```
   ├── main.py
   ├── yolov8s-world.pt
   ├── yolov8n-pose.pt
   ├── alarm.wav
   └── snapshots/
   ```
4. Jalankan program:

   ```bash
   python main.py
   ```
5. Tekan **Q** untuk keluar dari aplikasi

---

## 📌 Fungsi Program

Program ini adalah **Smart CCTV berbasis AI** untuk:

* Mendeteksi **manusia dan hewan** secara real-time
* Mengurangi salah deteksi menggunakan **pose manusia**
* Membunyikan **alarm** saat hewan terdeteksi
* Mengambil **snapshot otomatis**
* Mengirim **notifikasi foto ke Telegram**
* Menyimpan **riwayat deteksi** ke file teks

---

## 🧠 Cara Kerja Singkat

1. Kamera menangkap video
2. YOLO mendeteksi objek (manusia / hewan)
3. Jika bukan manusia, dicek menggunakan **YOLO Pose**
4. Jika hewan terdeteksi beberapa frame:

   * Alarm berbunyi
   * Foto diambil
   * Foto dikirim ke Telegram

---

## 🐾 Objek yang Dideteksi

* Manusia
* Anjing, Kucing, Ular, Ayam, Burung, Sapi, Kambing

---

## 🔔 Alarm & Anti Spam

* Alarm aktif jika hewan muncul **≥ 3 frame berturut-turut**
* Delay alarm: **10 detik**
* Snapshot tidak dikirim berulang (anti spam)

---

## 📩 Telegram

Edit token dan chat ID:

```python
TELEGRAM_BOT_TOKEN = "ISI_TOKEN_BOT"
TELEGRAM_CHAT_ID = "ISI_CHAT_ID"
```

Telegram akan menerima **foto + nama hewan + waktu**.

---

## 📂 Output

* `snapshots/` → foto hasil deteksi
* `riwayat_deteksi.txt` → log kejadian

---

## 📝 Catatan

* Gunakan kamera posisi tetap
* Pencahayaan cukup agar deteksi stabil
* Resolusi default: **640x480**

---

## 🚀 Penutup

Cocok untuk **keamanan rumah, kebun, kandang, atau sawah**.
Dapat dikembangkan ke CCTV IP, dashboard web, atau multi kamera.
