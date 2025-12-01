import sys
import io
import base64
import cv2
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn import modules # Kita butuh akses ke modul internal ultralytics

# ==================~========================
# 1. TEMPEL ARSITEKTUR CUSTOM KAMU DI SINI
#    (Wajib ada biar .pt bisa dibaca)
# ==========================================

# Import komponen dasar YOLO biar tidak error
from ultralytics.nn.modules import Conv, C3 

class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, g=g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, g=c_, act=act)
    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)

class GhostBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            GhostConv(c_, c2, k, s, act=False)
        )
        self.shortcut = nn.Sequential(Conv(c1, c2, 1, s, act=False)) if s == 2 or c1 != c2 else nn.Identity()
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class C3Ghost(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

# ==========================================
# 2. "SUNTIK" CLASS KE ULTRALYTICS
#    (Biar library YOLO kenal sama class di atas)
# ==========================================
setattr(modules, 'GhostConv', GhostConv)
setattr(modules, 'GhostBottleneck', GhostBottleneck)
setattr(modules, 'C3Ghost', C3Ghost)

# ==========================================
# 3. SETUP FLASK & LOAD MODEL
# ==========================================
app = Flask(__name__)

# Ganti path ini sesuai lokasi file .pt kamu
MODEL_PATH = 'best.pt' 

print(f"Sedang meload model custom dari: {MODEL_PATH} ...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model BERHASIL di-load!")
except Exception as e:
    print(f"❌ Gagal load model: {e}")
    print("Pastikan file .pt ada di folder yang sama atau path-nya benar.")

@app.route('/')
def home():
    # Pastikan file index.html ada di folder 'templates'
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Baca Gambar
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # --- PREDIKSI ---
        results = model(img, conf=0.25) # conf = threshold keyakinan
        
        # --- PROSES HASIL ---
        # 1. Ambil gambar hasil plot (kotak-kotak deteksi)
        res_plotted = results[0].plot()
        
        # 2. Convert warna BGR (OpenCV) ke RGB (Web)
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # 3. Encode ke string base64 buat dikirim ke HTML
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR))
        encoded_img = base64.b64encode(buffer).decode('utf-8')

        # 4. Ambil data teks (Label & Confidence)
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({
                "class": cls_name,
                "confidence": round(conf, 2)
            })

        return jsonify({
            'message': 'Success',
            'image_data': encoded_img,
            'detections': detections
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)