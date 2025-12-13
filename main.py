import os
import csv
import joblib
import pandas as pd
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional
from services.economic_data import get_current_economic_indicators

app = FastAPI(
    title="Bank Marketing Intelligence API",
    description="API untuk memprediksi potensi nasabah deposito dengan data ekonomi otomatis.",
    version="1.0.0"
)

# --- KONFIGURASI & LOAD MODEL ---
MODEL_PATH = "models/model_bank_lead_scoring.joblib"
MODEL_THRESHOLD = 0.27
model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded dari {MODEL_PATH}")
    else:
        print(f"RITICAL: Model tidak ditemukan di {MODEL_PATH}")

# --- SKEMA INPUT (Hanya Data yang Diisi Admin) ---
class AdminInputData(BaseModel):
    # Identitas (Opsional, untuk tracking saja)
    nama_nasabah: Optional[str] = Field(None, example="Budi Santoso")
    nomor_telepon: Optional[str] = Field(None, example="08123456789")

    # Data Demografis & Finansial
    age: int = Field(..., ge=17, le=100, example=35)
    job: str = Field(..., example="admin.")
    marital: str = Field(..., example="married")
    education: str = Field(..., example="university.degree")
    default: str = Field(..., example="no")
    housing: str = Field(..., example="yes")
    loan: str = Field(..., example="no")

    # Data Kampanye
    contact: str = Field(..., example="cellular")
    month: str = Field(..., example="may")
    day_of_week: str = Field(..., example="mon")
    duration: int = Field(..., ge=0, example=200, description="Durasi telepon dalam detik")
    campaign: int = Field(..., ge=1, example=1)
    pdays: int = Field(..., ge=0, example=999)
    previous: int = Field(..., ge=0, example=0)
    poutcome: str = Field(..., example="nonexistent")

# --- FEATURE ENGINEERING (Wajib Sama Persis dengan Training) ---
def apply_custom_features(df_input):
    df = df_input.copy()
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1)
    df['is_contacted_before'] = (df['pdays'] != 999).astype(int)
    df['is_success_campaign'] = (df['poutcome'] == 'success').astype(int)
    df['euribor_emp'] = df['euribor3m'] * df['nr.employed']
    df['fatigued_client'] = (df['campaign'] > 4).astype(int)
    return df

# --- ENDPOINTS ---

@app.get("/", tags=["General"])
def index():
    return {"status": "active", "service": "Bank ML API"}

@app.get("/health", tags=["General"])
def health_check():
    """Cek kesehatan sistem dan ketersediaan model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum siap")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict/nasabah", tags=["Prediction"])
def predict_nasabah(data: AdminInputData):
    """
    Endpoint utama untuk Admin.
    Menerima data nasabah, menggabungkan dengan data ekonomi terkini,
    dan mengembalikan rekomendasi.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model sedang offline.")

    try:
        # 1. Ambil data dari Admin
        input_dict = data.dict()
        nama = input_dict.pop("nama_nasabah")
        telp = input_dict.pop("nomor_telepon")

        # 2. Ambil data ekonomi terkini (Otomatis)
        economic_data = get_current_economic_indicators()
        full_input = {**input_dict, **economic_data}

        # 3. Buat DataFrame & Feature Engineering
        df_raw = pd.DataFrame([full_input])
        # Rename kolom agar sesuai dengan training data (jika ada titik)
        df_raw.rename(columns={
            'emp_var_rate': 'emp.var.rate',
            'cons_price_idx': 'cons.price.idx',
            'cons_conf_idx': 'cons.conf.idx',
            'nr_employed': 'nr.employed'
        }, inplace=True)
        
        df_ready = apply_custom_features(df_raw)

        # 4. Prediksi
        MODEL_THRESHOLD = 0.5  # Sesuaikan dengan hasil tuning terakhir Anda
        proba = model.predict_proba(df_ready)[0][1]
        prediction = 1 if proba >= MODEL_THRESHOLD else 0

        # 5. Siapkan Respon yang Mudah Dibaca Admin
        rekomendasi = "HUBUNGI SEGERA" if prediction == 1 else "JANGAN PRIORITASKAN"
        alasan = []
        if proba >= 0.8:
            alasan.append("Probabilitas sangat tinggi (>80%)")
        if df_ready['deposit_influencer'][0] == 1:
            alasan.append("Kondisi ekonomi mendukung & riwayat sukses")
        if df_ready['fatigued_client'][0] == 1:
             rekomendasi = "HATI-HATI (Risko Spam)"
             alasan.append("Nasabah sudah terlalu sering dihubungi")

        return {
            "nasabah": {"nama": nama, "telepon": telp},
            "hasil_analisis": {
                "rekomendasi": rekomendasi,
                "skor_potensi": f"{proba*100:.1f}%",
                "catatan_penting": alasan if alasan else ["Potensi standar"]
            },
            "data_ekonomi_digunakan": economic_data # Transparansi untuk admin
        }

    except Exception as e:
        # Log error sebenarnya di server, tapi jangan tampilkan detail ke user
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat memproses data.")
    
@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """
    Menerima file Excel (.xlsx) atau CSV berisi banyak data nasabah.
    Otomatis menambahkan indikator ekonomi terkini ke SEMUA data,
    lalu mengembalikan hasil prediksi untuk setiap baris.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model offline.")

    # 1. Validasi Format File
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.xls')):
        raise HTTPException(400, detail="Hanya terima .csv atau .xlsx")

    try:
        # 2. Baca File ke DataFrame
        contents = await file.read()
        if filename.endswith('.csv'):
            # Coba 3 kemungkinan delimiter — pasti salah satu benar
            for sep in [';', ',', '\t']:
                try:
                    temp_df = pd.read_csv(io.BytesIO(contents), sep=sep, nrows=5)
                    if len(temp_df.columns) > 10:  # kalau kolom >10 → benar
                        df_batch = pd.read_csv(io.BytesIO(contents), sep=sep, on_bad_lines='skip')
                        print(f"CSV berhasil dibaca dengan separator: '{sep}'")
                        break
                except:
                    continue
            else:
                raise HTTPException(400, detail="Gagal baca CSV — format tidak dikenali")
        else:
            df_batch = pd.read_excel(io.BytesIO(contents))

        # 3. Validasi Kolom Wajib (Harus ada di Excel user)
        required_columns = [
            'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome', 'duration'
        ]
        missing_cols = [col for col in required_columns if col not in df_batch.columns]
        if missing_cols:
            raise HTTPException(400, detail=f"Kolom wajib hilang di file: {missing_cols}")

        # 4. Injeksi Data Ekonomi Terkini (Sama untuk semua baris di batch ini)
        econ_data = get_current_economic_indicators()
        for col, val in econ_data.items():
            # Rename kolom ekonomi agar sesuai format training (jika ada titik)
            col_name = col
            if col == 'emp.var.rate': col_name = 'emp.var.rate' # Jaga-jaga
            elif col == 'cons.price.idx': col_name = 'cons.price.idx'
            elif col == 'cons.conf.idx': col_name = 'cons.conf.idx'
            elif col == 'nr.employed': col_name = 'nr.employed'
            
            df_batch[col_name] = val

        # 5. Feature Engineering & Prediksi Massal
        # Kita hanya ambil kolom yang dibutuhkan model untuk prediksi
        cols_for_model = required_columns + list(econ_data.keys())
        # Handle renaming key ekonomi jika perlu disesuaikan dengan training data persis
        # (Asumsi di atas sudah benar nama kolomnya dengan yang ada di training)

        df_ready = apply_custom_features(df_batch)

        # Prediksi probabilitas untuk semua baris sekaligus (Vectorized operation = CEPAT)
        probabilities = model.predict_proba(df_ready)[:, 1]

        # 6. Susun Hasil Akhir
        results = []
        for idx, row in df_batch.iterrows():
            prob = probabilities[idx]
            pred = 1 if prob >= MODEL_THRESHOLD else 0

            # Tambahkan insight per baris
            rekomendasi = "HUBUNGI" if pred == 1 else "IGNORE"
            notes = []
            if prob >= 0.8: notes.append("High Potential")
            if df_ready.iloc[idx]['fatigued_client'] == 1:
                rekomendasi = "HATI-HATI (Spam Risk)"
                notes.append("Terlalu sering dihubungi")

            # Kembalikan data asli + hasil prediksi
            row_res = row.to_dict()
            row_res.update({
                "PREDIKSI_REKOMENDASI": rekomendasi,
                "SKOR_PROBABILITAS": float(round(float(prob), 4)),
                "CATATAN": ", ".join(notes) if notes else "-"
            })
            results.append(row_res)

        return {
            "status": "success",
            "total_data": len(results),
            "data_ekonomi_digunakan": econ_data,
            "hasil_batch": results
        }

    except Exception as e:
        print(f"Batch Error: {e}")
        raise HTTPException(500, detail=f"Gagal memproses file: {str(e)}")