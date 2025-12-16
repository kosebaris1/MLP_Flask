import pickle
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from flask import Flask, render_template, request

app = Flask(__name__)

# 1) Model ve scaler'ı yükle
with open("fish_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 2) Tüm feature isimleri (scaler bunlara göre eğitim aldı)
with open("feature_names.json", "r") as f:
    feature_names = json.load(f)

# 3) Backward Elimination sonrası seçilen kolonlar
with open("be_cols.json", "r") as f:
    be_cols = json.load(f)

# BE sırasında sabit terim de listede olabilir, **çıkarıyoruz**
be_cols = [c for c in be_cols if c != "const"]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    form_data = {
        "species": "",
        "length1": "",
        "length2": "",
        "length3": "",
        "height": "",
        "width": ""
    }

    if request.method == "POST":
        try:
            # Form değerlerini al (hata durumunda da korumak için)
            form_data["species"] = request.form.get("species", "")
            form_data["length1"] = request.form.get("length1", "")
            form_data["length2"] = request.form.get("length2", "")
            form_data["length3"] = request.form.get("length3", "")
            form_data["height"] = request.form.get("height", "")
            form_data["width"] = request.form.get("width", "")
            
            # ---- Formdan gelen değerler ve validasyon ----
            # Boş değer kontrolü
            if not request.form.get("species"):
                error = "Lütfen bir balık türü seçiniz."
                return render_template("index.html", prediction=prediction, error=error, form_data=form_data)
            
            # Sayısal değerleri al ve kontrol et
            try:
                length1 = float(request.form.get("length1", 0))
                length2 = float(request.form.get("length2", 0))
                length3 = float(request.form.get("length3", 0))
                height = float(request.form.get("height", 0))
                width = float(request.form.get("width", 0))
            except (ValueError, TypeError):
                error = "Lütfen tüm alanları geçerli sayısal değerlerle doldurunuz."
                return render_template("index.html", prediction=prediction, error=error, form_data=form_data)
            
            species = request.form["species"]
            
            # Değer aralığı kontrolü
            if length1 <= 0 or length1 > 200:
                error = "Uzunluk 1 değeri 0 ile 200 cm arasında olmalıdır."
                return render_template("index.html", prediction=prediction, error=error, form_data=form_data)
            
            if length2 <= 0 or length2 > 200:
                error = "Uzunluk 2 değeri 0 ile 200 cm arasında olmalıdır."
                return render_template("index.html", prediction=prediction, error=error, form_data=form_data)
            
            if length3 <= 0 or length3 > 200:
                error = "Uzunluk 3 değeri 0 ile 200 cm arasında olmalıdır."
                return render_template("index.html", prediction=prediction, error=error, form_data=form_data)
            
            if height <= 0 or height > 50:
                error = "Yükseklik değeri 0 ile 50 cm arasında olmalıdır."
                return render_template("index.html", prediction=prediction, error=error, form_data=form_data)
            
            if width <= 0 or width > 50:
                error = "Genişlik değeri 0 ile 50 cm arasında olmalıdır."
                return render_template("index.html", prediction=prediction, error=error, form_data=form_data)
            
            # Mantık kontrolü - uzunluklar
            if length2 < length1:
                error = "Uzunluk 2, Uzunluk 1'den küçük olamaz."
                return render_template("index.html", prediction=prediction, error=error, form_data=form_data)
            
            if length3 < length2:
                error = "Uzunluk 3, Uzunluk 2'den küçük olamaz."
                return render_template("index.html", prediction=prediction, error=error, form_data=form_data)
            
            # Mantık kontrolü - yükseklik ve genişlik
            if width > height * 2:
                error = "Genişlik, yüksekliğin 2 katından fazla olamaz."
                return render_template("index.html", prediction=prediction, error=error, form_data=form_data)
            
            # Geçerli tür kontrolü
            valid_species = ["Bream", "Roach", "Whitefish", "Parkki", "Perch", "Pike", "Smelt"]
            if species not in valid_species:
                error = "Geçersiz balık türü seçildi."
                return render_template("index.html", prediction=prediction, error=error, form_data=form_data)

            # ---- 1 satırlık feature dataframe'i oluştur ----
            # Önce tüm feature'lar için 0'larla bir dict hazırlıyoruz
            data = {col: 0.0 for col in feature_names}

            # NOT: Buradaki kolon isimleri Colab'deki X dataframe'indeki ile aynı olmalı
            # Örn: X.columns -> ["Length1","Length2","Length3","Height","Width", ...]
            data["Length1"] = length1
            data["Length2"] = length2
            data["Length3"] = length3
            data["Height"] = height
            data["Width"] = width

            # Species dummy kolonlarını ayarla
            # Bream base category olduğu için (drop_first=True) hiçbir şey set etmiyoruz
            if species == "Parkki":
                data["Species_Parkki"] = 1.0
            elif species == "Perch":
                data["Species_Perch"] = 1.0
            elif species == "Pike":
                data["Species_Pike"] = 1.0
            elif species == "Roach":
                data["Species_Roach"] = 1.0
            elif species == "Smelt":
                data["Species_Smelt"] = 1.0
            elif species == "Whitefish":
                data["Species_Whitefish"] = 1.0
            # Bream için hiçbir şey yapmıyoruz, tüm dummy'ler 0 kalacak

            df = pd.DataFrame([data])  # (1, n_features)

            # ---- Ölçekleme (scaler eğitimde ne gördüyse aynı sırada) ----
            scaled = scaler.transform(df)
            df_scaled = pd.DataFrame(scaled, columns=feature_names)

            # ---- BE sonrası seçilen kolonlar ----
            df_be = df_scaled[be_cols]

            # !!!!! ÖNEMLİ KISIM !!!!!
            # Statsmodels modeli sabit terimle (const) eğitildi,
            # bu yüzden tahmin öncesi tekrar const ekliyoruz:
            df_be_const = sm.add_constant(df_be, has_constant="add")

            # ---- Tahmin ----
            y_pred = model.predict(df_be_const)[0]  # shape uyuşuyor: (1, k) · (k,) -> (1,)
            prediction = round(float(y_pred), 2)
            
            # Negatif tahmin kontrolü
            if prediction < 0:
                error = "Tahmin edilen değer negatif çıktı. Lütfen girdi değerlerini kontrol ediniz."
                prediction = None
                return render_template("index.html", prediction=prediction, error=error, form_data=form_data)
        
        except Exception as e:
            error = f"Bir hata oluştu: {str(e)}"
            return render_template("index.html", prediction=prediction, error=error, form_data=form_data)

    return render_template("index.html", prediction=prediction, error=error, form_data=form_data)


if __name__ == "__main__":
    # Debug açık olursa hata ekranını görüyorsun
    app.run(debug=True)
