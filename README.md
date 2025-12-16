#  Balık Ağırlık Tahmin Uygulaması

Makine öğrenmesi kullanarak balığın fiziksel özelliklerinden ağırlığını tahmin eden web uygulaması.

## 🖼 Proje Önizlemesi

<img width="697" height="825" alt="image" src="https://github.com/user-attachments/assets/87a5975f-ac56-4d0e-8938-58be6301a4bd" />

*Yukarıdaki görsel: Flask web arayüzü - kullanıcıdan balık özelliklerini alan form*

##  Projenin Amacı

Bu projede balık ağırlığını tahmin etmek için:

1️⃣ **Backward Elimination** ile özellik seçimi yapıldı  
2️⃣ **OLS (Ordinary Least Squares) Regresyon** modeli kuruldu  
3️⃣ Model `.pkl` formatında kaydedildi  
4️⃣ **Flask** tabanlı bir web arayüzü ile kullanıcıdan veri alınıp gerçek zamanlı tahmin yapılması sağlandı

Veri bilimi sürecinin tüm aşamaları `.ipynb` dosyasında açıklamalı olarak gösterilmiştir.

---

## 1️⃣ Veri Ön İşleme (Data Preprocessing)

###  Öznitelik (Feature) Seçimi

Kullandığım veri setinde şu öznitelikler bulunuyordu:

- **Length1** → Dikey uzunluk (cm)
- **Length2** → Çapraz uzunluk (cm)
- **Length3** → En uzun çapraz uzunluk (cm)
- **Height** → Yükseklik (cm)
- **Width** → Genişlik (cm)
- **Species** → Balık türü (Bream, Pike, Perch, Roach, Smelt, Whitefish, Parkki)
- **Weight** → Hedef değişken (ağırlık - gram)

<img width="538" height="521" alt="image" src="https://github.com/user-attachments/assets/34f78eaf-4c86-45c2-8d0a-91fb600180f5" />


*Yukarıdaki görsel: `df.info()` çıktısı - veri setinin yapısı, kolonlar ve veri tipleri*

Bu özniteliklerin hepsi anlamlı olduğu için veri setinden çıkarılacak "gereksiz" bir kolon yoktu. Ayrıca öznitelik sayısı 7 olup ödev gereği belirlenen maksimum 10 sınırının içerisindedir.

###  Kayıp Veri (Missing Values) Analizi

`.isnull().sum()` ile veri incelendi:

<img width="253" height="276" alt="image" src="https://github.com/user-attachments/assets/e0f4adcf-2a84-42e5-b1e8-ef59a4c90338" />

*Yukarıdaki görsel: Eksik veri kontrolü - tüm kolonlarda 0 eksik veri*

**Sonuç**: Hiçbir eksik veri bulunmadığı için doldurma (imputation) işlemi yapılmasına gerek olmadı. Eksik veri olsaydı ortalama/medyan ile doldurmayı tercih ederdik.

###  Kategorik Verilerin Kodlanması (Encoding)

####  One-Hot Encoding neden tercih edildi?

Species (Balık Türü) gibi değişkenler nominal (sırasız) kategoriktir.

**Label Encoding kullanılsaydı:**
- Bream = 1, Pike = 2 gibi değerler atanacak
- Model bu sayıları sıralı sanıp yanlış ilişki kuracaktı

**✔ Bu nedenle One-Hot Encoding en doğru yöntemdir:**
- Kategoriler ayrı sütunlarda 0/1 şeklinde gösterilir
- "Dummy Variable Trap" oluşmaması için `drop_first=True` kullanıldı

<img width="1169" height="254" alt="image" src="https://github.com/user-attachments/assets/a6090291-4734-47ab-a36f-e55d098a053b" />

*Yukarıdaki görsel: `df_encoded.head()` çıktısı - Species kolonu one-hot encoding ile dönüştürüldü*

###  Veri Ölçekleme

**Kullanılan Yöntem**: StandardScaler

**Gerekçe**:
- Length1, Length2, Length3, Height, Width gibi özellikler farklı ölçeklerdedir (cm cinsinden)
- Ölçekleme yapılmazsa, büyük değerlere sahip özellikler (örn: Length3) model üzerinde daha fazla etkili olabilir
- StandardScaler kullanarak tüm özellikler ortalaması 0, standart sapması 1 olacak şekilde normalize edilir
- Bu işlem modelin daha iyi performans göstermesini sağlar

**Not**: Scaler sadece eğitim verisiyle fit edilir, test verisi sadece transform edilir (data leakage önlemek için)

---

## 2️⃣ Geriye Doğru Eleme (Backward Elimination)

Modelin istatistiksel anlamlılığını ölçmek için:

1. Tüm özelliklerle OLS modeli kuruldu
2. p-value değerleri kontrol edildi
3. **p > 0.05** olan kolonlar tek tek elenerek model sadeleştirildi

**Süreç**:
- Her iterasyonda en yüksek p-value'ya sahip özellik bulunur
- Eğer p-value > 0.05 ise, o özellik çıkarılır
- Tüm özelliklerin p-value'su < 0.05 olana kadar işlem devam eder

**Sonuç**:

<img width="755" height="723" alt="image" src="https://github.com/user-attachments/assets/a7a4b141-2e14-4663-89bf-d18e837fad69" />

*Yukarıdaki görsel: Backward Elimination süreci - hangi özelliklerin çıkarıldığı gösteriliyor*

**Seçilen Özellikler**:
- ✔ Length2 (Çapraz Uzunluk) - En önemli özellik
- ✔ Species_Parkki
- ✔ Species_Pike
- ✔ Species_Smelt

**Elenen Özellikler**:
- ❌ Width (p-value: 0.798)
- ❌ Species_Roach (p-value: 0.571)
- ❌ Species_Whitefish (p-value: 0.705)
- ❌ Species_Perch (p-value: 0.533)
- ❌ Length3 (p-value: 0.343)
- ❌ Height (p-value: 0.667)
- ❌ Length1 (p-value: 0.158)

---

## 3️⃣ Model Kurulumu ve Değerlendirme

Eğitim/Test ayrımı yapıldıktan sonra OLS (Ordinary Least Squares) Regresyon modeli eğitildi.

### 📌 Final Model Özeti


*Yukarıdaki görsel: OLS Regression Results - model katsayıları, p-value'lar ve istatistikler*

**Model İstatistikleri**:
- **R² (R-squared)**: 0.926 (%92.6)
- **Adj. R-squared**: 0.924
- **F-statistic**: 382.5
- **Prob (F-statistic)**: 5.33e-68 (çok yüksek anlamlılık)
- **No. Observations**: 127 (eğitim seti)

**Katsayılar (Coefficients)**:
- **const**: 386.79 (sabit terim)
- **Length2**: 419.16 (en yüksek katsayı - en önemli özellik)
- **Species_Parkki**: 18.22 (p-value: 0.046)
- **Species_Pike**: -113.46 (p-value: 0.000)
- **Species_Smelt**: 62.75 (p-value: 0.000)

### 📌 Performans Metrikleri

<img width="311" height="84" alt="image" src="https://github.com/user-attachments/assets/e03db41a-d800-420b-bb58-f29535e0b6fb" />

*Yukarıdaki görsel: Test seti üzerinde hesaplanan performans metrikleri*

**Test Seti Sonuçları**:

- **R² Score**: **0.9477** (%94.77) → Model veriyi %95 oranında açıklamaktadır
- **MAE (Mean Absolute Error)**: **65.43 gram** → Ortalama tahmin hatası
- **MSE (Mean Squared Error)**: **7432.55**
- **RMSE (Root Mean Squared Error)**: **86.21 gram** → Karekök ortalama karesel hata

### 📌 Yorum

- **R² skorunun 0.95 olması** modelin balık ağırlığındaki değişimin %95'ini açıkladığını gösterir → **çok iyi bir sonuç**
- **MAE ≈ 65 gram**, ortalama tahmin hatasını ifade eder
- Balıkların ağırlıkları 200-1500 gram bandında değiştiği düşünülürse, 65 gramlık ortalama hata **başarılı bir sonuçtur**
- **Length2** özelliği en önemli tahmin edici değişkendir (katsayı: 419.16)

---

## 4️⃣ Flask Arayüz Uygulaması

Model `.pkl` dosyasına kaydedildi:
- `fish_model.pkl` - Eğitilmiş OLS regresyon modeli
- `scaler.pkl` - StandardScaler nesnesi
- `feature_names.json` - Tüm özellik isimleri
- `be_cols.json` - Backward Elimination sonrası seçilen özellikler

### 💻 Kullanıcıdan Alınan Veriler

<img width="697" height="825" alt="image" src="https://github.com/user-attachments/assets/87a5975f-ac56-4d0e-8938-58be6301a4bd" />

*Yukarıdaki görsel: Flask web arayüzü - kullanıcıdan balık özelliklerini alan form*

**Form Alanları**:
1. **Balık Türü**: Dropdown menüden seçim (Bream, Pike, Perch, Roach, Smelt, Whitefish, Parkki)
2. **Uzunluk 1**: Dikey uzunluk (cm)
3. **Uzunluk 2**: Çapraz uzunluk (cm)
4. **Uzunluk 3**: En uzun çapraz uzunluk (cm)
5. **Yükseklik**: Balığın yüksekliği (cm)
6. **Genişlik**: Balığın genişliği (cm)

###  Arayüz Özellikleri

<img width="697" height="550" alt="image" src="https://github.com/user-attachments/assets/5cbaa817-1139-484f-a2a2-3edc0e7daf7f" />

*Yukarıdaki görsel: Balık türü seçim dropdown menüsü*

- ✅ Türkçe kullanıcı arayüzü
- ✅ Gerçek zamanlı form validasyonu
- ✅ Her alan için açıklayıcı hint yazıları
- ✅ Placeholder örnekleri
- ✅ Görsel geri bildirim (geçerli/geçersiz alanlar)
- ✅ Form değerleri korunur (submit sonrası sıfırlanmaz)

### Tahmin Sonucu
<img width="786" height="878" alt="image" src="https://github.com/user-attachments/assets/dc37873f-6e80-49e2-8d4d-593419b0bd23" />

*Yukarıdaki görsel: Form doldurulduktan sonra gösterilen tahmin sonucu*

**Örnek Tahmin**:
- Girdi: Smelt türü, Length1=23.4, Length2=25.3, Length3=30, Height=11.56, Width=4.08
- **Tahmin Edilen Ağırlık: 526.48 gram**

---

##  Kurulum ve Kullanım

### Gereksinimler

- Python 3.7 veya üzeri
- pip (Python paket yöneticisi)

### Adım 1: Projeyi İndirin

git clone https://github.com/kosebaris1/MLP_Flask.git
cd Fish_Weight_Predictor### Adım 2: Sanal Ortam Oluşturun (Önerilen)

# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate### Adım 3: Gerekli Paketleri Yükleyin

pip install flask pandas numpy scikit-learn statsmodels### Adım 4: Uygulamayı Başlatın

python app.pyUygulama `http://localhost:5000` adresinde çalışacaktır.

---

