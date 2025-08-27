# Akademik Yayın Tahmin ve Analiz Sistemi

Bu proje, belirli akademik alanlar için geçmiş yıllara ait yayın sayıları verilerini kullanarak **gelecekteki yayın sayılarını tahmin eden** bir analiz sistemidir. Proje, hem **Facebook Prophet** hem de **Linear Regression (numpy.polyfit kullanılarak uygulanmıştır)** modellerini kullanarak farklı tahmin stratejileri sunar.

---

## 📌 Özellikler

- **SQL Server bağlantısı** ile doğrudan veritabanından veri çekme.
- **Prophet** ve **Linear Regression (numpy.polyfit ile)** modelleri ile tahmin yapma.
- **Yıllık yayın sayısı analizleri**.
- **Alan bazlı filtreleme ve karşılaştırma**.
- Kullanıcı dostu **etkileşimli terminal menüsü**.
- **Grafiksel çıktı** üretimi (Matplotlib & Seaborn).
- Esnek **tahmin yılı ve model seçimi**.

---

## 🛠 Kullanılan Teknolojiler

- **Python** (3.8+)
- **pandas** – Veri işleme
- **numpy** – Matematiksel işlemler, Linear Regression numpy.polyfit ile
- **matplotlib / seaborn** – Grafik ve görselleştirme
- **prophet** – Zaman serisi tahmini
- **SQLAlchemy** – Veritabanı bağlantısı
- **python-dotenv** – Ortam değişkenleri yönetimi
- **SQL Server** – Veri kaynağı

---

## 📂 Proje Yapısı

```
cu_veri_analizi/
│
├── analiz.py          # Ana Python kodu (veri çekme, analiz ve tahmin)
├── .env               # Veritabanı bağlantı bilgileri (gizli)
├── .gitignore         # Git izleme dışı dosyalar
├── sorgu.sql          # SQL sorgu dosyası (isteğe bağlı, .gitignore altında)
├── *.png              # Üretilen grafikler proje klasörüne kaydedilir
└── README.md          # Proje dokümantasyonu
```

---

## ⚙️ Kurulum

1. Depoyu klonlayın:
```bash
git clone https://github.com/<kullanici-adi>/cu-veri-analizi.git
cd cu-veri-analizi
```

2. Gerekli Python paketlerini yükleyin (requirements.txt dosyası mevcutsa):
```bash
pip install -r requirements.txt
```

3. `.env` dosyasını oluşturun ve veritabanı bilgilerinizi girin:
```
DB_HOST=server_adresi
DB_USER=kullanici_adi
DB_PASSWORD=sifre
DB_NAME=veritabani_adi
DB_DRIVER=ODBC Driver 18 for SQL Server
DB_ENCRYPT=Optional
DB_TRUST_SERVER_CERT=true
```
> Not: Mac'te yüklü sürücü 17 ise `DB_DRIVER=ODBC Driver 17 for SQL Server` yazın.

---

## ▶️ Kullanım

Programı çalıştırmak için:
```bash
python analiz.py
```

Çalıştırma sırasında kullanıcıdan:
- **Model seçimi** (Prophet veya Linear Regression)
- **Tahmin yılı** (varsayılan 2026)
- **Alan filtreleri** (opsiyonel)
gibi seçimler istenir.

Yıl doğrulaması, mevcut verideki son yıldan sonra olacak şekilde anında yapılır ve hatalı girişlerde program kapanmadan tekrar sorulur.

---

## 📊 Örnek Çıktı

- **Terminal Çıktısı**
```
Seçilen Alan: Kimya
Tahmin Yılı: 2026
Tahmin Edilen Yayın Sayısı (Prophet): 125
Tahmin Edilen Yayın Sayısı (Linear Regression): 119
```

- **Grafiksel Çıktı**
Görseller proje klasörüne `.png` dosyaları olarak kaydedilir (otomatik açma yok).  
Dosya isimleri seçilen alan adına göre oluşturulur.

---

## ⚠️ Notlar

- `.env` dosyası **kesinlikle** GitHub’a yüklenmemelidir.
- SQL sorguları `sorgu.sql` dosyasında saklanabilir ancak gizlilik için `.gitignore` içine eklenmiştir.
- Tahmin sonuçları kullanılan modele, verilerin güncelliğine ve parametrelere göre değişebilir.

---

## 📬 İletişim

Herhangi bir soru, öneri veya geri bildirim için bana şu e-posta adresinden ulaşabilirsiniz:  
**omrumguler35@gmail.com**

---

# Academic Publication Prediction and Analysis System

This project is an analysis system that **predicts future publication counts** using historical publication data for specific academic fields. The project offers different forecasting strategies by utilizing both **Facebook Prophet** and **Linear Regression (implemented with numpy.polyfit)** models.

---

## 📌 Features

- **SQL Server connection** for direct data retrieval from the database.
- Forecasting with **Prophet** and **Linear Regression (using numpy.polyfit)** models.
- **Annual publication count analyses.**
- **Field-based filtering and comparison.**
- User-friendly **interactive terminal menu.**
- **Graphical output** generation (Matplotlib & Seaborn).
- Flexible **forecast year and model selection.**

---

## 🛠 Technologies Used

- **Python** (3.8+)
- **pandas** – Data processing
- **numpy** – Mathematical operations, Linear Regression implemented with numpy.polyfit
- **matplotlib / seaborn** – Plotting and visualization
- **prophet** – Time series forecasting
- **SQLAlchemy** – Database connection
- **python-dotenv** – Environment variable management
- **SQL Server** – Data source

---

## 📂 Project Structure

```
cu_veri_analizi/
│
├── analiz.py          # Main Python code (data retrieval, analysis, and prediction)
├── .env               # Database connection info (hidden)
├── .gitignore         # Files excluded from Git tracking
├── sorgu.sql          # SQL query file (optional, ignored)
├── *.png              # Generated figures saved in the project folder
└── README.md          # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/<username>/cu-veri-analizi.git
cd cu-veri-analizi
```

2. Install the required Python packages (if requirements.txt is present):
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and enter your database information:
```
DB_HOST=server_address
DB_USER=username
DB_PASSWORD=password
DB_NAME=database_name
DB_DRIVER=ODBC Driver 18 for SQL Server
DB_ENCRYPT=Optional
DB_TRUST_SERVER_CERT=true
```
> Note: If your Mac has only driver 17 installed, set `DB_DRIVER=ODBC Driver 17 for SQL Server`.

---

## ▶️ Usage

To run the program:
```bash
python analiz.py
```

During execution, the user will be prompted for:
- **Model selection** (Prophet or Linear Regression)
- **Forecast year** (default 2026)
- **Field filters** (optional)
and similar options.

Year validation happens immediately (must be after the last available data year); on invalid input the program re-prompts without exiting.

---

## 📊 Example Output

- **Terminal Output**
```
Selected Field: Chemistry
Forecast Year: 2026
Predicted Publication Count (Prophet): 125
Predicted Publication Count (Linear Regression): 119
```

- **Graphical Output**
Figures are saved as `.png` files in the project folder (no auto-open).  
File names are generated based on the selected field name.

---

## ⚠️ Notes

- The `.env` file **must not** be uploaded to GitHub.
- SQL queries can be stored in the `sorgu.sql` file but are included in `.gitignore` for privacy.
- Prediction results may vary depending on the model used, data recency, and parameters.

---

## 📬 Contact

For any questions, suggestions, or feedback, feel free to contact me at:  
**omrumguler35@gmail.com**