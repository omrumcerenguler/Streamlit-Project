# 📊 Çukurova Üniversitesi Yayın Analizi — Streamlit

## Amaç
Bu proje, Çukurova Üniversitesi akademik yayın verilerini analiz etmek ve görselleştirmek amacıyla geliştirilmiştir. Kullanıcıların verileri interaktif olarak inceleyebilmesi, yayın trendlerini takip edebilmesi ve alan/yazar bazında özet istatistikler çıkarabilmesi hedeflenmiştir.

## 🚀 Kurulum
1. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
2. Uygulamayı çalıştırın:
   ```bash
   streamlit run streamlitapp.py
   ```

## 🧭 Kullanım
Açılan tarayıcı arayüzünde filtreleri kullanarak yıllara, alanlara veya yazarlara göre verileri süzebilir; farklı görünümler arasında geçiş yapabilirsiniz.

## 📂 Dosya Yapısı
- `streamlitapp.py` — Streamlit uygulama dosyası
- `requirements.txt` — Gerekli kütüphaneler
- `.gitignore` — Gereksiz/gizli dosyaların takibini engeller

## 🎯 Özellikler
- Yıllara göre toplam yayın sayısı
- En çok yayın yapan yazarlar
- En çok atıf alan yazarlar
- En fazla yayın yapılan alanlar

## 🧰 Kullanılan Teknolojiler
- Streamlit — İnteraktif web arayüzü
- pandas — Veri işleme
- matplotlib / plotly — Görselleştirme
- SQL Server — Veri kaynağı

## 📌 Notlar
- Bu projede örnek veri seti bulunmamaktadır.
- Tüm veriler SQL Server üzerinden alınmaktadır.

## 🔧 Gelecek Geliştirmeler
- Kullanıcı girişi ve yetkilendirme
- Daha fazla görselleştirme seçeneği
- Otomatik rapor oluşturma/indirme

## 🤝 Katkı
Katkıda bulunmak isterseniz bir **issue** açıp önerinizi yazabilir, ardından **fork → pull request** akışıyla değişiklik önerebilirsiniz.

---

# 📊 Cukurova University Publication Analysis — Streamlit

## Purpose
This project analyzes and visualizes Cukurova University’s academic publication data. It enables users to interactively explore trends and derive summary statistics by field/author.

## 🚀 Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run streamlitapp.py
   ```

## 🧭 Usage
Use the filters in the web UI to slice data by year, field, or author, and switch between different views.

## 📂 File Structure
- `streamlitapp.py` — Streamlit application
- `requirements.txt` — Dependencies
- `.gitignore` — Excludes unnecessary/secret files

## 🎯 Features
- Total publications by year
- Top publishing authors
- Most cited authors
- Fields with the highest number of publications

## 🧰 Technologies
- Streamlit — Interactive UI
- pandas — Data wrangling
- matplotlib / plotly — Visualization
- SQL Server — Data source

## 📌 Notes
- There is no sample dataset included in this repository.
- All data is fetched directly from SQL Server.

## 🔧 Future Improvements
- Authentication & authorization
- Additional visualization options
- Automated report generation/download

## 🤝 Contributing
Open an **issue** to discuss changes you’d like to make, then propose them via **fork → pull request**.