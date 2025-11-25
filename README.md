# 📄 Fırat Üniversitesi Mevzuat Asistanı (RAG Projesi - 1. Hafta)

Bu proje, **Tek Kaynaklı RAG (Retrieval-Augmented Generation)** mimarisini kullanarak, kullanıcıların yüklediği PDF dokümanları (özellikle mevzuat metinleri) ile sohbet etmesini sağlayan bir yapay zeka asistanıdır.

**Ders/Görev:** 1. Hafta — Tek Kaynaklı RAG Kurulumu

## 🚀 Projenin Amacı
Kullanıcı tarafından yüklenen uzun ve karmaşık PDF dosyalarını analiz etmek, kullanıcının sorularına belgeye dayalı, kaynak göstererek ve halüsinasyon (uydurma) yapmadan cevap vermek.

## ✨ Özellikler

* **PDF İşleme:** Yüklenen PDF dosyasını belirlenen karakter limitlerine (Chunk Size) göre parçalara ayırır.
* **Semantik Arama:** Sorulan soruyla en alakalı metin parçalarını **Vektör Veritabanı (ChromaDB)** üzerinden bulur.
* **Google Gemini Entegrasyonu:** En güncel **Gemini 2.5 Flash** modelini kullanarak akıcı ve Türkçe cevaplar üretir.
* **Kaynak Gösterimi:** Cevabın hangi sayfadan ve hangi parçadan alındığını şeffaf bir şekilde gösterir.
* **Ayarlanabilir Parametreler:**
    * `Chunk Size`: Metin parçalama boyutu.
    * `k Değeri`: Cevap için kaç parça metin kullanılacağı.
* **Kullanıcı Dostu Arayüz:** Streamlit ile geliştirilmiş modern ve hızlı arayüz.

## 🛠️ Kullanılan Teknolojiler

* **Dil:** Python 3.10
* **Arayüz:** Streamlit
* **Orkestrasyon:** LangChain (v0.2 Stable)
* **LLM & Embedding:** Google Gemini API (`gemini-2.5-flash`)
* **Veritabanı:** ChromaDB (Ephemeral/Bellek içi)

## 📸 Ekran Görüntüleri

### 1. Soru Cevap ve Kaynak Gösterimi
Kullanıcı "Mazeret sınavı hakkı nedir?" diye sorduğunda sistemin verdiği kaynaklı cevap:

![Soru Cevap Örneği](screenshots/1.hafta/ss1.png)

![Soru Cevap Örneği](screenshots/1.hafta/ss3.png)

![Soru Cevap Örneği](screenshots/1.hafta/ss2.png)

### 2. Ayarlar ve Doküman Yükleme
PDF yükleme alanı ve Chunk/k ayarları:

![Ayarlar Menüsü](screenshots/1.hafta/ayarlar.png)

## ⚙️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1.  **Gereksinimleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **API Anahtarını Ayarlayın:**
    `app.py` dosyası içerisine Google AI Studio'dan aldığınız API anahtarını ekleyin.

3.  **Uygulamayı Başlatın:**
    Terminalde şu komutu çalıştırın:
    ```bash
    streamlit run app.py
    ```

## 🧪 Test Edilen Senaryolar (1. Hafta)

* [x] PDF yükleme ve metin parçalama (Chunking).
* [x] Vektör veritabanına kayıt.
* [x] "Bağlamda yoksa uydurma" kuralının uygulanması.
* [x] Cevapların madde madde listelenmesi.
* [x] Cevabın dayandığı kaynakların (Sayfa no) gösterilmesi.

---


---
## 🧠 2. Hafta: Gelişmiş RAG (MMR & Prompt Engineering)

Bu aşamada sisteme "çeşitlilik" kazandırılmış ve cevap kalitesi artırılmıştır.

### ✨ Yapılan Geliştirmeler
1.  **MMR (Maximal Marginal Relevance) Entegrasyonu:**
    * Sadece en benzer kelimeleri değil, anlamsal olarak farklı noktaları da taraması sağlandı.
    * *Kanıt:* "Yönetmeliğin amacı" sorusunda Similarity yöntemi sadece 2. ve 6. sayfaya bakarken, MMR yöntemi 1., 6. ve 8. sayfalardan veri toplayarak daha kapsamlı bir özet çıkardı.

2.  **Gelişmiş Prompt Tasarımı:**
    * Modele *"Önce 1 cümlelik yönetici özeti çıkar, sonra detayları maddeleştir"* talimatı verildi.
    * Cevapların okunabilirliği ve profesyonelliği artırıldı.

3.  **Dinamik Parametreler:**
    * Arayüz üzerinden `Similarity` vs `MMR` geçişi yapabilme özelliği eklendi.

### 📸 Karşılaştırma Testleri

**Test 1: MMR Farkı**
Aynı soruya MMR (Çeşitlilik) modunda verilen cevap ve kaynakların geniş dağılımı:
![MMR Örneği](screenshots/2.hafta/ss4.png)

**Test 2: Dürüstlük Kontrolü (Hallucination Check)**
Metinde tam listesi olmayan bir soru sorulduğunda modelin "Bilmiyorum" deme yeteneği:
![Disiplin Sorusu](screenshots/2.hafta/ss1.png)



---
## 📚 3. Hafta: Çok Kaynaklı RAG (Multi-Source Retrieval)

Projenin final aşamasında sistem, birden fazla PDF dokümanını aynı anda işleyip, sorulara bu belgeleri sentezleyerek cevap verecek hale getirilmiştir.

### ✨ Eklenen Özellikler
1.  **Çoklu Dosya Yükleme:**
    * `st.file_uploader` güncellenerek `accept_multiple_files=True` yapıldı.
    * Kullanıcı aynı anda Lisans Yönetmeliği ve Lisansüstü Danışmanlık Yönergesi gibi farklı belgeleri yükleyebilir.

2.  **Metadata Tabanlı Kaynak Takibi:**
    * Her metin parçasına (chunk) `metadata["source"]` etiketi eklendi.
    * Bu sayede modelin hangi bilgiyi hangi dosyadan aldığı takip edilebilir hale geldi.

3.  **Sentez Cevaplama:**
    * Model, farklı dosyalardan gelen parçaları birleştirerek bütünleşik cevaplar üretir.

### 📸 Test Sonucu (Kanıt)
Aşağıdaki ekran görüntüsünde görüldüğü üzere, sistem tek bir soru için **iki farklı PDF dosyasından** (Kaynak 4 farklı, diğerleri farklı) parça getirmiş ve bunları kullanmıştır:

![Çoklu Kaynak Kanıtı](screenshots/3.hafta/ss1.png)