import google.generativeai as genai
import os

# --- ANAHTARINI BURAYA YAZ ---
MY_API_KEY = "AIzaSyDKqVM_cGh3ceLzd4V-t58QhgodRnZZ4Yc"

genai.configure(api_key=MY_API_KEY)

print("--- KULLANILABİLİR MODELLER LİSTENİZ ---")
try:
    available_models = []
    for m in genai.list_models():
        # Sadece mesaj üretme (generateContent) yeteneği olanları göster
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            available_models.append(m.name)

    if not available_models:
        print("HATA: Hiçbir modele erişim yok. API Anahtarı veya Bölge kısıtlaması olabilir.")

except Exception as e:
    print(f"BİR HATA OLUŞTU: {e}")