import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ==========================================
# 🔑 API KEY AYARI
# ==========================================
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDKqVM_cGh3ceLzd4V-t58QhgodRnZZ4Yc"

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Fırat Üni Mevzuat Asistanı (Pro)", layout="wide", page_icon="🎓")
st.title("🎓 Fırat Üniversitesi Mevzuat Asistanı (Çoklu Kaynak)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Doküman Yönetimi")
    uploaded_files = st.file_uploader(
        "PDF Dosyalarını Yükle",
        type="pdf",
        accept_multiple_files=True
    )

    st.divider()
    st.header("⚙️ Parametreler")
    search_type = st.radio("Arama Algoritması", ["MMR (Çeşitlilik)", "Similarity (Benzerlik)"])
    chunk_size = st.slider("Parça Boyutu (Chunk Size)", 500, 2000, 1000)
    k_value = st.slider("Kaynak Sayısı (k)", 2, 10, 5)

    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.clear()
        st.rerun()


# --- FONKSİYONLAR ---
def process_pdfs(files, chunk_s):
    all_docs = []
    if not os.path.exists("temp_files"):
        os.makedirs("temp_files")

    status_bar = st.progress(0)

    for i, file in enumerate(files):
        safe_name = os.path.basename(file.name)
        file_path = os.path.join("temp_files", safe_name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Her sayfaya kaynak etiketi basıyoruz
        for page in pages:
            page.metadata["source"] = safe_name

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_s, chunk_overlap=200)
        docs = splitter.split_documents(pages)
        all_docs.extend(docs)

        # İlerleme çubuğunu güncelle
        status_bar.progress((i + 1) / len(files))

    status_bar.empty()
    return all_docs


def get_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(docs, embeddings)
    return db


# --- UYGULAMA AKIŞI ---
if uploaded_files:
    if "db" not in st.session_state:
        with st.spinner("Dokümanlar analiz ediliyor ve vektör veritabanı oluşturuluyor..."):
            try:
                docs = process_pdfs(uploaded_files, chunk_size)
                st.session_state.db = get_vectorstore(docs)
                st.success(f"✅ Hazır! Toplam {len(uploaded_files)} dosya ve {len(docs)} bilgi parçası işlendi.")
            except Exception as e:
                st.error(f"Hata oluştu: {e}")

    # Soru Alanı
    st.markdown("---")
    question = st.text_input("Sorunuzu yazın:",
                             placeholder="Örn: Lisans yönetmeliği ile yüksek lisans danışmanlığı arasındaki farklar neler?")

    if question and "db" in st.session_state:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

        prompt_template = """
        Sen uzman bir akademik asistansın. Birden fazla kaynağı tarayarak cevap veriyorsun.

        GÖREVLER:
        1. Soruyu aşağıdaki bağlama göre cevapla.
        2. Cevabın içinde hangi bilginin hangi dosyadan geldiğini belirtmeye çalış (Örn: "Yönetmelik X'e göre...").
        3. Cevabı madde madde ve anlaşılır şekilde düzenle.

        Bağlam: {context}
        Soru: {question}

        Cevap:
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        if search_type == "MMR (Çeşitlilik)":
            retriever = st.session_state.db.as_retriever(search_type="mmr", search_kwargs={"k": k_value, "fetch_k": 20})
        else:
            retriever = st.session_state.db.as_retriever(search_kwargs={"k": k_value})

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        with st.spinner("Yapay zeka kaynakları tarıyor..."):
            res = qa_chain.invoke({"query": question})

            st.markdown("### 🤖 Asistan Cevabı")
            st.write(res["result"])

            st.markdown("---")
            st.subheader("📚 Kullanılan Kaynaklar")

            for i, doc in enumerate(res["source_documents"]):
                source_name = doc.metadata.get("source", "Bilinmiyor")
                page_num = doc.metadata.get("page", 0) + 1

                # Farklı dosyaları farklı renklerle göstermek için basit bir ikon mantığı
                icon = "📄" if "1747" in source_name else "📑"

                with st.expander(f"{icon} Kaynak {i + 1}: {source_name} (Sayfa {page_num})"):
                    st.info(doc.page_content)

elif not uploaded_files:
    st.info("👋 Başlamak için lütfen sol menüden PDF dosyalarınızı yükleyin.")