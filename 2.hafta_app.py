import streamlit as st
import os

# ==========================================
# 🔑 ANAHTARI BURAYA YAPIŞTIR
# ==========================================
MY_API_KEY = "AIzaSyDKqVM_cGh3ceLzd4V-t58QhgodRnZZ4Yc"
os.environ["GOOGLE_API_KEY"] = MY_API_KEY

# --- IMPORTLAR ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Mevzuat Asistanı v2", layout="wide")
st.title("🧠 2. Hafta: Gelişmiş RAG (MMR & Prompt Ayarı)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Gelişmiş Ayarlar")
    uploaded_file = st.file_uploader("PDF Yükle", type="pdf")

    st.divider()

    # 2. HAFTA YENİLİĞİ: Arama Tipi Seçimi
    search_type = st.radio(
        "Arama Yöntemi (Retriever)",
        ["Similarity (Benzerlik)", "MMR (Çeşitlilik)"]
    )

    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    k_value = st.slider("k Değeri (Parça Sayısı)", 1, 10, 3)

    if st.button("Sohbeti Temizle"):
        st.session_state.clear()
        st.rerun()


# --- FONKSİYONLAR ---
def process_pdf(file, chunk_s):
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_s,
        chunk_overlap=150  # Overlap'i biraz artırdık, bağlam kopmasın diye
    )
    docs = splitter.split_documents(pages)
    return docs


def get_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(docs, embeddings)
    return db


# --- UYGULAMA AKIŞI ---
if uploaded_file:

    # PDF İşleme
    if "db" not in st.session_state:
        with st.spinner("PDF Analiz Ediliyor..."):
            try:
                docs = process_pdf(uploaded_file, chunk_size)
                st.session_state.db = get_vectorstore(docs)
                st.success(f"İşlem Tamam! {len(docs)} parça oluşturuldu.")
            except Exception as e:
                st.error(f"Hata: {e}")

    # Soru Sorma
    question = st.text_input("Sorunuz:", placeholder="Örn: Disiplin cezasına itiraz süresi nedir?")

    if question and "db" in st.session_state:
        # LLM (En güçlü model)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

        # 2. HAFTA YENİLİĞİ: Gelişmiş Prompt
        prompt_template = """
        Sen uzman bir üniversite mevzuat asistanısın.
        Aşağıdaki bağlamı kullanarak soruyu cevapla.

        KURALLAR:
        1. Önce cevabın 1 cümlelik net bir özetini yaz.
        2. Ardından detayları madde madde sırala.
        3. Eğer cevap metinde yoksa "Bilgim yok" de, asla uydurma.

        Bağlam: {context}
        Soru: {question}

        Cevap:
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # 2. HAFTA YENİLİĞİ: Retriever Mantığı
        if search_type == "MMR (Çeşitlilik)":
            # MMR: Benzer ama birbirinden farklı parçaları getirir
            retriever = st.session_state.db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k_value, "fetch_k": 20, "lambda_mult": 0.5}
            )
        else:
            # Similarity: Sadece en çok benzeyenleri getirir
            retriever = st.session_state.db.as_retriever(
                search_kwargs={"k": k_value}
            )

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        with st.spinner("Analiz ediliyor..."):
            res = qa_chain.invoke({"query": question})

            st.markdown("### 🤖 Cevap:")
            st.write(res["result"])

            st.divider()
            st.caption(f"Kullanılan Yöntem: **{search_type}** | Parça Sayısı: **{k_value}**")
            for i, doc in enumerate(res["source_documents"]):
                with st.expander(f"Kaynak {i + 1} (Sayfa {doc.metadata.get('page', 0) + 1})"):
                    st.write(doc.page_content)

elif not uploaded_file:
    st.info("Başlamak için PDF yükleyin.")