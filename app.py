import streamlit as st
import os


MY_API_KEY = "AIzaSyDKqVM_cGh3ceLzd4V-t58QhgodRnZZ4Yc"


os.environ["GOOGLE_API_KEY"] = MY_API_KEY


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


st.set_page_config(page_title="Fırat Üni Asistanı", layout="wide")
st.title("1. Hafta: Mevzuat RAG ")


with st.sidebar:
    st.header("⚙️ Ayarlar")
    uploaded_file = st.file_uploader("Bir PDF Yükle", type="pdf")

    st.markdown("---")
    chunk_size = st.slider("Parça Boyutu (Chunk Size)", 500, 2000, 1000)
    k_value = st.slider("Kaç Parça Getirilsin? (k)", 1, 10, 3)


    if st.button("Sohbeti Temizle"):
        st.session_state.clear()
        st.rerun()



def process_pdf(file, chunk_s):

    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    # Bölme işlemi
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_s,
        chunk_overlap=100
    )
    docs = splitter.split_documents(pages)
    return docs


def get_vectorstore(docs):

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(docs, embeddings)
    return db



if uploaded_file:


    if "db" not in st.session_state:
        with st.spinner("PDF taranıyor ve vektörlere çevriliyor..."):
            try:
                docs = process_pdf(uploaded_file, chunk_size)
                st.session_state.db = get_vectorstore(docs)
                st.success(f"İşlem Başarılı! Belge {len(docs)} parçaya ayrıldı.")
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")


    question = st.text_input("Sorunuzu buraya yazın:", placeholder="Örn: Mazeret sınavı hakkı nedir?")

    if question and "db" in st.session_state:

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


        prompt_template = """
        Aşağıdaki bağlamı kullanarak soruyu cevapla.
        Eğer cevap metinde yoksa "Bilgim yok" de, uydurma.
        Cevabı sade ve anlaşılır maddeler halinde yaz.

        Bağlam: {context}
        Soru: {question}

        Cevap:
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Zinciri Kur
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=st.session_state.db.as_retriever(search_kwargs={"k": k_value}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Cevabı Al
        with st.spinner("Yapay zeka düşünüyor..."):
            res = qa_chain.invoke({"query": question})

            st.markdown("### 🤖 Cevap:")
            st.write(res["result"])


            st.markdown("---")
            st.caption("🔍 Kullanılan Kaynak Parçalar:")
            for i, doc in enumerate(res["source_documents"]):
                with st.expander(f"Kaynak {i + 1} (Sayfa {doc.metadata.get('page', 0) + 1})"):
                    st.info(doc.page_content)

elif not uploaded_file:
    st.info("Sol taraftan bir PDF yükleyerek başlayın.")