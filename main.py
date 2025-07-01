import os

from langchain_core.prompts import PromptTemplate

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
import gradio as gr

VECTOR_DB_DIR = "./vector_db"

def get_ollama_model():
    llm = ChatOllama(model="gemma3:1b")  # or "mistral", "gemma", etc.
    return llm

def load_document(file):
    loader = TextLoader(file)
    loaded_document = loader.load()
    return loaded_document

def split_text(data):
    text_splitter = CharacterTextSplitter(
        separator="############",
        chunk_size=8751,
        chunk_overlap=0,
        length_function=len,
    )
    largerChunks = text_splitter.split_documents(data)

    chunks = []
    fine_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
    )
    for chunk in largerChunks:
        if len(chunk.page_content) > 1000:
            chunks.extend(fine_splitter.split_documents([chunk]))
        else:
            chunks.append(chunk)
    return chunks

def build_vector_database(chunks, persist_directory=VECTOR_DB_DIR):
    embedding_model = embedding()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    # Force persistence by calling .persist() only if available
    if hasattr(vectordb, "persist"):
        vectordb.persist()
    return vectordb

def get_vector_db(file_path, persist_directory=VECTOR_DB_DIR):
    if os.path.exists(persist_directory):
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding())
    else:
        splits = load_document(file_path)
        chunks = split_text(splits)
        vectordb = build_vector_database(chunks, persist_directory)
    return vectordb

def embedding():
    embedding_model = HuggingFaceEmbeddings()
    return embedding_model

def retriever(file):
    splits = load_document(file)
    chunks = split_text(splits)
    vectordb = build_vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

def retriever_qa(_, query):
    response = qa_chain.invoke(query)
    return response['result']

# Global initialization
retriever_obj = get_vector_db("./data/airwindows.txt").as_retriever()
llm = get_ollama_model()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever_obj)

# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        # gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),  # Drag and drop file upload
        gr.Text(value="./data/airwindows.txt", visible=False),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Markdown(label="Output"),
    title="Airwindows Plugins Expert",
    description="Ask any question related to Airwindows Plugins. The answer is based on Airwindopedia's content."
)

# Launch the app
rag_application.launch(server_name="0.0.0.0", server_port= 7860)
