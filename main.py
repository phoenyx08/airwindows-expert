import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

def get_ollama_model():
    llm = ChatOllama(model="gemma3:1b")  # or "mistral", "gemma", etc.
    return llm

def load_document(file):
    loader = TextLoader(file)
    loaded_document = loader.load()
    return loaded_document

def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def build_vector_database(chunks):
    embedding_model = embedding()

    vectordb = Chroma.from_documents(chunks, embedding_model)
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

def retriever_qa(file, query):
    llm = get_ollama_model()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever_obj,
                                    return_source_documents=False)
    response = qa.invoke(query)
    return response['result']

# Get the query from the user input
query = input("Please enter your query: ")

# Print the generated response
print(retriever_qa("./data/airwindows.txt", query))
