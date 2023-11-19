from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chat_models import ChatOpenAI
# from dotenv import load_dotenv
import os
import timeit
import sys
# load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY','4d84e809-0ebf-4b0d-af9f-e439ab6630e3')
PINECONE_API_ENV=os.environ.get('PINECONE_API_ENV', 'us-central1-gcp')

def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents

extracted_data=load_pdf_file(data='data/')

def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=text_split(extracted_data)
print("Length of Text Chunks", len(text_chunks))

def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

start = timeit.default_timer()
embeddings = download_hugging_face_embeddings()

query_result = embeddings.embed_query("Hello world")
print("Length", len(query_result))


#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="llama"
