from src.helper import LoadPDF, TextSplit, DownloadEmbeddings
from langchain_pinecone import PineconeVectorStore
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['PINECONE_API_KEY'] = os.environ.get('PINECONE_API_KEY')
index_name = "medical-chatbot"

extracted_data = LoadPDF("data/")
text_chunks = TextSplit(extracted_data)
embeddings = DownloadEmbeddings()

vectorstore_from_docs = PineconeVectorStore.from_documents(
    text_chunks,
    index_name=index_name,
    embedding=embeddings
)