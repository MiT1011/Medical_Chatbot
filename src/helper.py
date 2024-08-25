from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

class HuggingFaceLLM:
    def __init__(self, model_name, api_key, max_length=200, temperature=0.7):
        self.generator = pipeline("text-generation", model=model_name, use_auth_token=api_key)
        self.max_length = max_length
        self.temperature = temperature

    def __call__(self, prompt):
        return self.generator(prompt, max_length=self.max_length, temperature=self.temperature)[0]["generated_text"]


# Extract Data from the PDF
def LoadPDF(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def TextSplit(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def DownloadEmbeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
