from flask import Flask, render_template, jsonify, request
from src.helper import DownloadEmbeddings
from langchain_community.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

os.environ['PINECONE_API_KEY'] = os.environ.get('PINECONE_API_KEY')

embeddings = DownloadEmbeddings()
index_name = "medical-chatbot"

docsearch = Pinecone.from_existing_index(index_name, embeddings)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

llm = CTransformers(model=r"model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                    # model=r"model\llama-2-7b-chat.ggmlv3.q2_K.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512,
                            'temperature': 0.8,})

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwar={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Default Route of the flask
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
# msg = "what is Therapeutic abortion?"
    print(msg)
    result=qa.invoke({"query":msg})
    print("Response : ", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(debug=True)