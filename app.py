from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import DirectoryLoader, FireCrawlLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.docstore.document import Document
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import torch
import os
langsmith_api = "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"
local_llm = 'llama3'


app = Flask(__name__)
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = langsmith_api

llm = ChatOllama(model=local_llm, temperature=0)

def load_documents(PATH):
    loader = DirectoryLoader(PATH, glob="*.md")
    documents = loader.load()
    return documents

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

alic_doc = load_documents("db")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(alic_doc)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
)
retriever = vectorstore.as_retriever()
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. You always answer in casual and friendly way. 
    You only answer in {language}.. If you don't know the answer, Use the following pieces of retrieved context to answer the question. 
    Lookup from your knowledge without using give context first.Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

rag_chain = prompt | llm | StrOutputParser()

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_chat_response(input)

def get_chat_response(text):
    question = str(text)
    docs = retriever.invoke(question)
    language = "Thai"
    generation = rag_chain.invoke({"context": docs, "question": question, "language":language})
    return generation

if __name__ == "__main__":
    app.run(debug=True)
