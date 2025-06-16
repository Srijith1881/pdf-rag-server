import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Globals
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
prompt = hub.pull("rlm/rag-prompt")
os.environ["GOOGLE_API_KEY"] = os.getenv("API")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
parser = StrOutputParser()

def create_retriever_from_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
    return vectorstore.as_retriever()

def build_rag_chain(retriever):
    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
