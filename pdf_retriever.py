
from langchain.tools import Tool
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import Ollama

def load_pdf_retriever():
    loader = DirectoryLoader("docs/")  # Load all PDFs in /docs
    docs = loader.load()

    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = Chroma(collection_name='test',embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

retriever = load_pdf_retriever()

pdf_tool = Tool(
    name="PDFResearch",
    func=lambda q: "\n".join([d.page_content for d in retriever.get_relevant_documents(q)]),
    description="Useful for answering questions based on local PDF documents."
)
