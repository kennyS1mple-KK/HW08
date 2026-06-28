# src/document_utils.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdf(file_path):
    """
    读取 PDF 文件并将其切分为适合大模型阅读的文本块
    """
    # 1. 加载 PDF 文本内容
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # 2. 文本切分：设定块大小为 1000 字，重叠 200 字以保持上下文连续性
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    return splits