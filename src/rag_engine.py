# src/rag_engine.py
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def create_vector_store(text_splits):
    """
    将文本切片转化为向量并构建本地 FAISS 检索库
    """
    embeddings = OpenAIEmbeddings(model="BAAI/bge-m3")
    vector_store = FAISS.from_documents(text_splits, embeddings)
    return vector_store

def ask_ai(vector_store, user_question):
    """
    基于向量库检索上下文，并调用大语言模型回答问题
    """
    # 调用硅基流动平台的免费千问大模型
    llm = ChatOpenAI(model="Qwen/Qwen2.5-7B-Instruct", temperature=0.3)
    
    # 封装严格的 Prompt 模板防止大模型产生幻觉（满足作业 3.3 要求）
    system_prompt = (
        "你是一个专业、严谨的 AI 智能文档助手。\n"
        "请严格使用以下检索到的上下文来回答用户的问题。如果你从上下文中找不到答案，"
        "请直接诚实地回答：'抱歉，在上传的文档中没有找到相关内容。'，切勿编造或凭空想象。\n\n"
        "【上下文信息】:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 构建标准的 RAG 检索生成链
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # 检索最相关的 3 个文本块
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # 返回执行结果（包含 answer 和 context 来源）
    return rag_chain.invoke({"input": user_question})