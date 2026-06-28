# app.py
import os
import streamlit as st
import tempfile
from dotenv import load_dotenv

# 从 src 模块中安全导入业务逻辑
from src.document_utils import load_and_split_pdf
from src.rag_engine import create_vector_store, ask_ai

# 启动时自动加载本地 .env 环境配置
load_dotenv()

st.set_page_config(page_title="AI 智能文档助手", page_icon="", layout="wide")

st.title(" 个人知识库：AI智能文档问答助手")
st.caption("基于 RAG (检索增强生成) 技术的高效知识提取工具")

# 初始化 Streamlit 会话状态，持久化存储向量库
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# 布局：左侧边栏处理文件，右侧主界面问答
with st.sidebar:
    st.header(" 1. 文档管理中心")
    uploaded_file = st.file_uploader("请上传一个 PDF 格式的参考文档", type="pdf")
    
    if uploaded_file:
        # 避免重复处理相同文件
        if st.session_state.current_file != uploaded_file.name:
            if st.button("开始解析与向量化"):
                with st.spinner("AI 正在深度阅读并构建知识库索引..."):
                    # 建立临时文件以供 PyPDFLoader 读取
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    try:
                        # 执行后端解耦的核心逻辑
                        splits = load_and_split_pdf(tmp_path)
                        st.session_state.vector_store = create_vector_store(splits)
                        st.session_state.current_file = uploaded_file.name
                        st.success(f" 《{uploaded_file.name}》解析成功！知识库已就绪。")
                    except Exception as e:
                        st.error(f"解析失败: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
        else:
            st.info(f"ℹ 当前活动文档: {st.session_state.current_file}")

# 右侧主界面
st.header(" 2. 语义智能问答")
user_question = st.text_input("请输入您想要针对文档提出的问题：", placeholder="例如：请帮我总结一下这份报告的核心结论是什么？")

if user_question:
    if st.session_state.vector_store is None:
        st.warning(" 请先在左侧栏上传 PDF 并点击解析构建知识库！")
    else:
        with st.spinner(" AI 正在检索文档并组织语言..."):
            try:
                # 调用RAG引擎
                response = ask_ai(st.session_state.vector_store, user_question)
                
                # 打印AI回答
                st.markdown("###  AI 的深度回答：")
                st.info(response["answer"])
                
                # 可视化展示可追溯的原文片段
                with st.expander(" 查看 AI 本次回答参考的知识库原文片段"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**【参考片段 {i+1}】 (来自原书/文档第 {doc.metadata.get('page', 0) + 1} 页):**")
                        st.caption(doc.page_content)
                        st.markdown("---")
            except Exception as e:
                st.error(f"请求失败，请检查网络或 API 配置。错误信息: {e}")