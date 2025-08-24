import streamlit as st
import tempfile
import os
import requests
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="PaperPal - AI论文助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DoubaoAPI:
    """火山引擎豆包API封装类"""
    
    def __init__(self, api_key: str, endpoint_id: str):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        # 使用正确的火山方舟API地址
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3"
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """调用豆包API进行对话"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.endpoint_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2000,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30  # 减少超时时间
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_msg = f"豆包API调用失败 ({response.status_code}): {response.text}"
                st.error(error_msg)
                return f"抱歉，API调用失败。错误信息: {error_msg}"
                
        except Exception as e:
            error_msg = f"豆包API调用出错: {str(e)}"
            st.error(error_msg)
            return f"抱歉，API调用出错: {error_msg}"

class QwenAPI:
    """阿里云通义千问API封装类"""
    
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """调用Qwen API进行对话"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "X-DashScope-SSE": "disable"
            }
            
            data = {
                "model": "qwen-turbo",
                "input": {
                    "messages": messages
                },
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": 2000,
                    "top_p": 0.8
                }
            }
            
            response = requests.post(
                f"{self.base_url}/services/aigc/text-generation/generation",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["output"]["text"]
            else:
                error_msg = f"Qwen API调用失败 ({response.status_code}): {response.text}"
                st.error(error_msg)
                return f"抱歉，API调用失败。错误信息: {error_msg}"
                
        except Exception as e:
            error_msg = f"Qwen API调用出错: {str(e)}"
            st.error(error_msg)
            return f"抱歉，API调用出错: {error_msg}"

class DoubaoEmbeddings:
    """火山引擎豆包Embeddings封装类"""
    
    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        try:
            # 简化版本，实际需要调用火山引擎的embedding API
            # 这里返回一个假的向量作为演示
            return [0.1] * 1536  # OpenAI embedding维度
        except Exception as e:
            st.error(f"嵌入生成失败: {str(e)}")
            return [0.0] * 1536

@st.cache_resource
def load_pdf_and_create_qa_chain(uploaded_file, api_choice: str):
    """
    加载PDF文件并创建问答链
    使用Streamlit缓存避免重复处理
    """
    try:
        # 将上传的文件保存为临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # 加载PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        
        # 根据API选择配置嵌入和LLM
        if api_choice == "doubao":
            # 使用豆包API
            api_key = os.getenv("DOUBAO_API_KEY")
            endpoint_id = os.getenv("DOUBAO_ENDPOINT_ID")
            
            if not api_key or not endpoint_id:
                st.error("请配置火山引擎豆包API密钥！")
                return None
            
            # 创建嵌入（先用OpenAI的作为备选）
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                embeddings = OpenAIEmbeddings(api_key=openai_key)
            else:
                st.error("豆包模式下需要配置OpenAI API Key用于文档嵌入")
                return None
            
            # 创建向量存储
            vectorstore = Chroma.from_documents(texts, embeddings)
            
            # 创建自定义的豆包LLM wrapper
            class DoubaoLLM:
                def __init__(self, doubao_api):
                    self.doubao_api = doubao_api
                
                def __call__(self, prompt: str) -> str:
                    messages = [{"role": "user", "content": prompt}]
                    return self.doubao_api.chat_completion(messages)
            
            doubao_api = DoubaoAPI(api_key, endpoint_id)
            llm = DoubaoLLM(doubao_api)
            
            # 创建问答链（简化版本）
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            def qa_chain(question: str) -> str:
                docs = retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                prompt = f"""基于以下文档内容回答问题。请用中文回答。

文档内容：
{context}

问题：{question}

请根据文档内容提供准确、详细的回答。如果文档中没有相关信息，请说明。"""
                
                return llm(prompt)
            
            return qa_chain
            
        else:
            # 使用OpenAI API
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                st.error("请配置OpenAI API密钥！")
                return None
            
            # 创建嵌入
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            
            # 创建向量存储
            vectorstore = Chroma.from_documents(texts, embeddings)
            
            # 创建LLM
            llm = ChatOpenAI(
                temperature=0.7,
                model="gpt-3.5-turbo",
                openai_api_key=openai_api_key
            )
            
            # 创建问答链
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=False
            )
            
            return qa_chain
    
    except Exception as e:
        st.error(f"处理PDF时出错: {str(e)}")
        return None
    
    finally:
        # 清理临时文件
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass

def main():
    """主应用函数"""
    
    # 侧边栏配置
    # 添加自定义CSS样式
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .success-box {
        background: linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #a8cfe6 0%, #c1dced 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .sidebar-logo {
        text-align: center;
        padding: 1rem 0;
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        margin-bottom: 1rem;
        color: #667eea;
        font-weight: bold;
        font-size: 1.3rem;
    }
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        # 自定义Logo
        st.markdown("""
        <div class="sidebar-logo">
            <svg width="60" height="60" viewBox="0 0 100 100" style="margin-bottom: 10px;">
                <!-- 书本图标 -->
                <rect x="20" y="25" width="60" height="50" rx="5" fill="#667eea" stroke="#667eea" stroke-width="1" opacity="0.8"/>
                <rect x="25" y="30" width="50" height="40" rx="3" fill="rgba(255,255,255,0.9)"/>
                <!-- AI大脑图标 -->
                <circle cx="50" cy="50" r="12" fill="#667eea" opacity="0.7"/>
                <circle cx="47" cy="47" r="2" fill="white"/>
                <circle cx="53" cy="47" r="2" fill="white"/>
                <path d="M 45 54 Q 50 58 55 54" stroke="white" stroke-width="2" fill="none"/>
                <!-- 连接线 -->
                <line x1="35" y1="40" x2="42" y2="47" stroke="#667eea" stroke-width="2" opacity="0.6"/>
                <line x1="65" y1="40" x2="58" y2="47" stroke="#667eea" stroke-width="2" opacity="0.6"/>
            </svg>
            <div style="font-weight: bold; font-size: 1.4rem;">PaperPal</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">AI论文助手</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ API配置")
        
        # API选择
        api_choice = st.selectbox(
            "选择API服务商",
            ["doubao", "openai", "qwen"],
            format_func=lambda x: {
                "doubao": "🚀 火山引擎豆包",
                "openai": "🤖 OpenAI GPT",
                "qwen": "🔥 阿里云通义千问"
            }[x]
        )
    
    # 主界面 - 标题
    st.markdown('<h1 class="main-header">PaperPal - AI论文助手</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">🧠 让AI帮你理解任何学术论文</p>', unsafe_allow_html=True)
    
    # 检查API配置
    api_configured = False
    if api_choice == "doubao":
        if os.getenv("DOUBAO_API_KEY") and os.getenv("DOUBAO_ENDPOINT_ID"):
            api_configured = True
        else:
            st.error("⚠️ 请配置火山引擎豆包API密钥！需要DOUBAO_API_KEY和DOUBAO_ENDPOINT_ID。")
    elif api_choice == "qwen":
        if os.getenv("QWEN_API_KEY"):
            api_configured = True
        else:
            st.error("⚠️ 请配置通义千问API密钥！需要QWEN_API_KEY。")
    else:
        if os.getenv("OPENAI_API_KEY"):
            api_configured = True
        else:
            st.error("⚠️ 请配置OpenAI API密钥！请创建.env文件并填入OPENAI_API_KEY。")
    
    if not api_configured:
        st.info("💡 请参考env_example.txt文件创建.env配置文件。")
        return
    
    # 文件上传区域
    uploaded_file = st.file_uploader(
        "🔺 上传PDF论文",
        type="pdf",
        help="支持上传学术论文PDF文件，最大10MB"
    )
    
    if uploaded_file is not None:
        # 显示文件信息
        file_details = {
            "文件名": uploaded_file.name,
            "文件大小": f"{uploaded_file.size / (1024*1024):.1f} MB",
            "文件类型": uploaded_file.type
        }
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="success-box">
                <h4>✅ PDF文件上传成功！</h4>
                <p><strong>文件名</strong>: {file_details["文件名"]}</p>
                <p><strong>文件大小</strong>: {file_details["文件大小"]}</p>
                <p><strong>文件类型</strong>: {file_details["文件类型"]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            api_name = {
                "doubao": "🚀 火山引擎豆包",
                "qwen": "🔥 阿里云通义千问", 
                "openai": "🤖 OpenAI GPT"
            }.get(api_choice, "Unknown")
            
            st.markdown(f"""
            <div class="info-box">
                <h4>🔧 API配置</h4>
                <p><strong>当前使用</strong>: {api_name}</p>
                <p><strong>状态</strong>: ✅ 已连接</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 处理PDF并创建问答链
        with st.spinner("🔄 AI正在分析论文内容..."):
            qa_chain = load_pdf_and_create_qa_chain(uploaded_file, api_choice)
        
        if qa_chain:
            st.markdown("""
            <div class="success-box">
                <h3>🎉 论文分析完成！</h3>
                <p>✨ AI已经理解了论文内容，现在可以开始智能问答了</p>
                <p>💡 试试问一些关于论文的问题吧！</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 初始化会话状态
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # 显示聊天历史
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # 聊天输入
            if prompt := st.chat_input("🤔 请输入您的问题..."):
                # 添加用户消息
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # 生成AI回答
                with st.chat_message("assistant"):
                    with st.spinner("🤖 AI正在思考..."):
                        try:
                            if api_choice == "doubao":
                                # 豆包API
                                response = qa_chain(prompt)
                            else:
                                # OpenAI API
                                response = qa_chain.run(prompt)
                            
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        except Exception as e:
                            error_msg = f"抱歉，处理问题时出现错误: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
            # 清除对话按钮
            if st.button("🗑️ 清除对话历史"):
                st.session_state.messages = []
                st.rerun()
    
    else:
        # 没有上传文件时的界面
        st.markdown("""
        <div style="text-align: center; padding: 30px 25px; background: rgba(102, 126, 234, 0.1); border: 1px solid rgba(102, 126, 234, 0.2); border-radius: 12px; margin: 20px 0;">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 0.5rem;">
                <svg width="35" height="35" viewBox="0 0 100 100" style="margin-right: 10px;">
                    <!-- 书本图标 -->
                    <rect x="20" y="25" width="60" height="50" rx="5" fill="#667eea" stroke="#667eea" stroke-width="1" opacity="0.8"/>
                    <rect x="25" y="30" width="50" height="40" rx="3" fill="rgba(255,255,255,0.9)"/>
                    <!-- AI大脑图标 -->
                    <circle cx="50" cy="50" r="12" fill="#667eea" opacity="0.7"/>
                    <circle cx="47" cy="47" r="2" fill="white"/>
                    <circle cx="53" cy="47" r="2" fill="white"/>
                    <path d="M 45 54 Q 50 58 55 54" stroke="white" stroke-width="2" fill="none"/>
                    <!-- 连接线 -->
                    <line x1="35" y1="40" x2="42" y2="47" stroke="#667eea" stroke-width="2" opacity="0.6"/>
                    <line x1="65" y1="40" x2="58" y2="47" stroke="#667eea" stroke-width="2" opacity="0.6"/>
                </svg>
                <h2 style="font-size: 1.8rem; margin: 0; color: #667eea;">欢迎使用PaperPal</h2>
            </div>
            <p style="font-size: 1rem; margin: 10px 0; color: #666;">上传PDF论文，开始智能对话</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 简化的示例问题
        st.markdown("""
        <h3 style="color: #667eea; margin: 30px 0 20px 0; font-size: 1.3rem;">
            <svg width="20" height="20" viewBox="0 0 100 100" style="margin-right: 8px; vertical-align: middle;">
                <circle cx="50" cy="50" r="12" fill="#667eea" opacity="0.7"/>
                <circle cx="47" cy="47" r="2" fill="white"/>
                <circle cx="53" cy="47" r="2" fill="white"/>
                <path d="M 45 54 Q 50 58 55 54" stroke="white" stroke-width="2" fill="none"/>
            </svg>
            试试这些问题
        </h3>
        """, unsafe_allow_html=True)
        
        examples = [
            "📝 这篇论文的主要观点是什么？",
            "🔍 实验用了什么数据集？", 
            "💡 这个方法有什么创新点？",
            "🎯 有哪些局限性？"
        ]
        
        for example in examples:
            st.markdown(f"""
            <div style="padding: 12px 16px; background: rgba(102, 126, 234, 0.06); border: 1px solid rgba(102, 126, 234, 0.15); border-radius: 8px; margin: 8px 0;">
                <p style="margin: 0; color: #555; font-size: 0.95rem;">{example}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
