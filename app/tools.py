# Agent 工具模块
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.agents import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# 导入预置chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import PydanticOutputParser
import os

class Config:
    """ 配置信息 """

    def __init__(self):
        """ 初始化，载入环境变量 """
        load_dotenv()
        self.setup_environment()

    @staticmethod
    def setup_environment():
        required_vars = [
            "SERPAPI_API_KEY",
            "OPENAI_API_KEY",
            "OPENAI_API_BASE"
        ]
        
        # 环境变量检查，如果环境变量不存在，则抛出异常
        for var in required_vars:
            if not os.getenv(var):
                raise EnvironmentError(f"Missing required environment variable: {var}")
            
        os.environ.update({
            "SERPAPI_API_KEY": os.getenv("SERPAPI_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE")
        })

@tool
def search(query:str) -> str:
    """只有需要了解实时信息或不知道的事情的时候才会使用这个工具."""
    serp = SerpAPIWrapper()
    return serp.run(query)

@tool
def get_info_from_local(query:str) -> str:
    """从本地知识库获取信息。

    Args:
        query (str): 用户的查询问题

    Returns:
        str: 从知识库中检索到的答案
    """
    # 获取用户ID
    userid = get_user("userid")

    # 定义llm
    llm = ChatOpenAI(
            model=os.getenv("BASE_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=0
    )

    # 定义缓存
    memory = MemoryClass(
        memory_key=os.getenv("MEMORY_KEY"),
        model=os.getenv("BASE_MODEL")
    )

    # 根据用户ID取出聊天历史记录
    chat_history = memory.get_memory(session_id=userid).messages if userid else []
    
    # 定义提示词
    condense_question_prompt = ChatPromptTemplate.from_messages([
        ("system", "给出聊天记录和最新的用户问题。可能会引用聊天记录中的上下文，提出一个可以理解的独立问题。没有聊天记录，请勿回答。必要时重新配制，否则原样退还。"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    # 连接 Qdrant 向量数据库
    client = QdrantClient(
        path=os.getenv("PERSIST_DIR","./vector_store")
    )

    # 创建一个向量存储对象（Vector Store）
    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=os.getenv("EMBEDDING_COLLECTION"), 
        embedding=OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
            api_key=os.getenv("EMBEDDING_API_KEY"),
            base_url=os.getenv("EMBEDDING_API_BASE")
        )
    )

    # 构建检索器，使用MMR进行相似性检索
    retriever = vector_store.as_retriever(
        # 不只考虑“相似度”，还考虑**“多样性”**
        search_type="mmr",
        search_kwargs={
            # 最多返回5条数据
            "k": 5, 
            "fetch_k": 10
        }
    )

    # 定义查询重写检索chain
    # 所以流程大概是这样：
	#   1.用户发来一个问题 input
	#   2.retriever 组件用问题去向量数据库检索，拿到一段相关的 context
	#   3.retriever 把 context 自动填到 Prompt 里的 {context} 这个位置
	#   4.整个 chain 把补充好的 prompt 发给大语言模型 LLM 去生成答案
    qa_chain = create_retrieval_chain(
        create_history_aware_retriever(
            llm, 
            retriever, 
            # 给 retriever 的 prompt
            # 重写问题，根据历史聊天记录把用户的问题改写成完整的检索查询
            condense_question_prompt),
        create_stuff_documents_chain(
            llm,
            ChatPromptTemplate.from_messages([
                ("system", "你是回答问题的助手。使用下列检索到的上下文回答这个问题。如果你不知道答案，就说你不知道。最多使用三句话，并保持回答简明扼要。\n\n{context}"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
            ])
        )
    )

    # 执行向量检索
    response = qa_chain.invoke({
        "input": query,
        "chat_history": chat_history,
    })

    return response["answer"]

# 实例化配置
Config()