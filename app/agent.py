# Agent模块
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import ConfigurableField
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

from .prompt import PromptClass
from .emotion import EmotionClass
from .memory import MemoryClass
from .storage import StorageClass
from .tools import search, get_info_from_local

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置 LLM 内存缓存（避免重复问题再次请求API）
set_llm_cache(InMemoryCache())

class AgentClass:

    def __init__(self, user_id: str):
        """初始化Agent"""
        self.user_id = user_id
        self.storage = StorageClass()

        # 主模型 + 回退模型
        fallback_llm = ChatDeepSeek(
            model=os.getenv("BACKUP_MODEL"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_API_BASE"),
        )

        self.chatmodel = ChatOpenAI(
            model=os.getenv("BASE_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        ).with_fallbacks([fallback_llm])

        # 工具列表
        self.tools = [search, get_info_from_local]

        # 情绪识别模块
        self.emotion = EmotionClass()

        # Memory 模块
        self.memory_key = os.getenv("MEMORY_KEY")

        self.memory = MemoryClass(
            memory_key=self.memory_key
        )

        # 初始化初始情绪（中性）
        self.feeling = {"feeling": "default", "score": 5}

        # Prompt模板（初始）
        self.prompt = PromptClass(
            memory_key=self.memory_key,
            feeling=self.feeling
        ).Prompt_Structure()

        # 构建Agent
        self.agent = create_tool_calling_agent(
            self.chatmodel,
            self.tools,
            self.prompt,
        )

        # 注意：这里只初始化 agent_chain，不绑定 memory
        self.agent_chain = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
        ).configurable_fields(
            memory=ConfigurableField(
                id="agent_memory",
                name="Agent Memory",
                description="Dynamic memory per user"
            )
        )

    def run_agent(self, input_text: str):
        """运行Agent处理输入"""
        try:
            # 情绪感知（更新feeling）
            
            self.feeling = self.emotion.Emotion_Sensing(input_text)

            # 根据新的情绪状态，更新Prompt（可缓存优化）
            self.prompt = PromptClass(
                memory_key=os.getenv("MEMORY_KEY"),
                feeling=self.feeling
            ).Prompt_Structure()

            # 每次调用时，根据当前user_id设置Memory
            user_memory = MemoryClass(
                memory_key=self.memory_key
            )

            # 运行链，传入动态Memory
            response = self.agent_chain.with_config({
                "agent_memory": self.memory.set_memory(user_id=self.user_id)
            }).invoke({
                "input": input_text
            })
            # response
            return {"output": response["output"]}
            
        except Exception as e:
            print(f"[Agent Error]: {str(e)}")
            return {"error": "Agent internal error", "detail": str(e)}