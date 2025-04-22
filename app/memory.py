# 缓存模块
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from .prompt import PromptClass
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 定义Redis连接URL
redis_url = os.environ.get("REDIS_URL")

class MemoryClass:

    def __init__(self,memory_key):
        """ 缓存初始化方法 """
        self.memory_key = memory_key
        self.memory = []

        # 定义llm
        self.chatmodel = ChatOpenAI(
            model=os.getenv("BASE_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=0
        )

    def summary_chain(self, store_message):
        """缓存信息总结"""
        try:
            SystemPrompt = PromptClass().SystemPrompt.format(who_you_are="")

            Moods = PromptClass().MOODS

            prompt = ChatPromptTemplate.from_messages([
                ("system", SystemPrompt + "\n这是一段你和用户的对话记忆，对其进行总结摘要，摘要使用第一人称'我'，并且提取其中的关键信息，以如下格式返回：\n总结摘要 | 过去对话关键信息\n例如 用户张三问候我好，我礼貌回复，然后他问我langchain的向量库信息，我回答了他今年的问题，然后他又问了比特币价格。|Langchain, 向量库,比特币价格"),
                ("user", "{input}")
            ])
            chain = prompt | self.chatmodel

            # 🚨 把store_message转成字符串
            if isinstance(store_message, list):
                input_text = "\n".join(
                    f"{type(msg).__name__}: {msg.content}" for msg in store_message
                )
            else:
                input_text = str(store_message)

            summary = chain.invoke({
                "input": input_text,
                "who_you_are": Moods["default"]["roloSet"]
            })
            return summary
        except KeyError as e:
            print(e)

    def get_memory(self, user_id: str):
        """ 根据user_id获取缓存信息 """
        try:
            chat_message_history = RedisChatMessageHistory(
                url=redis_url,
                session_id=user_id
            )
            return chat_message_history
            # 对超长的聊天记录进行摘要
            store_message = chat_message_history.messages
            if len(store_message) > 80:
                str_message = ""
                for message in store_message:
                     # 先判断message有没有content属性，而且内容非空
                    if hasattr(message, "content") and message.content:
                        str_message += f"{type(message).__name__}: {message.content}\n"
                summary = self.summary_chain(str_message)
                chat_message_history.clear()  # 清空原有的对话
                chat_message_history.add_message(summary)  # 保存总结
                return chat_message_history
            else:
                return chat_message_history
        except Exception as e:
            print(e)
            return None
        

    def set_memory(self,user_id : str):
        """ 向指定user_id中写入缓存 """
    
        # 根据user_id获取缓存信息
        chat_memory = self.get_memory(user_id=user_id)
        if chat_memory is None:
            # 创建一个默认的 RedisChatMessageHistory 实例
            chat_memory = RedisChatMessageHistory(url=redis_url, session_id=user_id)
        self.memory = ConversationBufferMemory(
            llm=self.chatmodel,
            human_prefix="user",
            ai_prefix="小羊",
            memory_key=self.memory_key,
            output_key="output",
            return_messages=True,
            max_token_limit=1000,
            chat_memory=chat_memory,
        )
        return self.memory