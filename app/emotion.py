# 情感侦测模块
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

class EmotionClass:
    """ 用户情感分析模块 """

    def __init__(self,):
        """ EmotionClass 初始化方法 """
        
        self.chat = None

        # 构造llm
        self.chat_model = ChatOpenAI(
            model=os.getenv("BASE_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=0
        )

    def Emotion_Sensing(self,input:str):
        """ 执行情感分析 """
        # 1.处理用户输入词长度
        original_input = input
        if len(input) > 1000:
            input = input[1000]
            print(f"Input is too long,only the first 1000 characters will be used,Original length : {len(original_input)}")

        # 定义JSON schema
        json_schema = {
            "title": "emotions",
            "description": "emotion analysis with feeling type and negativity score",
            "type": "object",
            "properties": {
                "feeling": {
                    "type": "string",
                    "description": "the emotional state detected in the input",
                    "enum": [
                        "default", "upbeat", "angry", 
                        "cheerful", "depressed", "friendly"
                    ]
                },
                "score": {
                    "type": "string",
                    "description": "negativity score from 1 to 10, where 10 represents extremely negative emotions",
                    "enum": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
                }
            },
            "required": ["feeling", "score"]
        }

        # 定义提示词 - 从 langsmith上拉取
        prompt_emotion = """
        分析用户输入的文本情绪，返回情绪类型和负面程度评分。

        评分规则：
        - 分数范围为1-10
        - 分数越高表示情绪越负面
        - 1-3分：积极正面的情绪
        - 4-5分：中性或轻微情绪波动
        - 6-8分：明显的负面情绪
        - 9-10分：强烈的负面情绪

        情绪类型对照：
        - default: 中性、平静的情绪状态
        - upbeat: 积极向上、充满活力的情绪
        - angry: 愤怒、生气的情绪
        - cheerful: 开心愉快、充满欢乐的情绪
        - depressed: 沮丧、压抑的情绪
        - friendly: 友好、亲切的情绪

        情绪分类指南：
        1. default: 用于表达中性或普通的情绪状态
        2. upbeat: 用于表达积极向上、充满干劲的状态
        3. angry: 用于表达愤怒、不满、生气的情绪
        4. cheerful: 用于表达欢快、喜悦的情绪
        5. depressed: 用于表达消极、低落、压抑的情绪
        6. friendly: 用于表达友善、亲切的情绪

        示例：
        - "我特别生气！" -> {{"feeling": "angry", "score": "8"}}
        - "今天天气真好" -> {{"feeling": "cheerful", "score": "2"}}
        - "随便吧，都可以" -> {{"feeling": "default", "score": "5"}}
        - "我很难过" -> {{"feeling": "depressed", "score": "9"}}
        - "谢谢你的帮助" -> {{"feeling": "friendly", "score": "1"}}

        用户输入内容: {input}
        请根据以上规则分析情绪并返回相应的feeling和score。

        """
        # 定义一个标准输出格式的llm
        llm = self.chat_model.with_structured_output(json_schema)

        # 情绪分析链
        
        EmotionChain = ChatPromptTemplate.from_messages([("system", prompt_emotion), ("user", input)]) | llm
    
        try:
            if not input.strip():
                print("Empty input received")
                return None
            
            if EmotionChain is not None:
                
                result = EmotionChain.invoke({"input": input})

            else:
                raise ValueError("EmotionChain is not properly instantiated.")
            
            return result
        except Exception as e:
            print(f"Error in Emotion_Sensing: {str(e)}")
            return {"feeling": "default", "score": "5"}