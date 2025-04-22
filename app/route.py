# HTTP接口路由类
from fastapi import APIRouter
from pydantic import BaseModel
from .agent import AgentClass

class AskRequest(BaseModel):
    user_id: str
    question: str

class APIRouteClass:

    def __init__(self, prefix: str = ""):
        """ API Route 初始化函数，构造一个APIRoute实例  """
        self.router = APIRouter(prefix=prefix)
        self._add_routes()

    def _add_routes(self):
        """ 所有API接口都注册在_add_routes函数中 """

        @self.router.get("/")
        def read_root():
            return {"message": "Hello World"}
        
        @self.router.post("/chat")
        def chat(req: AskRequest):
            """ 当用户输入问题时，需要调用agent进行处理 """
             # 创建Agent实例
            agent = AgentClass(user_id=req.user_id) 
            # 调用Agent处理输入
            response = agent.run_agent(req.question)
            return response
        
    def get_router(self):
        """ 返回 APIRoute实例绑定的所有请求接口 """
        return self.router
