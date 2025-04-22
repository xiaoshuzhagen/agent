# 程序主启动类
from fastapi import FastAPI
from app.route import APIRouteClass

app = FastAPI()

# 实力化APIRoute类
api_routes = APIRouteClass(prefix="/lambagent/api")  

# 把 api_routes 里定义的一组接口（API endpoints），注册到 FastAPI 应用（app）里
app.include_router(api_routes.get_router())
