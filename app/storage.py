# 全局用户存储

class StorageClass:

    def __init__(self):
        """ 存储初始化类 """
        self.user_storage = {
            # 用户ID: 用户信息
            1000010: {
                "name": "张三",
                "age": 25,
                "user_id": 1,
                "user_name": "张三",
                "user_age": 25,
            }
        }

    def add_user(self,user_id, user_data):
        """ 添加用户信息 """
        self.user_storage[user_id] = user_data

    def get_user(self,user_id):
        """ 通过用户ID获取用户信息 """
        return self.user_storage.get(user_id) 

    def get_all_users(self,):
        """ 获取所有用户信息 """
        return self.user.ser_storage

    def delete_user(self,user_id):
        """ 根据用户ID删除用户信息 """
        if user_id in self.user_storage:
            del self.user_storage[user_id]
        return True
