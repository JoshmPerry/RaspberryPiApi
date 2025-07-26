# def obsurefunc(myString):
#     return myString * 2

# myglobalVars = {
#     'hi': 'car'
# }

from fastapi import FastAPI
from controllers.PCAController import PCAController

class RoutesHandler:
    def __init__(self):
        self.pca_controller = PCAController()

    def register_routes(self, app: FastAPI):
        self.pca_controller.register_routes(app)
# def register_routes(app):
#     @app.get("/")
#     async def read_root():
#         return {"message": "Welcome to my api!"}

#     @app.get("/items/{item_id}")
#     async def read_item(item_id: int, q: str = None):
#         return {"item_id": item_id, "q": q}

#     @app.get("/recognizeDigit")
#     async def recognize_digit(vector: str):
#         returnvalue = []
#         temp = vector.split(',')
#         for i in temp:
#             returnvalue.append(float(i))
#         return {"recievedItem": returnvalue}

#     @app.get("/localization")
#     async def localization():
#         return {"computed": obsurefunc(myglobalVars['hi'])}