# def obsurefunc(myString):
#     return myString * 2

# myglobalVars = {
#     'hi': 'car'
# }

from fastapi import FastAPI
from controllers.PCAController import PCAController
from controllers.MLPController import MLPController

class RoutesHandler:
    def __init__(self):
        self.pca_controller = PCAController()
        self.mlp_controller = MLPController()

    def register_routes(self, app: FastAPI):
        self.pca_controller.register_routes(app)
        self.mlp_controller.register_routes(app)
