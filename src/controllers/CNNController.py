from fastapi import FastAPI
from client.ImageRequest import ImageRequest
from client.PredictionResponse import PredictionResponse
from services.CNNService import CNNService

class CNNController:
    def __init__(self):
        self.CNNService = CNNService()
    def register_routes(self, app):
        @app.get("/cnn")
        async def mlp_endpoint(request: ImageRequest):
            data_list = request.data
            # Perform PCA on the data
            answer, confidence = self.CNNService.predict(data_list)
            return PredictionResponse(answer=answer, confidence=confidence)