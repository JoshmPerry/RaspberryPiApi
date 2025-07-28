from fastapi import FastAPI
from client.ImageRequest import ImageRequest
from client.PredictionResponse import PredictionResponse
from services.MLPService import MLPService

class MLPController:
    def __init__(self):
        self.MLPService = MLPService()
    def register_routes(self, app):
        @app.get("/mlp")
        async def mlp_endpoint(request: ImageRequest):
            data_list = request.data
            # Perform PCA on the data
            answer, confidence = self.MLPService.predict(data_list)
            return PredictionResponse(answer=answer, confidence=confidence)