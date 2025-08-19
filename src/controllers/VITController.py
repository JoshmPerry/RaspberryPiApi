from fastapi import FastAPI
from client.ImageRequest import ImageRequest
from client.PredictionResponse import PredictionResponse
from services.VITService import VITService

class VITController:
    def __init__(self):
        self.VITService = VITService()
    def register_routes(self, app):
        @app.get("/vit")
        async def vit_endpoint(request: ImageRequest):
            data_list = request.data
            # Perform VIT on the data
            answer, confidence = self.VITService.predict(data_list)
            return PredictionResponse(answer=answer, confidence=confidence)