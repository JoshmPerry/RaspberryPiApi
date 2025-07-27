from fastapi import FastAPI
from client.ImageRequest import ImageRequest
from client.ImageResponse import ImageResponse
from services.PCAService import PCAService

class PCAController:
    def __init__(self):
        self.PCAService = PCAService()
    def register_routes(self, app):
        @app.get("/pca")
        async def pca_endpoint(request: ImageRequest):
            data_list = request.data
            # Perform PCA on the data
            pca_result = list(self.PCAService.compressThenReconstruct(data_list))
            return ImageResponse(data=pca_result)