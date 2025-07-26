from pydantic import BaseModel

class ImageRequest(BaseModel):
    data: list[float]