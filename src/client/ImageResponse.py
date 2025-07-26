from pydantic import BaseModel

class ImageResponse(BaseModel):
    data: list[float]