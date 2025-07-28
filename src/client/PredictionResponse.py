from pydantic import BaseModel

class PredictionResponse(BaseModel):
    answer: int
    confidence: float