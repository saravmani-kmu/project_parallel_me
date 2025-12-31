from pydantic import BaseModel
from typing import Literal

class InputRequest(BaseModel):
    message: str
    model_provider: Literal["openai", "gemini"] = "gemini"

class OutputResponse(BaseModel):
    response: str
