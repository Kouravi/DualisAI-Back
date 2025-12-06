from pydantic import BaseModel
from typing import Optional, Union, Dict

class PredictionModel(BaseModel):
    filename: str
    tipo_modelo: str
    resultado: Dict[str, Union[str, float]]
    timestamp: Optional[str]