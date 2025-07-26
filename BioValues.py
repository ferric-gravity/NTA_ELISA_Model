from pydantic import BaseModel
class Biovalue(BaseModel):
    Gender : str
    ELISA  : float
    NTA_Scatter : int