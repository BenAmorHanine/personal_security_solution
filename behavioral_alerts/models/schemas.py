from pydantic import BaseModel

#OPTIONALLL

class CaptureRequest(BaseModel):
    user_id: str
    latitude: float
    longitude: float
    emergency: bool = False
