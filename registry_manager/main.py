# import uvicorn
# from fastapi import FastAPI, HTTPException
# from registry import FileRegistry
# from pydantic import BaseModel, model_validator, field_validator
# from typing import Optional, Literal

# file_registry_client = FileRegistry()

# StatusType = Literal["downloaded", "parsed", "embedded", "error"]

# class StatusUpdate(BaseModel):
#     filename: str
#     status: StatusType
#     error_msg: Optional[str] = None

#     @field_validator('*', mode='before')
#     @classmethod
#     def strip_strings(cls, v):
#         if isinstance(v, str):
#             return v.strip()
        
#         return v

#     @model_validator(mode='after')
#     def validate_error_status(self):
#         if self.status == "error":
#             if not self.error_msg:
#                 raise ValueError("Missing Error Message. Send error message to log reason for error.")
#         return self

# class GetStatus(BaseModel):
#     filename: str

#     @field_validator('*', mode='before')
#     @classmethod
#     def strip_strings(cls, v):
#         if isinstance(v, str):
#             return v.strip()
        
#         return v

# registry = FastAPI()

# @registry.get("/")
# def read_root():
#     return {"Welcome": " to registry manager!"}

# @registry.post('/healthcheck')
# def healthcheck():
#     return {"Status": "Okay"}

# @registry.post('/update_status')
# def update_file_status(request: StatusUpdate):
#     resp_update = file_registry_client.update_status(**request.model_dump())
#     if resp_update:
#         return {"success": True}
#     return {"success": False}

# @registry.get('/get_status')
# def get_file_status(request: GetStatus):
#     print("Request received")
#     stat_response = file_registry_client.get_status(**request.model_dump())
#     print("Status Response: ", stat_response)
#     return {"status": stat_response}

# # registry.get('/check_file_existance')
# # def file_existance_validation(request: GetStatus):
# #     file_registry_client.exists(**request.model_dump())


# if __name__ == "__main__":
#     uvicorn.run("main:registry", host="127.0.0.1", port=4000, reload=True)

import uvicorn
from fastapi import FastAPI
from registry import FileRegistry
from pydantic import BaseModel, model_validator, field_validator
from typing import Optional
from datetime import datetime
from config import STATUS_TYPE as StatusType

file_registry_client = FileRegistry()
registry = FastAPI()



class StatusUpdate(BaseModel):
    filename: str
    status: StatusType
    domain: Optional[str] = None       # NEW
    published_at: Optional[datetime] = None # NEW
    error_msg: Optional[str] = None

    @field_validator('*', mode='before')
    @classmethod
    def strip_strings(cls, v):
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode='after')
    def validate_error_status(self):
        if self.status == "error" and not self.error_msg:
            raise ValueError("Missing Error Message.")
        return self

class GetStatus(BaseModel):
    filename: str

class GetCheckpoint(BaseModel):
    domain: str

@registry.post('/update_status')
def update_file_status(request: StatusUpdate):
    # Pass all fields to the backend
    resp_update = file_registry_client.update_status(**request.model_dump())
    return {"success": resp_update}

@registry.get('/get_status')
def get_file_status(request: GetStatus): # Changed to query param friendly if needed, but Body is fine
    stat_response = file_registry_client.get_status(request.filename)
    return {"status": stat_response}

@registry.post('/get_last_checkpoint')
def get_last_checkpoint(request: GetCheckpoint):
    """Returns the date of the most recent paper downloaded for this domain."""
    last_date = file_registry_client.get_last_domain_date(request.domain)
    return {"last_checkpoint": last_date}

if __name__ == "__main__":
    uvicorn.run("main:registry", host="127.0.0.1", port=4000, reload=True)