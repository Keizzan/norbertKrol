# Import libraries
from pydantic import BaseModel, Field, EmailStr, ConfigDict


# Validator class for form 
class Validator(BaseModel):
    # Strip white spaces
    model_config = ConfigDict(str_strip_whitespace=True)
    # Validate each field from form
    name: str = Field(min_length=1, max_length=50)
    email: EmailStr
    subject: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=1000)