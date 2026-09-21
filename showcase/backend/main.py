# Import libraries
from fastapi import FastAPI, HTTPException
from validator import Validator
import resend
from fastapi.middleware.cors import CORSMiddleware
import resend
import config

resend.api_key = config.resend_api_key

# Initialize FastAPI app
app = FastAPI()

# Configure CORS to accept requests from frontend
app.add_middleware(CORSMiddleware, allow_origins=[config.site], allow_methods=["POST"], allow_headers=["*"])

# Define endpoint for form submission
@app.post("/submit")
def submit(form: Validator):
    try:
        resend.Emails.send({
            'from': config.from_email,
            'to': config.to_email,
            'subject':f'New form submission: {form.subject}',
            'text': f'Name: {form.name}\nEmail: {form.email}\nMessage: {form.message}'
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "Form submitted successfully"}