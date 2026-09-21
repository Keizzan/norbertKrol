# Showcase SPA

## Backend
FastAPI server that receives form from frontend contact section in JSON format and using resend API (https://resend.com/) sends via email.

### Tech Stack
- FastAPI
- Pydantic
- Resend
- Uvicorn

### Setup
```sh
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Server will be hosted at http://localhost:8000

### Variables
#### config.py
-site -> Frontend url (default: http://localhost:5173)
-resend_api_key -> Resend API KEY
-from_email -> Email that will send emails
-to_email -> Email that will receive emails

### Endpoint
/submit 

## Frontend
Vue3 & Vite SPA portfolio site.

### Tech Stack
-Vue3
-Vite
-Bootstrap
-AOS

### Sections
-Hero
-About
-Skills
-Resume
-Portfolio
-Contact

### Setup
```sh
cd frontend
npm run dev (development)
npm run build(production)
```
