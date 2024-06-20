from fastapi import FastAPI
from app import endpoints
app = FastAPI()


# Include routers
app.include_router(endpoints.router, prefix="", tags=["Query LLM"])
