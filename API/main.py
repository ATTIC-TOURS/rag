from fastapi import FastAPI
from src.api_routers import messenger


app = FastAPI(
    title="Attic Tours API",
    summary="This is for supporting Attic Tours Operational Process",
    version="1.0.0",
)

app.include_router(messenger.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Hello From Attic Tours!"}
