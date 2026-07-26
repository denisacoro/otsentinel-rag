from fastapi import FastAPI

app = FastAPI(
    title="OTSentinel AI",
    description=(
        "Multilingual RAG and fine-tuned LLM platform for CPS, SCADA and Industrial IoT security."
    ),
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    """Return basic application information."""
    return {
        "application": "OTSentinel AI",
        "status": "running",
        "documentation": "/docs",
    }


@app.get("/api/v1/health")
def health_check() -> dict[str, str]:
    """Return the health status of the API."""
    return {
        "status": "healthy",
        "version": "0.1.0",
    }
