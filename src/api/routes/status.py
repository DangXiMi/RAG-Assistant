import json

from fastapi import (
    APIRouter,
    HTTPException,
)

from arq import create_pool
from arq.connections import RedisSettings
import os

from src.ingestion.loaders import FORMAT_DESCRIPTIONS, supported_extension_list
from src.ingestion.loaders.capabilities import ocr_available, ocr_unavailable_reason

router = APIRouter()

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))


@router.get("/api/v1/formats")
async def get_supported_formats():
    """Report which file types can be ingested, and whether OCR is usable.

    The UI reads this so its upload widget always matches the backend, and so
    OCR being unavailable is visible before a user uploads a scan.
    """
    return {
        "extensions": supported_extension_list(),
        "groups": FORMAT_DESCRIPTIONS,
        "ocr": {
            "available": ocr_available(),
            "reason": None if ocr_available() else ocr_unavailable_reason(),
        },
    }


@router.get("/api/v1/job/{job_id}")
async def get_job_status(
    job_id: str
):

    redis = await create_pool(
        RedisSettings(host=redis_host, port=redis_port)
    )


    data = await redis.get(
        f"job:status:{job_id}"
    )


    await redis.close()


    if not data:

        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )


    return json.loads(data)