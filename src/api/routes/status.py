import json

from fastapi import (
    APIRouter,
    HTTPException,
)

from arq import create_pool
from arq.connections import RedisSettings
import os

router = APIRouter()

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
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