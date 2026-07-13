from pathlib import Path
import uuid

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
)

from arq import create_pool
from arq.connections import RedisSettings

router = APIRouter()


UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(
    exist_ok=True
)


@router.post("/api/v1/ingest")
async def ingest(
    file: UploadFile = File(...)
):

    try:

        job_id = str(
            uuid.uuid4()
        )


        filename = (
            f"{job_id}_{file.filename}"
        )


        file_path = (
            UPLOAD_DIR / filename
        )


        # Save uploaded file
        with open(file_path,"wb") as f:
            while chunk := await file.read(
                1024 * 1024
            ):
                f.write(chunk) 

        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
        }


        redis = await create_pool(
            RedisSettings()
        )


        await redis.enqueue_job(
            "ingest_document",
            str(file_path),
            metadata,
            job_id,
        )


        # initial status
        await redis.set(
            f"job:status:{job_id}",
            '{"status":"queued","result":null}',
        )


        await redis.close()


        return {
            "job_id": job_id,
            "status": "queued",
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )