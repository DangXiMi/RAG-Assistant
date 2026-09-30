from pathlib import Path
import uuid
import os

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
)

from arq import create_pool
from arq.connections import RedisSettings

from src.ingestion.loaders import (
    FORMAT_DESCRIPTIONS,
    SUPPORTED_EXTENSIONS,
    is_supported,
)

router = APIRouter()
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))



UPLOAD_DIR = Path("uploads")

UPLOAD_DIR.mkdir(
    exist_ok=True
)

# Upload size ceiling, to keep a single job inside the worker timeout.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))


@router.post("/api/v1/ingest")
async def ingest(
    file: UploadFile = File(...)
):

    try:

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file has no filename.",
            )

        # Reject unsupported formats here rather than queueing a job that is
        # guaranteed to fail in the worker.
        if not is_supported(file.filename):
            supported = ", ".join(sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS))
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type '{Path(file.filename).suffix or '(none)'}'. "
                    f"Supported types: {supported}"
                ),
            )

        job_id = str(
            uuid.uuid4()
        )


        filename = (
            f"{job_id}_{file.filename}"
        )


        file_path = (
            UPLOAD_DIR / filename
        )


        # Save uploaded file, enforcing the size ceiling as we stream.
        written = 0
        with open(file_path,"wb") as f:
            while chunk := await file.read(
                1024 * 1024
            ):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    f.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the maximum upload size of "
                            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                        ),
                    )
                f.write(chunk) 

        metadata = {
            "filename": file.filename,
            # The on-disk name is prefixed with the job id; keep the real name
            # so citations read "report.pdf" and not "<uuid>_report.pdf".
            "original_filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": written,
        }


        redis = await create_pool(
            RedisSettings(host=redis_host, port=redis_port)
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

    except HTTPException:
        # Validation failures (415/413/400) must reach the client unchanged.
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )