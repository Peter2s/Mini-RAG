import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for domain errors that map to HTTP responses."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class BadRequestError(AppError):
    status_code = 400
    code = "bad_request"


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"


class LLMError(AppError):
    status_code = 502
    code = "llm_error"


class LLMRefusalError(LLMError):
    status_code = 422
    code = "llm_refusal"


class VectorStoreError(AppError):
    status_code = 502
    code = "vector_store_error"


class EmbeddingError(AppError):
    status_code = 502
    code = "embedding_error"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, error: AppError) -> JSONResponse:
        if error.status_code >= 500:
            logger.error("%s: %s", error.code, error.message)
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": error.message}},
        )
