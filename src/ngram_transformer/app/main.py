from __future__ import annotations

import os

import uvicorn

from ngram_transformer.app.api import create_app

app = create_app()


def run() -> None:
    uvicorn.run(
        "ngram_transformer.app.main:app",
        host=os.getenv("APP_HOST", "127.0.0.1"),
        port=int(os.getenv("APP_PORT", "8000")),
        reload=os.getenv("APP_ENV") == "development",
    )


if __name__ == "__main__":
    run()
