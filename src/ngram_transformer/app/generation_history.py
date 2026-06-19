from __future__ import annotations

from ngram_transformer.app.schemas import GenerateRequest, GenerateResponse
from ngram_transformer.infra.supabase_gateway import (
    GenerationInsert,
    GenerationRecord,
    SupabaseGateway,
)


def generation_insert_from_result(
    request: GenerateRequest,
    result: GenerateResponse,
) -> GenerationInsert:
    return GenerationInsert(
        model_type=result.model_name,
        model_version_label=result.model_version_label,
        prompt=result.prompt,
        generated_text=result.generated_text,
        generation_params=result.generation_params.model_dump(),
        seed=request.seed,
    )


async def save_generation_result(
    gateway: SupabaseGateway,
    access_token: str,
    request: GenerateRequest,
    result: GenerateResponse,
) -> GenerateResponse:
    record = await gateway.save_generation(
        access_token,
        generation_insert_from_result(request, result),
    )
    return result.model_copy(update={"saved_generation_id": record.id})


def format_history(records: list[GenerationRecord]) -> str:
    return "\n\n".join(
        f"{record.created_at} | {record.model_version_label}\n{record.generated_text}"
        for record in records
    )
