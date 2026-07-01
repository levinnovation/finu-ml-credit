"""POST /api/ocr/extract — structured document OCR via vision LLM (LiteLLM/DeepInfra)."""

from __future__ import annotations

import base64
import json
import logging
import re
from datetime import date, datetime
from typing import Any, Literal, Optional

import httpx
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ocr", tags=["ocr"])

OCRDocumentType = Literal[
    "cedula_front",
    "cedula_back",
    "salary_proof",
    "bank_statement",
    "financial_statements",
    "personeria_juridica",
    "other",
]


class CedulaOCRFields(BaseModel):
    cedula: Optional[str] = None
    nombre: Optional[str] = None
    apellidos: Optional[str] = None
    fecha_nacimiento: Optional[date] = None
    fecha_vencimiento: Optional[date] = None
    sexo: Optional[Literal["M", "F"]] = None
    confidence: float = Field(default=0.0, ge=0, le=100)


class SalaryProofOCRFields(BaseModel):
    ingreso_mensual_bruto: Optional[float] = None
    ingreso_mensual_neto: Optional[float] = None
    employer_name: Optional[str] = None
    fecha_documento: Optional[date] = None
    cedula: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0, le=100)


class GenericOCRFields(BaseModel):
    ruc: Optional[str] = None
    razon_social: Optional[str] = None
    ventas_anuales: Optional[float] = None
    raw_fields: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0, le=100)


class OCRExtractResponse(BaseModel):
    document_type: str
    provider: str = "deepseek_vlm"
    mime_type: str
    fields: dict[str, Any]
    raw_text: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0, le=100)


def _check_auth(x_internal_secret: Optional[str]) -> None:
    expected = settings.ml_internal_secret
    if expected and x_internal_secret != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _litellm_endpoint() -> str:
    base = (getattr(settings, "litellm_base_url", None) or "").strip()
    if base:
        return f"{base.rstrip('/')}/chat/completions"
    if getattr(settings, "deepinfra_api_key", None):
        return "https://api.deepinfra.com/v1/openai/chat/completions"
    return "https://api.deepinfra.com/v1/openai/chat/completions"


def _litellm_api_key() -> str:
    return (
        getattr(settings, "litellm_api_key", None)
        or getattr(settings, "deepinfra_api_key", None)
        or getattr(settings, "openai_api_key", None)
        or ""
    ).strip()


def _ocr_model() -> str:
    return (
        getattr(settings, "ocr_vlm_model", None)
        or "deepinfra/deepseek-ai/DeepSeek-V3.2"
    ).strip()


def _prompt_for_type(document_type: str) -> str:
    if document_type in ("cedula_front", "cedula_back"):
        return (
            "Extrae datos de una cédula de identidad costarricense. "
            "Responde SOLO con JSON válido (sin markdown) con estas claves: "
            '{"cedula":"1-2345-6789","nombre":"...","apellidos":"...","fecha_nacimiento":"YYYY-MM-DD",'
            '"fecha_vencimiento":"YYYY-MM-DD","sexo":"M|F","confidence":0-100}. '
            "Usa null para campos ilegibles. fecha_nacimiento suele estar en el reverso."
        )
    if document_type == "salary_proof":
        return (
            "Extrae datos de un comprobante de salario costarricense. "
            'JSON: {"ingreso_mensual_bruto":number,"ingreso_mensual_neto":number,'
            '"employer_name":"...","fecha_documento":"YYYY-MM-DD","cedula":"...","confidence":0-100}'
        )
    if document_type == "personeria_juridica":
        return (
            "Extrae datos de personería jurídica CR. "
            'JSON: {"ruc":"3-101-123456","razon_social":"...","confidence":0-100}'
        )
    return (
        "Extrae texto estructurado del documento financiero/legal costarricense. "
        'JSON: {"raw_fields":{...},"confidence":0-100}'
    )


def _parse_json_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _normalize_cedula(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) != 9:
        return value.strip() or None
    return f"{digits[0]}-{digits[1:5]}-{digits[5:]}"


def _fields_to_response(document_type: str, data: dict[str, Any]) -> tuple[dict[str, Any], float]:
    try:
        if document_type in ("cedula_front", "cedula_back"):
            parsed = CedulaOCRFields.model_validate(data)
            fields = parsed.model_dump(mode="json")
            if fields.get("cedula"):
                fields["cedula"] = _normalize_cedula(fields["cedula"])
            return fields, parsed.confidence
        if document_type == "salary_proof":
            parsed = SalaryProofOCRFields.model_validate(data)
            return parsed.model_dump(mode="json"), parsed.confidence
        parsed = GenericOCRFields.model_validate(data)
        return parsed.model_dump(mode="json"), parsed.confidence
    except ValidationError as err:
        logger.warning("OCR schema validation failed: %s", err)
        confidence = float(data.get("confidence") or 30)
        return {"raw_fields": data, "parse_errors": err.errors()}, confidence


async def _call_vision_llm(
    file_bytes: bytes,
    mime_type: str,
    document_type: str,
) -> tuple[dict[str, Any], str]:
    api_key = _litellm_api_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="LiteLLM/OpenAI API key not configured")

    image_b64 = base64.b64encode(file_bytes).decode()
    prompt = _prompt_for_type(document_type)
    model = _ocr_model()
    endpoint = _litellm_endpoint()

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 1200,
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )

    if resp.status_code >= 400:
        logger.error("Vision LLM error %s: %s", resp.status_code, resp.text[:500])
        raise HTTPException(status_code=502, detail=f"Vision LLM HTTP {resp.status_code}")

    body = resp.json()
    choices = body.get("choices") or []
    if not choices:
        raise HTTPException(status_code=502, detail="Vision LLM returned no choices")

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )

    try:
        parsed = _parse_json_from_text(str(content))
    except json.JSONDecodeError as err:
        logger.warning("OCR JSON parse failed: %s | raw=%s", err, str(content)[:300])
        return {"raw_fields": {"text": str(content)}}, str(content)

    return parsed, str(content)


@router.post("/extract", response_model=OCRExtractResponse)
async def extract_ocr(
    file: UploadFile = File(...),
    document_type: str = Form(default="other"),
    mime_type: Optional[str] = Form(default=None),
    x_internal_secret: Optional[str] = Header(default=None, alias="x-internal-secret"),
):
    _check_auth(x_internal_secret)

    if document_type not in (
        "cedula_front",
        "cedula_back",
        "salary_proof",
        "bank_statement",
        "financial_statements",
        "personeria_juridica",
        "other",
    ):
        raise HTTPException(status_code=422, detail="document_type inválido")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Archivo vacío")
    if len(file_bytes) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Archivo demasiado grande (máx 12MB)")

    resolved_mime = (mime_type or file.content_type or "application/octet-stream").split(";")[0].strip()
    if resolved_mime == "application/octet-stream" and file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        ext_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
            "pdf": "application/pdf",
        }
        resolved_mime = ext_map.get(ext, resolved_mime)

    raw_json, raw_text = await _call_vision_llm(file_bytes, resolved_mime, document_type)
    fields, confidence = _fields_to_response(document_type, raw_json)

    return OCRExtractResponse(
        document_type=document_type,
        provider="deepseek_vlm",
        mime_type=resolved_mime,
        fields=fields,
        raw_text=raw_text[:4000] if raw_text else None,
        confidence=confidence,
    )
