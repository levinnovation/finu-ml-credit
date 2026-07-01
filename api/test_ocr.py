"""Tests for OCR schema parsing helpers."""

from api.ocr import (
    CedulaOCRFields,
    _fields_to_response,
    _normalize_cedula,
    _parse_json_from_text,
)


def test_normalize_cedula():
    assert _normalize_cedula("118470559") == "1-1847-0559"
    assert _normalize_cedula("1-1847-0559") == "1-1847-0559"


def test_parse_json_from_fence():
    raw = '```json\n{"cedula":"118470559","confidence":90}\n```'
    data = _parse_json_from_text(raw)
    assert data["cedula"] == "118470559"


def test_fields_to_response_cedula():
    fields, confidence = _fields_to_response(
        "cedula_front",
        {
            "cedula": "118470559",
            "nombre": "MOSHE",
            "apellidos": "ROSENSTOCK SIHMAN",
            "fecha_nacimiento": "2002-06-23",
            "sexo": "M",
            "confidence": 92,
        },
    )
    assert fields["cedula"] == "1-1847-0559"
    assert fields["fecha_nacimiento"] == "2002-06-23"
    assert confidence == 92


def test_cedula_model_validation():
    parsed = CedulaOCRFields.model_validate(
        {"fecha_nacimiento": "2002-06-23", "confidence": 80}
    )
    assert parsed.fecha_nacimiento.isoformat() == "2002-06-23"
