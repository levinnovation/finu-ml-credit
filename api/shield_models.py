"""GET /shield/models — Shield model registry metadata."""

from fastapi import APIRouter

from shield.registry import get_active_metadata

router = APIRouter(prefix="/shield/models", tags=["shield"])


@router.get("")
async def shield_models():
    return get_active_metadata()
