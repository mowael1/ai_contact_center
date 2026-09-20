from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import (
    get_db,
    require_agent,
)
from app.models import User
from app.schemas.call import (
    CallResponse,
    MockCallResult,
)
from app.services.call_service import (
    set_mock_call_result,
)


router = APIRouter()


@router.patch(
    "/{call_id}/mock-result",
    response_model=CallResponse
)
def update_mock_call_result(
    call_id: int,
    data: MockCallResult,
    db: Session = Depends(get_db),
    current_agent: User = Depends(require_agent)
):
    return set_mock_call_result(
        db,
        current_agent,
        call_id,
        data
    )