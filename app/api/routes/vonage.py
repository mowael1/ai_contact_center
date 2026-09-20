from fastapi import (
    APIRouter,
    Query,
    Request,
)

from app.services.vonage_service import (
    build_question_ncco,
    process_call_event,
    process_speech_input,
)


router = APIRouter()


@router.get("/answer")
async def answer_call():

    return build_question_ncco(
        attempt=1
    )


@router.post("/input")
async def voice_input(
    request: Request,
    attempt: int = Query(default=1),
):

    data = await request.json()

    return process_speech_input(
        data=data,
        attempt=attempt,
    )


@router.post("/events")
async def call_events(
    request: Request
):

    data = await request.json()

    process_call_event(
        data
    )

    return {
        "status": "ok"
    }