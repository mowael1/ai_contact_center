from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.repositories import (
    call_repository,
    ticket_repository,
)
from app.services.llm_service import (
    classify_follow_up_response,
)


# =========================================
# Configuration
# =========================================

MAX_ATTEMPTS = 2

STT_CONFIDENCE_THRESHOLD = 0.60


# =========================================
# Audio files
# =========================================

ASK_QUESTION_AUDIO = "ask_question.mp3"

REPEAT_UNCLEAR_AUDIO = "repeat_unclear.mp3"

REDIRECT_OFF_TOPIC_AUDIO = "redirect_off_topic.mp3"

RESOLVED_AUDIO = "resolved_thanks.mp3"

NOT_RESOLVED_AUDIO = "not_resolved_closing.mp3"


# =========================================
# General helpers
# =========================================

def _utc_now() -> datetime:

    return datetime.now(
        timezone.utc
    ).replace(tzinfo=None)


def _base_url() -> str:

    return settings.PUBLIC_BASE_URL.rstrip("/")


def _audio_url(
    filename: str,
) -> str:

    return (
        f"{_base_url()}"
        f"/audio/{filename}"
    )


def _input_url(
    attempt: int,
) -> str:

    return (
        f"{_base_url()}"
        f"/api/v1/vonage/input"
        f"?attempt={attempt}"
    )


# =========================================
# NCCO helpers
# =========================================

def _stream_action(
    filename: str,
) -> dict[str, Any]:

    return {
        "action": "stream",
        "streamUrl": [
            _audio_url(filename)
        ],
    }


def _speech_input_action(
    attempt: int,
) -> dict[str, Any]:

    return {
        "action": "input",

        "type": [
            "speech"
        ],

        "eventUrl": [
            _input_url(attempt)
        ],

        "eventMethod": "POST",

        "speech": {
            "endOnSilence": 0.5,

            "startTimeout": 6,

            "maxDuration": 8,

            "provider": "google",

            "providerOptions": {
                "language_code": "ar-EG",
            },
        },
    }


# =========================================
# Conversation NCCOs
# =========================================

def build_question_ncco(
    attempt: int = 1,
) -> list[dict[str, Any]]:

    print("\n==============================")
    print("BUILD QUESTION NCCO")
    print("ATTEMPT:", attempt)
    print("AUDIO:", ASK_QUESTION_AUDIO)
    print("==============================\n")

    return [
        _stream_action(
            ASK_QUESTION_AUDIO
        ),

        _speech_input_action(
            attempt
        ),
    ]


def build_repeat_unclear_ncco(
    attempt: int,
) -> list[dict[str, Any]]:

    print("\n==============================")
    print("REPEAT UNCLEAR")
    print("ATTEMPT:", attempt)
    print("AUDIO:", REPEAT_UNCLEAR_AUDIO)
    print("==============================\n")

    return [
        _stream_action(
            REPEAT_UNCLEAR_AUDIO
        ),

        _speech_input_action(
            attempt
        ),
    ]


def build_redirect_off_topic_ncco(
    attempt: int,
) -> list[dict[str, Any]]:

    print("\n==============================")
    print("REDIRECT OFF TOPIC")
    print("ATTEMPT:", attempt)
    print("AUDIO:", REDIRECT_OFF_TOPIC_AUDIO)
    print("==============================\n")

    return [
        _stream_action(
            REDIRECT_OFF_TOPIC_AUDIO
        ),

        _speech_input_action(
            attempt
        ),
    ]


def build_resolved_ncco() -> list[dict[str, Any]]:

    print("\n==============================")
    print("BUILD RESOLVED RESPONSE")
    print("AUDIO:", RESOLVED_AUDIO)
    print("==============================\n")

    return [
        _stream_action(
            RESOLVED_AUDIO
        )
    ]


def build_not_resolved_ncco() -> list[dict[str, Any]]:

    print("\n==============================")
    print("BUILD NOT RESOLVED RESPONSE")
    print("AUDIO:", NOT_RESOLVED_AUDIO)
    print("==============================\n")

    return [
        _stream_action(
            NOT_RESOLVED_AUDIO
        )
    ]


def build_unclear_final_ncco() -> list[dict[str, Any]]:

    print("\n==============================")
    print("BUILD UNCLEAR FINAL RESPONSE")
    print("AUDIO:", NOT_RESOLVED_AUDIO)
    print("==============================\n")

    return [
        _stream_action(
            NOT_RESOLVED_AUDIO
        )
    ]


# =========================================
# Speech helpers
# =========================================

def _extract_best_speech_result(
    data: dict[str, Any],
) -> tuple[str, float | None]:

    speech = data.get(
        "speech",
        {}
    )

    results = speech.get(
        "results",
        []
    )

    if not results:
        return "", None

    best_result = results[0]

    text = (
        best_result
        .get(
            "text",
            "",
        )
        .strip()
    )

    confidence = best_result.get(
        "confidence"
    )

    try:

        if confidence is not None:

            confidence = float(
                confidence
            )

    except (
        TypeError,
        ValueError,
    ):

        confidence = None

    return (
        text,
        confidence,
    )


# =========================================
# Database helpers
# =========================================

def _get_call_by_provider_id(
    db: Session,
    provider_call_id: str | None,
):

    if not provider_call_id:
        return None

    return (
        call_repository
        .get_by_provider_call_id(
            db,
            provider_call_id
        )
    )


def _save_transcript(
    db: Session,
    call,
    *,
    text: str,
    attempt: int,
) -> None:

    if not text:
        return

    transcript_line = (
        f"Attempt {attempt}: {text}"
    )

    if call.transcript:

        call.transcript = (
            f"{call.transcript}\n"
            f"{transcript_line}"
        )

    else:

        call.transcript = transcript_line

    db.commit()
    db.refresh(call)

    print("\n==============================")
    print("TRANSCRIPT SAVED")
    print("CALL ID:", call.id)
    print("TEXT:", transcript_line)
    print("==============================\n")


def _save_call_outcome(
    db: Session,
    call,
    *,
    outcome: str,
) -> None:

    ticket = (
        ticket_repository
        .get_by_id_and_company(
            db,
            call.ticket_id,
            call.company_id
        )
    )

    if ticket is None:

        print("\n==============================")
        print("TICKET NOT FOUND")
        print("CALL ID:", call.id)
        print("TICKET ID:", call.ticket_id)
        print("==============================\n")

        return

    # -------------------------------------
    # Save business outcome
    # -------------------------------------

    call.outcome = outcome

    # -------------------------------------
    # Update Ticket
    # -------------------------------------

    if outcome == "resolved":

        ticket.status = "resolved"

    elif outcome in {
        "not_resolved",
        "unclear",
    }:

        ticket.status = "needs_agent"

    ticket.updated_at = _utc_now()

    # Important:
    # We DO NOT set call.status = completed here.
    # Vonage events control the call lifecycle status.

    db.commit()
    db.refresh(call)

    print("\n==============================")
    print("DATABASE OUTCOME UPDATED")
    print("CALL ID:", call.id)
    print("OUTCOME:", call.outcome)
    print("TICKET ID:", ticket.id)
    print("TICKET STATUS:", ticket.status)
    print("==============================\n")


def _keep_ticket_pending(
    db: Session,
    call,
) -> None:

    ticket = (
        ticket_repository
        .get_by_id_and_company(
            db,
            call.ticket_id,
            call.company_id
        )
    )

    if ticket is None:

        print(
            "TICKET NOT FOUND FOR CALL:",
            call.id
        )

        return

    ticket.status = "pending_follow_up"

    ticket.updated_at = _utc_now()

    # No commit here.
    # process_call_event() commits Call + Ticket together.


# =========================================
# Final unclear helper
# =========================================

def _finish_as_unclear(
    db: Session,
    call,
) -> list[dict[str, Any]]:

    print("FINAL OUTCOME: unclear")
    print("ACTION: SEND TO HUMAN AGENT")

    if call is not None:

        _save_call_outcome(
            db,
            call,
            outcome="unclear",
        )

    return build_unclear_final_ncco()


# =========================================
# Speech processing
# =========================================

def process_speech_input(
    *,
    db: Session,
    data: dict[str, Any],
    attempt: int,
) -> list[dict[str, Any]]:

    print("\n==============================")
    print("VONAGE SPEECH INPUT")
    print("ATTEMPT:", attempt)

    # Vonage Call UUID
    provider_call_id = data.get(
        "uuid"
    )

    print(
        "PROVIDER CALL ID:",
        provider_call_id
    )

    # -------------------------------------
    # Find our Call in DB
    # -------------------------------------

    call = _get_call_by_provider_id(
        db,
        provider_call_id
    )

    if call is None:

        print(
            "WARNING: CALL NOT FOUND "
            "FOR PROVIDER UUID:",
            provider_call_id
        )

    # -------------------------------------
    # Get speech transcript
    # -------------------------------------

    text, confidence = (
        _extract_best_speech_result(
            data
        )
    )

    print("TRANSCRIPT:", text)
    print("STT CONFIDENCE:", confidence)
    print("==============================\n")

    # -------------------------------------
    # Save transcript
    # -------------------------------------

    if (
        call is not None
        and text
    ):

        _save_transcript(
            db,
            call,
            text=text,
            attempt=attempt,
        )

    # =====================================
    # No speech detected
    # =====================================

    if not text:

        print("NO SPEECH DETECTED")

        if attempt < MAX_ATTEMPTS:

            print(
                "ACTION: ASK CUSTOMER TO REPEAT"
            )

            return build_repeat_unclear_ncco(
                attempt=attempt + 1
            )

        return _finish_as_unclear(
            db,
            call
        )

    # =====================================
    # Low STT confidence
    # =====================================

    if (
        confidence is not None
        and confidence
        < STT_CONFIDENCE_THRESHOLD
    ):

        print("LOW STT CONFIDENCE")
        print(
            "CONFIDENCE:",
            confidence
        )

        if attempt < MAX_ATTEMPTS:

            print(
                "ACTION: ASK CUSTOMER TO REPEAT"
            )

            return build_repeat_unclear_ncco(
                attempt=attempt + 1
            )

        return _finish_as_unclear(
            db,
            call
        )

    # =====================================
    # Send transcript to Groq LLM
    # =====================================

    print(
        "\nSENDING TRANSCRIPT TO GROQ..."
    )

    outcome = (
        classify_follow_up_response(
            text
        )
    )

    print(
        "LLM OUTCOME:",
        outcome
    )

    # =====================================
    # Resolved
    # =====================================

    if outcome == "resolved":

        print(
            "FINAL OUTCOME: resolved"
        )

        print(
            "ACTION: CLOSE CALL AS RESOLVED"
        )

        if call is not None:

            _save_call_outcome(
                db,
                call,
                outcome="resolved",
            )

        return build_resolved_ncco()

    # =====================================
    # Not resolved
    # =====================================

    if outcome == "not_resolved":

        print(
            "FINAL OUTCOME: not_resolved"
        )

        print(
            "ACTION: SEND TO HUMAN AGENT"
        )

        if call is not None:

            _save_call_outcome(
                db,
                call,
                outcome="not_resolved",
            )

        return build_not_resolved_ncco()

    # =====================================
    # Unclear
    # =====================================

    print(
        "LLM OUTCOME IS UNCLEAR"
    )

    # First unclear answer
    if attempt < MAX_ATTEMPTS:

        print(
            "ACTION: REDIRECT CUSTOMER "
            "BACK TO THE QUESTION"
        )

        return build_redirect_off_topic_ncco(
            attempt=attempt + 1
        )

    # Second unclear answer
    return _finish_as_unclear(
        db,
        call
    )


# =========================================
# Vonage call events
# =========================================

def process_call_event(
    *,
    db: Session,
    data: dict[str, Any],
) -> None:

    provider_call_id = data.get(
        "uuid"
    )

    vonage_status = data.get(
        "status"
    )

    detail = data.get(
        "detail"
    )

    print("\n==============================")
    print("VONAGE EVENT")
    print(
        "UUID:",
        provider_call_id
    )
    print(
        "STATUS:",
        vonage_status
    )
    print(
        "DETAIL:",
        detail
    )
    print("==============================\n")

    # -------------------------------------
    # Need UUID to identify our Call
    # -------------------------------------

    if not provider_call_id:

        print(
            "EVENT DOES NOT HAVE UUID"
        )

        return

    # -------------------------------------
    # Find Call in DB
    # -------------------------------------

    call = _get_call_by_provider_id(
        db,
        provider_call_id
    )

    if call is None:

        print(
            "CALL NOT FOUND FOR UUID:",
            provider_call_id
        )

        return

    now = _utc_now()

    # =====================================
    # Started
    # =====================================

    if vonage_status == "started":

        call.status = "initiating"

    # =====================================
    # Ringing
    # =====================================

    elif vonage_status == "ringing":

        call.status = "ringing"

    # =====================================
    # Answered
    # =====================================

    elif vonage_status == "answered":

        call.status = "in_progress"

    # =====================================
    # Busy
    # =====================================

    elif vonage_status == "busy":

        call.status = "busy"

        call.outcome = None

        call.ended_at = now

        _keep_ticket_pending(
            db,
            call
        )

    # =====================================
    # No answer
    # =====================================

    elif vonage_status in {
        "unanswered",
        "timeout",
    }:

        call.status = "no_answer"

        call.outcome = None

        call.ended_at = now

        _keep_ticket_pending(
            db,
            call
        )

    # =====================================
    # Failed
    # =====================================

    elif vonage_status in {
        "failed",
        "rejected",
        "cancelled",
    }:

        call.status = "failed"

        call.outcome = None

        call.ended_at = now

        _keep_ticket_pending(
            db,
            call
        )

    # =====================================
    # Completed
    # =====================================

    elif vonage_status == "completed":

        call.status = "completed"

        call.ended_at = now

        duration = data.get(
            "duration"
        )

        # Vonage may send duration
        if duration is not None:

            try:

                call.duration_seconds = int(
                    float(duration)
                )

            except (
                TypeError,
                ValueError,
            ):

                pass

        # Fallback if duration wasn't usable
        if (
            call.duration_seconds is None
            and call.started_at is not None
        ):

            call.duration_seconds = max(
                0,
                int(
                    (
                        now
                        - call.started_at
                    ).total_seconds()
                )
            )

    # =====================================
    # Unknown event
    # =====================================

    else:

        print(
            "IGNORING VONAGE STATUS:",
            vonage_status
        )

        return

    # =====================================
    # Save Call changes
    # =====================================

    db.commit()
    db.refresh(call)

    print("\n==============================")
    print("DATABASE CALL UPDATED")
    print("CALL ID:", call.id)
    print("STATUS:", call.status)
    print("OUTCOME:", call.outcome)
    print(
        "DURATION:",
        call.duration_seconds
    )
    print("==============================\n")