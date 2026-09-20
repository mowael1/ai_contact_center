from typing import Any

from app.core.config import settings
from app.services.llm_service import (
    classify_follow_up_response,
)


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
# URL helpers
# =========================================

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
# Speech processing
# =========================================

def process_speech_input(
    *,
    data: dict[str, Any],
    attempt: int,
) -> list[dict[str, Any]]:

    print("\n==============================")
    print("VONAGE SPEECH INPUT")
    print("ATTEMPT:", attempt)
    print("==============================")

    text, confidence = (
        _extract_best_speech_result(
            data
        )
    )

    print("TRANSCRIPT:", text)
    print("STT CONFIDENCE:", confidence)

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

        print("FINAL OUTCOME: unclear")

        return build_unclear_final_ncco()

    # =====================================
    # Low STT confidence
    # =====================================

    if (
        confidence is not None
        and confidence
        < STT_CONFIDENCE_THRESHOLD
    ):

        print(
            "LOW STT CONFIDENCE"
        )

        if attempt < MAX_ATTEMPTS:

            print(
                "ACTION: ASK CUSTOMER TO REPEAT"
            )

            return build_repeat_unclear_ncco(
                attempt=attempt + 1
            )

        print(
            "FINAL OUTCOME: unclear"
        )

        return build_unclear_final_ncco()

    # =====================================
    # Send transcript to Groq LLM
    # =====================================

    print("\nSENDING TRANSCRIPT TO GROQ...")

    outcome = (
        classify_follow_up_response(
            text
        )
    )

    print("LLM OUTCOME:", outcome)

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

        # Later:
        #
        # call.status = "completed"
        # call.outcome = "resolved"
        #
        # ticket.status = "resolved"

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

        # Later:
        #
        # call.status = "completed"
        # call.outcome = "not_resolved"
        #
        # ticket.status = "needs_agent"

        return build_not_resolved_ncco()

    # =====================================
    # Unclear
    # =====================================

    print(
        "LLM OUTCOME IS UNCLEAR"
    )

    if attempt < MAX_ATTEMPTS:

        print(
            "ACTION: REDIRECT CUSTOMER "
            "BACK TO THE QUESTION"
        )

        return build_redirect_off_topic_ncco(
            attempt=attempt + 1
        )

    print(
        "FINAL OUTCOME: unclear"
    )

    print(
        "ACTION: SEND TO HUMAN AGENT"
    )

    # Later:
    #
    # call.status = "completed"
    # call.outcome = "unclear"
    #
    # ticket.status = "needs_agent"

    return build_unclear_final_ncco()


# =========================================
# Vonage call events
# =========================================

def process_call_event(
    data: dict[str, Any],
) -> None:

    call_uuid = data.get(
        "uuid"
    )

    status = data.get(
        "status"
    )

    detail = data.get(
        "detail"
    )

    print("\n==============================")
    print("VONAGE EVENT")
    print("UUID:", call_uuid)
    print("STATUS:", status)
    print("DETAIL:", detail)
    print("==============================\n")

    # =====================================
    # Database integration comes later
    # =====================================
    #
    # Example:
    #
    # call = call_repository.get_by_provider_call_id(
    #     provider_call_id=call_uuid
    # )
    #
    # ringing
    # → call.status = "ringing"
    #
    # answered
    # → call.status = "in_progress"
    #
    # completed
    # → call.status = "completed"
    #
    # rejected / failed
    # → call.status = "failed"
    #
    # no_answer
    # → call.status = "no_answer"