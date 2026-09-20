from pathlib import Path

from vonage import Auth, Vonage
from vonage_voice import (
    CreateCallRequest,
    Phone,
    ToPhone,
)

from app.core.config import settings


# =========================================
# Project paths
# =========================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# =========================================
# Helpers
# =========================================

def _get_private_key() -> str:
    """
    Read the Vonage private key.

    If the path in .env is relative,
    it will be resolved from the project root.
    """

    private_key_path = Path(
        settings.VONAGE_PRIVATE_KEY_PATH
    )

    if not private_key_path.is_absolute():
        private_key_path = (
            PROJECT_ROOT
            / private_key_path
        )

    if not private_key_path.exists():
        raise FileNotFoundError(
            f"Vonage private key not found: "
            f"{private_key_path}"
        )

    return private_key_path.read_text(
        encoding="utf-8"
    )


def _normalize_phone_number(
    phone_number: str,
) -> str:
    """
    Convert the phone number to digits only.

    Example:
    +20 106 030 2906
    ->
    201060302906
    """

    normalized = (
        phone_number
        .strip()
        .replace("+", "")
        .replace(" ", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
    )

    if not normalized.isdigit():
        raise ValueError(
            "Invalid phone number"
        )

    return normalized


def _answer_url() -> str:
    """
    Vonage calls this endpoint
    after the customer answers.
    """

    base_url = (
        settings
        .PUBLIC_BASE_URL
        .rstrip("/")
    )

    return (
        f"{base_url}"
        f"/api/v1/vonage/answer"
    )


def _build_client() -> Vonage:

    private_key = (
        _get_private_key()
    )

    return Vonage(
        Auth(
            application_id=(
                settings
                .VONAGE_APPLICATION_ID
            ),
            private_key=private_key,
        )
    )


# =========================================
# Public integration function
# =========================================

def create_outbound_call(
    phone_number: str,
) -> str:
    """
    Create a real outbound Vonage call.

    Returns:
        Vonage call UUID
        which will be stored later in:
        calls.provider_call_id
    """

    to_number = (
        _normalize_phone_number(
            phone_number
        )
    )

    from_number = (
        _normalize_phone_number(
            settings.VONAGE_NUMBER
        )
    )

    client = _build_client()

    print("\n==============================")
    print("CREATING VONAGE CALL")
    print("TO:", to_number)
    print("FROM:", from_number)
    print("==============================\n")

    response = (
        client.voice.create_call(
            CreateCallRequest(
                answer_url=[
                    _answer_url()
                ],

                to=[
                    ToPhone(
                        number=to_number
                    )
                ],

                from_=Phone(
                    number=from_number
                ),
            )
        )
    )

    response_data = (
        response.model_dump()
    )

    print("\n==============================")
    print("VONAGE CALL CREATED")
    print("RESPONSE:", response_data)
    print("==============================\n")

    call_uuid = (
        response_data.get("uuid")
    )

    if not call_uuid:
        raise RuntimeError(
            "Vonage did not return "
            "a call UUID"
        )

    return call_uuid