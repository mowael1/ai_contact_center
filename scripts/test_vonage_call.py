from pprint import pprint

from vonage import Auth, Vonage
from vonage_voice import (
    CreateCallRequest,
    Phone,
    ToPhone,
)

from app.core.config import settings


def main():
    with open(
        settings.VONAGE_PRIVATE_KEY_PATH,
        "r",
        encoding="utf-8"
    ) as key_file:
        private_key = key_file.read()

    client = Vonage(
        Auth(
            application_id=settings.VONAGE_APPLICATION_ID,
            private_key=private_key,
        )
    )

    response = client.voice.create_call(
        CreateCallRequest(
            answer_url=[
                settings.VOICE_ANSWER_URL
            ],
            to=[
                ToPhone(
                    number=settings.VOICE_TO_NUMBER
                )
            ],
            from_=Phone(
                number=settings.VONAGE_NUMBER
            ),
        )
    )

    print("\nVonage call response:\n")

    pprint(
        response.model_dump()
    )


if __name__ == "__main__":
    main()