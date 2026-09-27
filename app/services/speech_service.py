"""Speech-to-text and text-to-speech integration.

This module contains provider-specific speech logic only.
It does not call or depend on the RAG pipeline.
"""

from functools import lru_cache
import logging
from pathlib import Path

from groq import Groq

from app.core.config import settings


logger = logging.getLogger(__name__)


class SpeechServiceError(RuntimeError):
    """Base error raised by the speech service."""


class SpeechInputError(SpeechServiceError):
    """Raised when the supplied audio or text is invalid."""


class SpeechProviderError(SpeechServiceError):
    """Raised when the external speech provider fails."""


@lru_cache
def _get_client() -> Groq:
    if not settings.GROQ_API_KEY:
        raise SpeechProviderError(
            "GROQ_API_KEY is not configured"
        )

    return Groq(
        api_key=settings.GROQ_API_KEY
    )


def transcribe_audio(
    audio_bytes: bytes,
    *,
    filename: str = "recording.webm",
    content_type: str | None = None,
) -> str:
    """Transcribe uploaded audio without invoking the RAG pipeline."""

    if not audio_bytes:
        raise SpeechInputError(
            "Audio file is empty"
        )

    max_bytes = (
        settings.SPEECH_MAX_AUDIO_MB
        * 1024
        * 1024
    )

    if len(audio_bytes) > max_bytes:
        raise SpeechInputError(
            f"Audio file exceeds "
            f"{settings.SPEECH_MAX_AUDIO_MB} MB"
        )

    safe_filename = (
        Path(filename).name
        if filename
        else "recording.webm"
    )

    try:
        transcription = (
            _get_client()
            .audio
            .transcriptions
            .create(
                file=(
                    safe_filename,
                    audio_bytes,
                    content_type
                    or "application/octet-stream",
                ),
                model=settings.GROQ_STT_MODEL,
                language="ar",
                response_format="json",
                temperature=0.0,
            )
        )

        return transcription.text.strip()

    except SpeechServiceError:
        raise

    except Exception as exc:
        logger.exception(
            "Groq speech transcription failed"
        )

        raise SpeechProviderError(
            "Speech transcription provider failed"
        ) from exc


def synthesize_speech(
    text: str,
) -> bytes:
    """Generate Arabic WAV audio without modifying the text answer."""

    normalized_text = text.strip()

    if not normalized_text:
        raise SpeechInputError(
            "Text is empty"
        )

    if (
        len(normalized_text)
        > settings.SPEECH_MAX_TEXT_CHARS
    ):
        raise SpeechInputError(
            f"Text exceeds "
            f"{settings.SPEECH_MAX_TEXT_CHARS} characters"
        )

    try:
        response = (
            _get_client()
            .audio
            .speech
            .create(
                model=settings.GROQ_TTS_MODEL,
                voice=settings.GROQ_TTS_VOICE,
                input=normalized_text,
                response_format="wav",
            )
        )

        return response.read()

    except SpeechServiceError:
        raise

    except Exception as exc:
        logger.exception(
            "Groq speech synthesis failed"
        )

        raise SpeechProviderError(
            "Speech synthesis provider failed"
        ) from exc
