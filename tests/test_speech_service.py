from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.core.config import settings
from app.services import speech_service
from app.services.speech_service import (
    SpeechInputError,
    SpeechProviderError,
)


def make_client(
    *,
    transcription=None,
    audio_bytes=b"RIFF-test-wav",
):
    transcription_create = Mock(
        return_value=SimpleNamespace(
            text=transcription or ""
        )
    )

    speech_response = Mock()
    speech_response.read.return_value = audio_bytes

    speech_create = Mock(
        return_value=speech_response
    )

    client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(
                create=transcription_create
            ),
            speech=SimpleNamespace(
                create=speech_create
            ),
        )
    )

    return (
        client,
        transcription_create,
        speech_create,
        speech_response,
    )


def test_transcribe_rejects_empty_audio():
    with pytest.raises(
        SpeechInputError,
        match="Audio file is empty",
    ):
        speech_service.transcribe_audio(b"")


def test_transcribe_rejects_oversized_audio(
    monkeypatch,
):
    monkeypatch.setattr(
        settings,
        "SPEECH_MAX_AUDIO_MB",
        0,
    )

    with pytest.raises(
        SpeechInputError,
        match="Audio file exceeds",
    ):
        speech_service.transcribe_audio(b"audio")


def test_transcribe_calls_groq_and_returns_text(
    monkeypatch,
):
    (
        client,
        transcription_create,
        _,
        _,
    ) = make_client(
        transcription="  المشكلة اتحلت  "
    )

    monkeypatch.setattr(
        speech_service,
        "_get_client",
        Mock(return_value=client),
    )

    result = speech_service.transcribe_audio(
        b"fake-webm-data",
        filename="../recording.webm",
        content_type="audio/webm",
    )

    assert result == "المشكلة اتحلت"

    transcription_create.assert_called_once_with(
        file=(
            "recording.webm",
            b"fake-webm-data",
            "audio/webm",
        ),
        model=settings.GROQ_STT_MODEL,
        language="ar",
        response_format="json",
        temperature=0.0,
    )


def test_transcribe_wraps_provider_failure(
    monkeypatch,
):
    client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(
                create=Mock(
                    side_effect=RuntimeError(
                        "Provider unavailable"
                    )
                )
            )
        )
    )

    monkeypatch.setattr(
        speech_service,
        "_get_client",
        Mock(return_value=client),
    )

    with pytest.raises(
        SpeechProviderError,
        match="Speech transcription provider failed",
    ):
        speech_service.transcribe_audio(
            b"fake-audio",
        )


def test_synthesize_rejects_empty_text():
    with pytest.raises(
        SpeechInputError,
        match="Text is empty",
    ):
        speech_service.synthesize_speech("   ")


def test_synthesize_calls_groq_and_returns_wav(
    monkeypatch,
):
    (
        client,
        _,
        speech_create,
        speech_response,
    ) = make_client(
        audio_bytes=b"RIFF-generated-wav"
    )

    monkeypatch.setattr(
        speech_service,
        "_get_client",
        Mock(return_value=client),
    )

    result = speech_service.synthesize_speech(
        "  أهلاً وسهلاً  "
    )

    assert result == b"RIFF-generated-wav"

    speech_create.assert_called_once_with(
        model=settings.GROQ_TTS_MODEL,
        voice=settings.GROQ_TTS_VOICE,
        input="أهلاً وسهلاً",
        response_format="wav",
    )

    speech_response.read.assert_called_once_with()


def test_synthesize_wraps_provider_failure(
    monkeypatch,
):
    client = SimpleNamespace(
        audio=SimpleNamespace(
            speech=SimpleNamespace(
                create=Mock(
                    side_effect=RuntimeError(
                        "Provider unavailable"
                    )
                )
            )
        )
    )

    monkeypatch.setattr(
        speech_service,
        "_get_client",
        Mock(return_value=client),
    )

    with pytest.raises(
        SpeechProviderError,
        match="Speech synthesis provider failed",
    ):
        speech_service.synthesize_speech(
            "اختبار"
        )
