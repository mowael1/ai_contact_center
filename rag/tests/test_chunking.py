"""Arabic/English/mixed sentence segmentation."""

import pytest

from rag.chunking.sentence_segmenter import split_sentences


# ---- sentence segmentation ------------------------------------------------
def test_english_sentence_splitting():
    text = "You can renew your bundle. Then pay via Vodafone Cash. Enjoy 30% extra!"
    assert len(split_sentences(text)) == 3


def test_arabic_sentence_splitting():
    text = "ازاي أجدد باقة الإنترنت؟ تقدر تجددها من التطبيق. السعر 50 جنيه."
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].endswith("؟")


def test_mixed_language_sentence_splitting():
    text = "Flex 70 بيديك 100 دقيقة. You can renew via *880#. مرحبا!"
    assert len(split_sentences(text)) == 3


def test_ussd_code_does_not_split():
    sentences = split_sentences("Dial *880*1# to renew. Then confirm.")
    assert len(sentences) == 2
    assert "*880*1#" in sentences[0]


def test_url_does_not_split():
    sentences = split_sentences("See https://web.vodafone.com.eg/en/flex for details. Thanks.")
    assert len(sentences) == 2
    assert "https://web.vodafone.com.eg/en/flex" in sentences[0]


def test_decimal_numbers_do_not_split():
    sentences = split_sentences("The bundle gives 1.5 GB and costs 70.50 EGP. Enjoy.")
    assert len(sentences) == 2
    assert "1.5" in sentences[0] and "70.50" in sentences[0]


def test_abbreviations_do_not_split():
    sentences = split_sentences("Dr. Ahmed approved it, e.g. for Flex. Done.")
    assert len(sentences) == 2


def test_bullet_lines_stay_separate():
    text = "- Activate subscriptions\n- WATCH IT on Flex 70\n- Anghami on Flex 100"
    assert len(split_sentences(text)) == 3


def test_empty_text_yields_no_sentences():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []
