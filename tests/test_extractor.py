from formatron.extractor import ChoiceExtractor, LiteralExtractor


def test_literal_extractor_matches_only_at_current_position():
    extractor = LiteralExtractor("abc")

    assert extractor.extract("abcdef") == ("def", "abc")
    assert extractor.extract("zabcdef") is None


def test_choice_extractor_skips_literal_found_later_in_input():
    extractor = ChoiceExtractor(
        [LiteralExtractor("b"), LiteralExtractor("ab")],
        capture_name="value",
        nonterminal="choice",
    )

    assert extractor.extract("ab") == ("", "ab")
