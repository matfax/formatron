from formatron.formatter import FormatterBuilder
from formatron.formats.json import JsonExtractor, from_str_to_kbnf_str
from formatron.schemas import json_schema


def test_json_extractor_returns_none_for_balanced_but_invalid_json():
    schema = json_schema.create_schema(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.com/person.json",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
    )

    extractor = FormatterBuilder().json(schema, capture_name="json")

    assert extractor.extract('{"name": }') is None


def test_json_extractor_still_returns_parsed_object_for_valid_json():
    schema = json_schema.create_schema(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.com/person.json",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
    )

    extractor = FormatterBuilder().json(schema, capture_name="json")

    assert extractor.extract('{"name": "Tabby"} trailing') == (" trailing", {"name": "Tabby"})


def test_json_extractor_allows_choice_backtracking_after_decode_failure():
    schema = json_schema.create_schema(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.com/person.json",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
    )

    extractor = FormatterBuilder().choose(
        FormatterBuilder().json(schema, capture_name="json"),
        '{"name": }',
        capture_name="value",
    )

    assert extractor.extract('{"name": } trailing') == (" trailing", '{"name": }')


def test_json_schema_escapes_utf8_and_regex_significant_keys():
    special_key_literal = from_str_to_kbnf_str(r"\(@^0^@)/")

    schema = json_schema.create_schema(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://example.com/utf8-keys.json",
            "type": "object",
            "properties": {
                "土豆": {"type": "string"},
                r"\(@^0^@)/": {"type": "string"},
                "🍎": {"type": "string"},
            },
        }
    )

    definition = JsonExtractor("start", "json", schema, lambda x: x).kbnf_definition

    assert special_key_literal in definition
    assert r'"\u571f\u8c46"' not in definition
