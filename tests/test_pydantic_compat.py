import collections.abc
import typing

import formatron.schemas.pydantic
from formatron.formatter import FormatterBuilder
from formatron.schemas.dict_inference import infer_mapping
from formatron.schemas.json_schema import create_schema
from formatron.schemas.schema import Schema


def test_formatter_import_smoke():
    formatter = FormatterBuilder()
    assert formatter is not None


def test_infer_mapping():
    schema_type = infer_mapping({"value": [1, "two"]})

    assert issubclass(schema_type, Schema)
    assert "value" in schema_type.fields()


def test_infer_mapping_empty_sequence():
    schema_type = infer_mapping({"value": []})
    annotation = schema_type.fields()["value"].annotation

    assert typing.get_origin(annotation) is collections.abc.Sequence
    assert typing.get_args(annotation) == (typing.Any,)


def test_infer_mapping_heterogeneous_sequence_preserves_order():
    schema_type = infer_mapping({"value": [1, "two", True]})
    annotation = schema_type.fields()["value"].annotation
    union_type = typing.get_args(annotation)[0]

    assert typing.get_origin(annotation) is collections.abc.Sequence
    assert typing.get_origin(union_type) is typing.Union
    assert typing.get_args(union_type) == (int, str, bool)


def test_array_metadata_preserves_item_type():
    schema_type = create_schema(
        {
            "$id": "https://example.com/array-metadata-preserves-item-type.json",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                }
            },
            "required": ["names"],
        }
    )
    annotation = schema_type.fields()["names"].annotation

    assert typing.get_origin(annotation.type) is list
    assert typing.get_args(annotation.type) == (str,)
    assert annotation.metadata["min_length"] == 1


def test_callable_schema_default_none():
    @formatron.schemas.pydantic.callable_schema
    def f(*, x: int | None = None):
        return x

    assert f.fields()["x"].required is False
    assert f.from_json("{}") is None


def test_callable_schema_from_json_defaults():
    @formatron.schemas.pydantic.callable_schema
    def add(a: int, b: int = 2, /, *, c: int = 3):
        return a + b + c

    assert add.from_json('{"a": 1}') == 6
