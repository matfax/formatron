import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)

pytest.importorskip("formatron.integrations.RWKV")
RWKV = pytest.importorskip("rwkv.model").RWKV
np = pytest.importorskip("numpy")

from pathlib import Path
from typing import Literal
from formatron.schemas import json_schema
from formatron.schemas.dict_inference import infer_mapping
from formatron.formatter import FormatterBuilder
import formatron.schemas.pydantic
import formatron

RWKV_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "RWKV-5-World-0.4B-v2-20231113-ctx4096.pth"


class Test(formatron.schemas.pydantic.ClassSchema):
    name: str
    weight: float
    color: str


def test_formatter(snapshot, normalize_for_snapshot):
    FormatterBuilder._formatter_builder_counter = 0
    f = FormatterBuilder()
    a = f.choose('railroad', 'orange', 'banana', capture_name='food')
    f.append_line(
        f"Today, I want to eat {a}")
    f.append_str(
        f"My food's ID is {f.choose(f.regex('[0-9]+'), f.regex('[a-z]+'), capture_name='ID')}.\n")
    f.append_multiline_str("""
                            What's more, indentations
                            are handled
                            appropriately.""")
    f.append_line(
        f"My weight is 14.4kg and my color is pink. This is my personal info json: {f.json(Test, capture_name='json')}")
    model = RWKV(
        str(RWKV_MODEL_PATH), 'cuda fp16')
    pipeline = formatron.integrations.RWKV.PIPELINE(model, "rwkv_vocab_v20230424", f)
    np.random.seed(42)
    assert pipeline.formatter.grammar_str == snapshot(name="grammar")
    assert pipeline.generate(
        "My name is Van. ",
        token_count=256,
        args=formatron.integrations.RWKV.PIPELINE_ARGS(top_p=0.5),
    ) == snapshot(name="output")
    assert normalize_for_snapshot(pipeline.formatter.captures) == snapshot(name="captures")


def test_formatter_str(snapshot, normalize_for_snapshot):
    FormatterBuilder._formatter_builder_counter = 0
    f = FormatterBuilder()
    f.append_line(f"{f.str(stop=['.'])}")
    model = RWKV(
        str(RWKV_MODEL_PATH), 'cuda fp16')
    pipeline = formatron.integrations.RWKV.PIPELINE(model, "rwkv_vocab_v20230424", f)
    np.random.seed(42)
    assert pipeline.formatter.grammar_str == snapshot(name="grammar")
    assert pipeline.generate(
        "My name is Van. ",
        token_count=256,
        args=formatron.integrations.RWKV.PIPELINE_ARGS(top_p=0.5),
    ) == snapshot(name="output")
    assert normalize_for_snapshot(pipeline.formatter.captures) == snapshot(name="captures")

def test_formatter_substr(snapshot, normalize_for_snapshot):
    FormatterBuilder._formatter_builder_counter = 0
    f = FormatterBuilder()
    f.append_str(f"{f.substr('Name: Umbrella; Price: 114.514 dollars;', extract_empty_substring=True, capture_name='substr')}<eos>")
    model = RWKV(
        str(RWKV_MODEL_PATH), 'cuda fp16')
    pipeline = formatron.integrations.RWKV.PIPELINE(model, "rwkv_vocab_v20230424", f)
    np.random.seed(42)
    assert pipeline.formatter.grammar_str == snapshot(name="grammar")
    assert pipeline.generate(
        "Umbrella Price: 114.514 dollars. The price of the umbrella is",
        token_count=256,
        args=formatron.integrations.RWKV.PIPELINE_ARGS(top_p=0.5),
    ) == snapshot(name="output")
    assert normalize_for_snapshot(pipeline.formatter.captures) == snapshot(name="captures")


def test_formatter_dict_inference(snapshot, normalize_for_snapshot):
    FormatterBuilder._formatter_builder_counter = 0
    f = FormatterBuilder()
    f.append_line(
        f"{f.json(infer_mapping({'name': 'xxx', 'gender': 'xxx'}), capture_name='json')}")
    model = RWKV(
        str(RWKV_MODEL_PATH), 'cuda fp16')
    pipeline = formatron.integrations.RWKV.PIPELINE(model, "rwkv_vocab_v20230424", f)
    np.random.seed(42)
    assert pipeline.formatter.grammar_str == snapshot(name="grammar")
    assert pipeline.generate(
        "This is a random json: ",
        token_count=256,
        args=formatron.integrations.RWKV.PIPELINE_ARGS(top_p=0.5),
    ) == snapshot(name="output")
    assert normalize_for_snapshot(pipeline.formatter.captures) == snapshot(name="captures")

def test_formatter_json_schema(snapshot, normalize_for_snapshot):
    FormatterBuilder._formatter_builder_counter = 0
    f = FormatterBuilder()
    schema = {
        "$id": "https://example.com/person.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            },
            "age": {
                "type": "integer"
            }
        },
        "required": ["name", "age"]
    }
    schema = json_schema.create_schema(schema)
    f.append_line(
        f"{f.json(schema, capture_name='json')}")
    model = RWKV(
        str(RWKV_MODEL_PATH), 'cuda fp16')
    pipeline = formatron.integrations.RWKV.PIPELINE(model, "rwkv_vocab_v20230424", f)
    np.random.seed(42)
    assert pipeline.generate(
        "This is a random json: ",
        token_count=256,
        args=formatron.integrations.RWKV.PIPELINE_ARGS(top_p=0.5),
    ) == snapshot(name="output")
    assert normalize_for_snapshot(pipeline.formatter.captures) == snapshot(name="captures")
    assert pipeline.formatter.grammar_str == snapshot(name="grammar")

def test_formatter_top_level_array_json_schema(snapshot, normalize_for_snapshot):
    FormatterBuilder._formatter_builder_counter = 0
    f = FormatterBuilder()
    schema = {
        "$id": "https://example.com/array.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
                "active": {"type": "boolean"}
            },
            "required": ["id", "name"]
        },
        "minItems": 1,
        "maxItems": 5
    }
    schema = json_schema.create_schema(schema)
    f.append_line(f"{f.json(schema, capture_name='json')}")
    model = RWKV(
        str(RWKV_MODEL_PATH), 'cuda fp16')
    pipeline = formatron.integrations.RWKV.PIPELINE(model, "rwkv_vocab_v20230424", f)
    np.random.seed(42)
    assert pipeline.formatter.grammar_str == snapshot(name="grammar")
    assert pipeline.generate(
        "Generate a JSON array of users: ",
        token_count=256,
        args=formatron.integrations.RWKV.PIPELINE_ARGS(top_p=0.5),
    ) == snapshot(name="output")
    assert normalize_for_snapshot(pipeline.formatter.captures) == snapshot(name="captures")


def test_formatter_callable_schema(snapshot, normalize_for_snapshot):
    @formatron.schemas.pydantic.callable_schema
    def add(a: int, b: int, /, *, c: int):
        return a + b + c

    FormatterBuilder._formatter_builder_counter = 0
    f = FormatterBuilder()
    f.append_line(
        f"{f.json(add, capture_name='json')}")
    model = RWKV(
        str(RWKV_MODEL_PATH), 'cuda fp16')
    pipeline = formatron.integrations.RWKV.PIPELINE(model, "rwkv_vocab_v20230424", f)
    np.random.seed(42)
    assert pipeline.formatter.grammar_str == snapshot(name="grammar")
    assert pipeline.generate(
        "This is a random json: ",
        token_count=256,
        args=formatron.integrations.RWKV.PIPELINE_ARGS(top_p=0.5),
    ) == snapshot(name="output")
    assert normalize_for_snapshot(pipeline.formatter.captures) == snapshot(name="captures")

def test_grammar_literal(snapshot, normalize_for_snapshot):
    FormatterBuilder._formatter_builder_counter = 0
    f = FormatterBuilder()
    class A(formatron.schemas.pydantic.ClassSchema):
        a: Literal['114', '514']
    f.append_line(
        f"{f.json(A, capture_name='json')}")
    model = RWKV(
        str(RWKV_MODEL_PATH), 'cuda fp16')
    pipeline = formatron.integrations.RWKV.PIPELINE(model, "rwkv_vocab_v20230424", f)
    np.random.seed(42)
    assert pipeline.formatter.grammar_str == snapshot(name="grammar")
    assert pipeline.generate(
        "This is a random json: ",
        token_count=256,
        args=formatron.integrations.RWKV.PIPELINE_ARGS(top_p=0.5),
    ) == snapshot(name="output")
    assert normalize_for_snapshot(pipeline.formatter.captures) == snapshot(name="captures")


def test_formatter_alternate_accept(snapshot, normalize_for_snapshot):
    FormatterBuilder._formatter_builder_counter = 0
    f = FormatterBuilder()
    f.append_str(f"Name: {f.str(stop=[','], capture_name='name')}")
    f.append_str(f"Age: {f.regex('[0-9]+', capture_name='age')}")

    model = RWKV(
        str(RWKV_MODEL_PATH), 'cuda fp16')
    pipeline = formatron.integrations.RWKV.PIPELINE(model, "rwkv_vocab_v20230424", f)
    
    formatter = pipeline.formatter
    
    # Simulate alternating between accept_token and accept_bytes
    tokens = pipeline.tokenizer.encode("Name: John,")
    for token in tokens:
        formatter.accept_token(token)
    formatter.accept_bytes(b"Age: ")
    tokens = pipeline.tokenizer.encode("30")
    for token in tokens:
        formatter.accept_token(token)
    
    assert normalize_for_snapshot(formatter.captures) == snapshot(name="captures")


def test_formatter_regex_complement(snapshot, normalize_for_snapshot):
    FormatterBuilder._formatter_builder_counter = 0
    f = FormatterBuilder()
    f.append_str(f"Text: {f.regex_complement('[0-9]', capture_name='non_numeric')}")
    f.append_line(f"Number: {f.regex('[0-9]+', capture_name='numeric')}")

    model = RWKV(
        str(RWKV_MODEL_PATH), 'cuda fp16')
    pipeline = formatron.integrations.RWKV.PIPELINE(model, "rwkv_vocab_v20230424", f)
    
    np.random.seed(42)
    assert pipeline.formatter.grammar_str == snapshot(name="grammar")
    assert pipeline.generate(
        "Here's some text followed by an integer: ",
        token_count=256,
        args=formatron.integrations.RWKV.PIPELINE_ARGS(top_p=0.5),
    ) == snapshot(name="output")
    assert normalize_for_snapshot(pipeline.formatter.captures) == snapshot(name="captures")

    # Test with manual input
    formatter = pipeline.formatter
    formatter.reset()
    
    input_text = "Text: Hello, world! Number: 42\n"
    for char in input_text:
        formatter.accept_bytes(char.encode('utf-8'))
    
    assert normalize_for_snapshot(formatter.captures) == snapshot(name="manual_captures")

def test_formatter_json_no_properties(snapshot, normalize_for_snapshot):
    import typing
    FormatterBuilder._formatter_builder_counter = 0
    f = FormatterBuilder()
    f.append_str(f"{f.json(typing.Dict[str, typing.Any], capture_name='data')}")

    model = RWKV(
        str(RWKV_MODEL_PATH), 'cuda fp16')
    pipeline = formatron.integrations.RWKV.PIPELINE(model, "rwkv_vocab_v20230424", f)
    
    formatter = pipeline.formatter
    
    # Test with manual input
    input_text = '{"key": "value", "number": 42}'
    for char in input_text:
        formatter.accept_bytes(char.encode('utf-8'))
    
    assert normalize_for_snapshot(formatter.captures) == snapshot(name="captures")

def test_utf8_json_key(snapshot, normalize_for_snapshot):
    FormatterBuilder._formatter_builder_counter = 0
    f = FormatterBuilder()
    schema = json_schema.create_schema({
        "$id": "https://example.com/array.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "土豆": {"type": "string"},
            "\(@^0^@)/": {"type": "string"},
            "🍎": {"type": "string"},
        }
    })
    f.append_line(f"{f.json(schema, capture_name='json')}")
    model = RWKV(
        str(RWKV_MODEL_PATH), 'cuda fp16')
    pipeline = formatron.integrations.RWKV.PIPELINE(model, "rwkv_vocab_v20230424", f)
    np.random.seed(42)
    assert pipeline.formatter.grammar_str == snapshot(name="grammar")
    assert pipeline.generate(
        "This is a random json: ",
        token_count=256,
        args=formatron.integrations.RWKV.PIPELINE_ARGS(top_p=0.5),
    ) == snapshot(name="output")
    assert normalize_for_snapshot(pipeline.formatter.captures) == snapshot(name="captures")
