import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA is unavailable", allow_module_level=True)

rwkv_integration = pytest.importorskip("formatron.integrations.RWKV")
RWKV = pytest.importorskip("rwkv.model").RWKV

from pathlib import Path

from formatron.formatter import FormatterBuilder

RWKV_MODEL_PATH = Path(__file__).resolve().parent / "assets" / "RWKV-5-World-0.4B-v2-20231113-ctx4096.pth"


def test_pipeline_without_formatter_builder(monkeypatch):
    class DummyTokenizer:
        idx2token = {}

        @staticmethod
        def decode(tokens):
            return "".join(map(str, tokens))

    def fake_base_init(self, model, word_name):
        self.model = model
        self.tokenizer = DummyTokenizer()

    monkeypatch.setattr(rwkv_integration.rwkv.utils.PIPELINE, "__init__", fake_base_init)

    vocabulary_created = False

    def fake_create_engine_vocabulary(word_name, tokenizer):
        nonlocal vocabulary_created
        vocabulary_created = True
        return object()

    monkeypatch.setattr(rwkv_integration, "create_engine_vocabulary", fake_create_engine_vocabulary)

    pipeline = rwkv_integration.PIPELINE(object(), "rwkv_vocab_v20230424")

    assert pipeline.formatter is None
    assert vocabulary_created is False


@pytest.mark.skipif(not RWKV_MODEL_PATH.exists(), reason="RWKV assets are unavailable")
def test_rwkv_integration(snapshot):
    model = RWKV(str(RWKV_MODEL_PATH), 'cuda fp16')
    f = FormatterBuilder()
    f.append_line(f"Hello, RWKV!")
    pipeline = rwkv_integration.PIPELINE(model, "rwkv_vocab_v20230424", f)
    assert pipeline.generate("你好！") == snapshot(name="first_output")
    assert pipeline.generate("你好！") == snapshot(name="second_output")
