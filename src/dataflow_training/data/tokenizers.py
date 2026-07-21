"""Tokenizer: one interface, two backends (hf-tokenizers, tiktoken).

CONTRACT: ``encode`` is deterministic (same text -> same ids; NO
truncation — length policy is the source's job); ``describe()`` is
JSON-clean (backend, name, vocab_size, eot_id) and rides the source's
describe() into run metadata. Construction fails loudly with an
install hint when the backing package (the [data] extra) is absent.

Spec strings (the ``tokenizer=`` source argument):
    gpt2                  -> tiktoken gpt2 (the historical default BPE)
    tiktoken:cl100k_base  -> any tiktoken encoding
    hf:openai-community/gpt2  -> any hub tokenizer (tokenizers lib)
"""
from __future__ import annotations

from typing import Protocol

INSTALL_HINT = ("— install the data extra: pip install -e '.[data]' "
                "(or uv pip install -e '.[data]')")


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def describe(self) -> dict: ...


class TiktokenTokenizer:
    def __init__(self, encoding: str = "gpt2"):
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(f"tiktoken not installed {INSTALL_HINT}") from exc
        self.encoding_name = encoding
        self.enc = tiktoken.get_encoding(encoding)
        specials = getattr(self.enc, "_special_tokens", {})
        self.eot_id = specials.get("<|endoftext|>")

    def encode(self, text: str) -> list[int]:
        return self.enc.encode_ordinary(text)

    def describe(self) -> dict:
        return {"backend": "tiktoken", "name": self.encoding_name,
                "vocab_size": self.enc.n_vocab, "eot_id": self.eot_id}


class HFTokenizer:
    def __init__(self, name: str):
        try:
            from tokenizers import Tokenizer as HFTok
        except ImportError as exc:
            raise ImportError(
                f"tokenizers not installed {INSTALL_HINT}") from exc
        self.name = name
        self.tok = HFTok.from_pretrained(name)
        self.eot_id = None
        for cand in ("<|endoftext|>", "</s>", "<|end_of_text|>",
                     "<|im_end|>"):
            got = self.tok.token_to_id(cand)
            if got is not None:
                self.eot_id = int(got)
                break

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def describe(self) -> dict:
        return {"backend": "hf", "name": self.name,
                "vocab_size": self.tok.get_vocab_size(),
                "eot_id": self.eot_id}


def resolve_tokenizer(spec: str) -> Tokenizer:
    """'gpt2' | 'tiktoken:ENC' | 'hf:NAME' -> a Tokenizer."""
    backend, sep, rest = spec.partition(":")
    if not sep:
        if spec == "gpt2":
            return TiktokenTokenizer("gpt2")
        return HFTokenizer(spec)
    if backend == "tiktoken":
        return TiktokenTokenizer(rest)
    if backend == "hf":
        return HFTokenizer(rest)
    raise ValueError(f"unknown tokenizer spec {spec!r} "
                     f"(forms: gpt2, tiktoken:ENC, hf:NAME)")
