# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.abstract_parser import (
    DelegatingParser,
    Parser,
    _WrappedParser,
)
from vllm.parser.parser_manager import ParserManager

__all__ = [
    "Parser",
    "DelegatingParser",
    "ParserManager",
    "_WrappedParser",
]

_PARSERS_TO_REGISTER = {
    "minimax_m2": (  # name
        "minimax_m2_parser",  # filename
        "MiniMaxM2Parser",  # class_name
    ),
}

_GRAMMAR_PARSERS_TO_REGISTER = {
    "gemma4_grammar": (
        "vllm.grammar_parser.unified_parsers",
        "Gemma4GrammarParser",
    ),
    "qwen3_grammar": (
        "vllm.grammar_parser.unified_parsers",
        "Qwen3GrammarParser",
    ),
}


def register_lazy_parsers():
    for name, (file_name, class_name) in _PARSERS_TO_REGISTER.items():
        module_path = f"vllm.parser.{file_name}"
        ParserManager.register_lazy_module(name, module_path, class_name)
    for name, (module_path, class_name) in _GRAMMAR_PARSERS_TO_REGISTER.items():
        ParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_parsers()
