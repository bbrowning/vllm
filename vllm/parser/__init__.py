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
    "deepseek_v4_grammar": (
        "vllm.grammar_parser.parsers.deepseek_v4",
        "DeepSeekV4GrammarParser",
    ),
    "gemma4_grammar": (
        "vllm.grammar_parser.parsers.gemma4",
        "Gemma4GrammarParser",
    ),
    "qwen3_grammar": (
        "vllm.grammar_parser.parsers.qwen3",
        "Qwen3GrammarParser",
    ),
    "qwen3_xml_grammar": (
        "vllm.grammar_parser.parsers.qwen3",
        "Qwen3XMLGrammarParser",
    ),
    "qwen3_coder_grammar": (
        "vllm.grammar_parser.parsers.qwen3",
        "Qwen3XMLGrammarParser",
    ),
    "nemotron_v3_grammar": (
        "vllm.grammar_parser.parsers.nemotron_v3",
        "NemotronV3GrammarParser",
    ),
}

_ADAPTER_MODULE = "vllm.grammar_parser.registered_adapters"

_GRAMMAR_REASONING_ADAPTERS_TO_REGISTER = {
    name: (_ADAPTER_MODULE, f"{cls}ReasoningAdapter")
    for name, (_, cls) in _GRAMMAR_PARSERS_TO_REGISTER.items()
}

_GRAMMAR_TOOL_ADAPTERS_TO_REGISTER = {
    name: (_ADAPTER_MODULE, f"{cls}ToolAdapter")
    for name, (_, cls) in _GRAMMAR_PARSERS_TO_REGISTER.items()
}


def register_lazy_parsers():
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager

    for name, (file_name, class_name) in _PARSERS_TO_REGISTER.items():
        module_path = f"vllm.parser.{file_name}"
        ParserManager.register_lazy_module(name, module_path, class_name)
    for name, (module_path, class_name) in _GRAMMAR_PARSERS_TO_REGISTER.items():
        ParserManager.register_lazy_module(name, module_path, class_name)
    for name, (
        module_path,
        class_name,
    ) in _GRAMMAR_REASONING_ADAPTERS_TO_REGISTER.items():
        ReasoningParserManager.register_lazy_module(name, module_path, class_name)
    for name, (module_path, class_name) in _GRAMMAR_TOOL_ADAPTERS_TO_REGISTER.items():
        ToolParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_parsers()
