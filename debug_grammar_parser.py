# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Debug script for grammar-based tool/reasoning parsers.

Usage:
    python debug_grammar_parser.py --model google/gemma-4-31b-it

This script:
1. Loads the real tokenizer for the model
2. Instantiates the grammar tool and reasoning parsers
3. Feeds sample model outputs through both streaming and non-streaming paths
4. Prints every semantic event and delta message so you can see what's happening

You can also paste real model output into REAL_MODEL_OUTPUT below.
"""

import argparse
import json
from unittest.mock import MagicMock

from transformers import AutoTokenizer

from vllm.grammar_parser.registered_parsers import (
    GrammarGemma4ReasoningParser,
    GrammarGemma4ToolParser,
)

# Paste real model output here if you have it, or pass via --output flag
REAL_MODEL_OUTPUT = ""

# Known-good synthetic examples for Gemma4 format
SYNTHETIC_EXAMPLES = {
    "tool_call": (
        '<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|>'
    ),
    "tool_call_with_reasoning": (
        "<|channel>thought\nLet me check the weather for the user.\n<channel|>"
        '<|tool_call>call:get_weather{location:<|"|>London<|"|>}'
        "<tool_call|>"
    ),
    "multi_tool": (
        '<|tool_call>call:get_weather{location:<|"|>London<|"|>}'
        "<tool_call|>"
        '<|tool_call>call:get_time{location:<|"|>London<|"|>}'
        "<tool_call|>"
    ),
    "plain_text": "Hello, how can I help you today?",
    "reasoning_only": (
        "<|channel>thought\nThe user wants help with something.\n"
        "<channel|>Sure, I can help!"
    ),
}


def debug_non_streaming(parser, text, label=""):
    print(f"\n{'=' * 60}")
    print(f"NON-STREAMING: {label}")
    print(f"Input text ({len(text)} chars): {text!r}")
    print(f"{'=' * 60}")

    mock_request = MagicMock()
    mock_request.tools = []
    mock_request.tool_choice = "auto"

    if hasattr(parser, "extract_tool_calls"):
        result = parser.extract_tool_calls(text, mock_request)
        print(f"  tools_called: {result.tools_called}")
        print(f"  content: {result.content!r}")
        for i, tc in enumerate(result.tool_calls):
            name = tc.function.name
            args = tc.function.arguments
            print(f"  tool_call[{i}]: name={name!r} args={args!r}")
    elif hasattr(parser, "extract_reasoning"):
        reasoning, content = parser.extract_reasoning(text, mock_request)
        print(f"  reasoning: {reasoning!r}")
        print(f"  content: {content!r}")


def debug_streaming(parser, text, tokenizer, label=""):
    print(f"\n{'=' * 60}")
    print(f"STREAMING: {label}")
    print(f"Input text ({len(text)} chars): {text!r}")
    print(f"{'=' * 60}")

    mock_request = MagicMock()
    mock_request.tools = []
    mock_request.tool_choice = "auto"

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    decoded_tokens = []
    for tid in token_ids:
        decoded_tokens.append(tokenizer.decode([tid]))

    print(f"  Token count: {len(token_ids)}")
    tok_list = list(zip(token_ids, decoded_tokens))[:20]
    suffix = "..." if len(token_ids) > 20 else ""
    print(f"  Tokens: {tok_list}{suffix}")

    # Check which special tokens the parser knows about
    vocab = tokenizer.get_vocab()
    if hasattr(parser, "grammar_config"):
        print("\n  Token ID terminal mappings:")
        for term_name, term_text in parser.grammar_config.token_id_terminals.items():
            tid = vocab.get(term_text)
            print(f"    {term_name}: {term_text!r} -> token_id={tid}")

    # Feed token by token
    previous_text = ""
    previous_token_ids = []
    all_deltas = []

    for i, (tid, tok_text) in enumerate(zip(token_ids, decoded_tokens)):
        current_text = previous_text + tok_text
        current_token_ids = previous_token_ids + [tid]

        if hasattr(parser, "extract_tool_calls_streaming"):
            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=tok_text,
                previous_token_ids=tuple(previous_token_ids),
                current_token_ids=tuple(current_token_ids),
                delta_token_ids=(tid,),
                request=mock_request,
            )
        elif hasattr(parser, "extract_reasoning_streaming"):
            delta = parser.extract_reasoning_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=tok_text,
                previous_token_ids=tuple(previous_token_ids),
                current_token_ids=tuple(current_token_ids),
                delta_token_ids=(tid,),
            )
        else:
            delta = None

        if delta is not None:
            all_deltas.append(delta)
            parts = []
            if delta.content:
                parts.append(f"content={delta.content!r}")
            if hasattr(delta, "reasoning") and delta.reasoning:
                parts.append(f"reasoning={delta.reasoning!r}")
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_parts = [f"idx={tc.index}"]
                    if tc.id:
                        tc_parts.append(f"id={tc.id!r}")
                    if tc.function and tc.function.name:
                        tc_parts.append(f"name={tc.function.name!r}")
                    if tc.function and tc.function.arguments:
                        tc_parts.append(f"args={tc.function.arguments!r}")
                    parts.append(f"tool_call({', '.join(tc_parts)})")
            print(f"  [{i:3d}] token={tid:6d} {tok_text!r:20s} -> {'; '.join(parts)}")

        previous_text = current_text
        previous_token_ids = current_token_ids

    if not all_deltas:
        print("  ** NO DELTAS EMITTED **")

    # Reconstruct final state
    print("\n  Summary:")
    content = "".join(d.content or "" for d in all_deltas)
    reasoning = "".join(getattr(d, "reasoning", None) or "" for d in all_deltas)
    tool_names = []
    tool_args = {}
    for d in all_deltas:
        if d.tool_calls:
            for tc in d.tool_calls:
                if tc.function and tc.function.name:
                    tool_names.append(tc.function.name)
                if tc.function and tc.function.arguments:
                    tool_args.setdefault(tc.index, []).append(tc.function.arguments)

    print(f"    content: {content!r}")
    print(f"    reasoning: {reasoning!r}")
    print(f"    tool_names: {tool_names}")
    for idx, arg_parts in tool_args.items():
        combined = "".join(arg_parts)
        print(f"    tool[{idx}] args: {combined!r}")
        try:
            print(f"    tool[{idx}] parsed: {json.loads(combined)}")
        except json.JSONDecodeError as e:
            print(f"    tool[{idx}] JSON ERROR: {e}")


def debug_engine_events(parser, text, tokenizer, label=""):
    """Show raw semantic events from the engine for detailed debugging."""
    from vllm.grammar_parser.parser_engine import StreamingParserEngine

    print(f"\n{'=' * 60}")
    print(f"RAW ENGINE EVENTS: {label}")
    print(f"{'=' * 60}")

    engine = StreamingParserEngine(parser.grammar_config, tokenizer)

    # Non-streaming: all events at once
    events = engine.parse_complete(text)
    for i, event in enumerate(events):
        etype = event.type.name
        tidx = event.tool_index
        val = event.value
        print(f"  [{i:3d}] {etype:20s} tool_idx={tidx} value={val!r}")

    if not events:
        print("  ** NO EVENTS **")


def main():
    parser = argparse.ArgumentParser(description="Debug grammar parsers")
    parser.add_argument("--model", default="google/gemma-4-31b-it")
    parser.add_argument("--output", help="File containing raw model output to debug")
    parser.add_argument("--text", help="Raw model output text to debug (inline)")
    args = parser.parse_args()

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"Vocab size: {len(tokenizer.get_vocab())}")

    # Check key tokens exist
    vocab = tokenizer.get_vocab()
    key_tokens = ["<|tool_call>", "<tool_call|>", "<think>", "</think>", '<|"|>']
    print("\nKey token IDs:")
    for tok in key_tokens:
        tid = vocab.get(tok)
        print(f"  {tok!r:20s} -> {tid}")

    tool_parser = GrammarGemma4ToolParser(tokenizer)
    reasoning_parser = GrammarGemma4ReasoningParser(tokenizer)

    # Determine what text to debug
    texts_to_test = {}
    if args.output:
        with open(args.output) as f:
            texts_to_test["file_output"] = f.read()
    elif args.text:
        texts_to_test["cli_output"] = args.text
    elif REAL_MODEL_OUTPUT:
        texts_to_test["real_output"] = REAL_MODEL_OUTPUT

    if not texts_to_test:
        texts_to_test = SYNTHETIC_EXAMPLES
        print("\nNo real model output provided. Using synthetic examples.")
        print("To debug real output, use --output <file> or --text '<output>'")

    for label, text in texts_to_test.items():
        debug_engine_events(tool_parser, text, tokenizer, f"tool - {label}")
        debug_non_streaming(tool_parser, text, f"tool - {label}")
        debug_streaming(tool_parser, text, tokenizer, f"tool - {label}")

        debug_engine_events(reasoning_parser, text, tokenizer, f"reasoning - {label}")
        debug_non_streaming(reasoning_parser, text, f"reasoning - {label}")
        debug_streaming(reasoning_parser, text, tokenizer, f"reasoning - {label}")


if __name__ == "__main__":
    main()
