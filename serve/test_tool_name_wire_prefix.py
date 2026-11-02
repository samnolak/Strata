"""omp/oh-my-pi (Ombra's engine) on an anthropic-messages provider sends its tools with Claude Code's wire prefix
(`_read`, `_bash`, `_write`, ...: the provider defaults to the Claude Code OAuth identity), while its system prompt's
tool inventory names them `read`, `bash`, `write`.  Qwen3.8 then often writes <function=write>: with declared_only
that went back as text ("did not read as a call of a declared tool ... the names the model wrote: write") and the
agent loop stopped.  The unprefixed name is that tool when it is unambiguous; an exact declared name wins.

Captured 2026-11-02 from the Strata monitor (request f4789bac5637 / 46256c244394, ombra-local 18.12.2).

    python -m unittest serve.test_tool_name_wire_prefix -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser  # noqa: E402

OMP_NAMES = ["_read", "_bash", "_edit", "_eval", "_glob", "_grep", "_task", "_wait", "_todo", "web_search", "_write"]


def tool(name, *params):
    return {"name": name, "parameters": {"type": "object",
                                         "properties": {p: {"type": "string"} for p in params or ("path",)}}}


OMP_TOOLS = [tool("_read", "path", "i"), tool("_bash", "command", "i"), tool("_write", "path", "content", "i")] + \
            [tool(n) for n in OMP_NAMES if n not in ("_read", "_bash", "_write")]

# the model's answer of request 46256c244394, verbatim
OBSERVED = ('<tool_call>\n<function=write>\n<parameter=path>\nxd://mcp__studio_status\n</parameter>\n'
            '<parameter=content>\n{}\n</parameter>\n<parameter=i>\nChecking studio status\n</parameter>\n</function>\n'
            '</tool_call>\n<tool_call>\n<function=read>\n<parameter=path>\nskill://local-inference\n</parameter>\n'
            '<parameter=i>\nReading local inference skill\n</parameter>\n</function>\n</tool_call>')


def run(text, tools, width=0, thinking=False):
    p = OutputParser(thinking=thinking, tools=tools, stream_tools=True, recover=True, declared_only=True)
    evs, step = [], width or len(text) or 1
    for k in range(0, len(text), step):
        evs += p.feed(text[k:k + step])
    evs += p.finish("stop")
    return evs, p


def calls(evs):
    return [e.call for e in evs if e.kind == "tool_call"]


def text_of(evs):
    return "".join(e.text or "" for e in evs if e.kind == "content")


class WirePrefixedToolNames(unittest.TestCase):
    def test_observed_answer_reads_as_two_calls_of_the_declared_tools(self):
        for width in (0, 1, 7):
            with self.subTest(width=width):
                evs, p = run(OBSERVED, OMP_TOOLS, width)
                found = calls(evs)
                self.assertEqual([c.name for c in found], ["_write", "_read"])   # the client gets the name it declared
                self.assertEqual(found[0].arguments, {"path": "xd://mcp__studio_status", "content": "{}",
                                                      "i": "Checking studio status"})
                self.assertEqual(text_of(evs).strip(), "")
                self.assertEqual((p.unread, p.unread_names), (0, []))

    def test_reasoning_text_then_a_call_with_a_fenced_markdown_body(self):
        body = "# Report\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```python\nprint('<tool_call>')\n```\n"
        raw = ("<think>\n先看看文件，然后写报告。\n</think>\n\nWriting the report now.\n\n<tool_call>\n<function=write>\n"
               f"<parameter=path>\nREPORT.md\n</parameter>\n<parameter=content>\n{body}</parameter>\n</function>\n</tool_call>")
        for width in (0, 1, 13):
            with self.subTest(width=width):
                evs, p = run(raw, OMP_TOOLS, width, thinking=True)
                found = calls(evs)
                self.assertEqual([c.name for c in found], ["_write"])
                self.assertEqual(found[0].arguments["content"], body.rstrip("\n"))
                self.assertEqual(text_of(evs).strip(), "Writing the report now.")
                self.assertEqual(p.unread, 0)

    def test_the_prefixed_spelling_still_works(self):
        evs, _ = run('<tool_call>\n<function=_bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>',
                     OMP_TOOLS)
        self.assertEqual([(c.name, c.arguments) for c in calls(evs)], [("_bash", {"command": "ls"})])

    def test_an_exactly_declared_unprefixed_tool_wins(self):
        tools = OMP_TOOLS + [tool("bash", "command")]
        evs, _ = run('<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>', tools)
        self.assertEqual([c.name for c in calls(evs)], ["bash"])

    def test_two_tools_with_the_same_unprefixed_name_are_not_guessed(self):
        tools = [tool("_bash", "command"), tool("__bash", "command")]
        evs, p = run('<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>', tools)
        self.assertEqual(calls(evs), [])
        self.assertEqual(p.unread_names, ["bash"])

    def test_an_undeclared_name_stays_text(self):
        evs, p = run('<tool_call>\n<function=python>\n<parameter=code>\n1\n</parameter>\n</function>\n</tool_call>',
                     OMP_TOOLS)
        self.assertEqual(calls(evs), [])
        self.assertIn("<function=python>", text_of(evs))
        self.assertEqual(p.unread_names, ["python"])


if __name__ == "__main__":
    unittest.main()
