"""ctl55-robust: broken tool-call endings regenerated (STRATA_TOOLCALL_RETRY), the thinking capped at a share of max_tokens
(STRATA_REASONING_CAP_FRAC), a small Anthropic max_tokens raised for a thinking request (STRATA_MAX_OUT_MIN).  All off by
default: the off tests pin the responses of the unpatched server.

    python -I -B serve/test_toolcall_retry.py -v
"""
from __future__ import annotations

import json
import os
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from serve import toolcall_retry as tcr  # noqa: E402
from serve.frontend import ChatTemplate, Event, ToolCall  # noqa: E402
from serve.server import REASONING_WRAP_UP, ByteTokenizer, MockEngine, Service, serve  # noqa: E402

TOOLS = [{"name": "read", "description": "read a file",
          "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
LEAD = "Let me look.\n\n"
BROKEN = "<tool_call><|im_start|>function>"                       # the ending seen in a real session (1 in 88 requests)
REST = "read>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>"
CALL = "<tool_call>\n<function=" + REST
END = "<|im_end|>"


class ScriptEngine(MockEngine):
    """Answers by what the prompt ends with: `rules` is a list of (suffix, output); the first match wins, "" matches all.
    Records every prompt and every sampling dict."""

    def __init__(self, tok, rules):
        super().__init__(tok, "x", max_context=8192)
        self.rules, self.prompts, self.samplings = rules, [], []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        self.samplings.append(dict(sampling or {}))
        text = self.tok.decode(ids)
        out = next(o for suffix, o in self.rules if text.endswith(suffix))
        for t in (self.tok.encode(out, parse_special=True) + self.tok.encode(END, parse_special=True))[:max_new]:
            if cancel.is_set():
                return
            yield t


class Base(unittest.TestCase):
    RULES = [(tcr.OPENER, REST), ("", LEAD + BROKEN)]

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = ScriptEngine(self.tok, list(self.RULES))
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
                return r.status, (json.loads(raw) if not body.get("stream") else raw)
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def messages(self, stream=False, thinking=False, **extra):
        body = {"model": "m", "max_tokens": 600, "stream": stream, "tools": TOOLS,
                "messages": [{"role": "user", "content": "read a.py"}], **extra}
        if not thinking:
            body["thinking"] = {"type": "disabled"}
        code, b = self.post("/v1/messages", body)
        self.assertEqual(code, 200, b)
        return b

    def totals(self):
        with urllib.request.urlopen(self.base + "/metrics", timeout=10) as r:
            return json.loads(r.read())["totals"]

    @staticmethod
    def sse(raw):
        return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]

    @staticmethod
    def texts(msg):
        return "".join(c.get("text", "") for c in msg["content"] if c["type"] == "text")

    @staticmethod
    def calls(msg):
        return [(c["name"], c["input"]) for c in msg["content"] if c["type"] == "tool_use"]


class Off(Base):
    def test_off_by_default_the_broken_ending_goes_out_as_before_and_is_counted(self):
        self.assertEqual(self.svc.toolcall_retry, 0)
        msg = self.messages()
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertIn("<tool_call>", self.texts(msg))
        self.assertEqual(msg["stop_reason"], "end_turn")
        self.assertNotIn("strata_rejected_tool_calls", msg)        # the response is the unpatched one
        t = self.totals()
        self.assertEqual(t.get("tool_calls_malformed"), 1)
        self.assertNotIn("toolcall_retries", t)

    def test_off_a_good_call_is_not_counted(self):
        self.engine.rules = [("", LEAD + CALL)]
        msg = self.messages()
        self.assertEqual(self.calls(msg), [("read", {"path": "a.py"})])
        self.assertNotIn("tool_calls_malformed", self.totals())


class Retry(Base):
    def setUp(self):
        super().setUp()
        self.svc.toolcall_retry = 2

    def test_the_broken_ending_is_regenerated_with_the_template_opener(self):
        msg = self.messages()
        self.assertEqual(self.calls(msg), [("read", {"path": "a.py"})])
        self.assertEqual(self.texts(msg), LEAD.strip())
        self.assertEqual(msg["stop_reason"], "tool_use")
        self.assertNotIn("strata_rejected_tool_calls", msg)
        first, second = self.engine.prompts
        # the same prefix plus the kept tokens plus the opener: the engine continues from what it holds
        self.assertEqual(second, first + self.tok.encode(LEAD) + self.tok.encode(tcr.OPENER, parse_special=True))
        t = self.totals()
        self.assertEqual((t.get("tool_calls_malformed"), t.get("toolcall_retries"), t.get("toolcall_retry_ok")), (1, 1, 1))
        self.assertEqual(t.get("toolcall_retry_tokens"), len(self.tok.encode(BROKEN, parse_special=True)) + 1)
        # usage counts what the client got: the lead, the opener and the rest (+ the end token)
        want = len(self.tok.encode(LEAD + CALL, parse_special=True)) + 1
        self.assertEqual(msg["usage"]["output_tokens"], want)

    def test_streamed_the_client_never_sees_the_broken_markup(self):
        evs = self.sse(self.messages(stream=True))
        text = "".join(e["delta"].get("text", "") for e in evs if e["type"] == "content_block_delta")
        self.assertEqual(text, LEAD.strip())
        self.assertNotIn("im_start", json.dumps(evs))
        starts = [e["content_block"] for e in evs if e["type"] == "content_block_start"]
        self.assertEqual([b["type"] for b in starts], ["text", "tool_use"])
        args = "".join(e["delta"].get("partial_json", "") for e in evs if e["type"] == "content_block_delta")
        self.assertEqual(json.loads(args), {"path": "a.py"})
        self.assertEqual(evs[-2]["delta"]["stop_reason"], "tool_use")

    def test_the_text_before_the_markup_streams_before_the_end(self):
        """Only the part from the first `<` that may be markup waits; the lead goes out as it is written."""
        hold = tcr.Hold(True)
        self.assertEqual([e.text for e in hold.push([Event("content", "Let me look.")])], ["Let me look."])
        self.assertEqual(hold.push([Event("content", "\n\n<tool_call>")]), [])
        self.assertTrue(hold.hard)

    def test_a_retry_that_breaks_the_same_way_under_greedy_is_not_repeated(self):
        self.engine.rules = [(tcr.OPENER, "<|im_start|>function>"), ("", LEAD + BROKEN)]
        msg = self.messages(temperature=0)
        self.assertEqual(len(self.engine.prompts), 2)              # one retry, then the same prefix again: stop
        self.assertEqual(msg["stop_reason"], "end_turn")
        self.assertEqual(msg["strata_rejected_tool_calls"][0]["reason"], "malformed")
        t = self.totals()
        self.assertEqual((t.get("tool_calls_malformed"), t.get("toolcall_retries")), (1, 1))
        self.assertNotIn("toolcall_retry_ok", t)

    def test_sampled_retries_stop_at_the_limit(self):
        self.engine.rules = [(tcr.OPENER, "<|im_start|>function>"), ("", LEAD + BROKEN)]
        self.messages()                                            # no temperature: the engine samples
        self.assertEqual(len(self.engine.prompts), 3)              # the original and two retries
        self.assertEqual(self.totals().get("toolcall_retries"), 2)

    def test_a_seeded_retry_of_the_same_prefix_gets_another_seed(self):
        self.engine.rules = [(tcr.OPENER, "<|im_start|>function>"), ("", LEAD + BROKEN)]
        self.messages(temperature=0, seed=7)
        self.assertEqual([s.get("seed") for s in self.engine.samplings], [7, 7, 9])

    def test_an_unterminated_call_is_regenerated(self):
        cut = "<tool_call>\n<function=read>\n<parameter=path>\na.p"
        self.engine.rules = [(tcr.OPENER, REST), ("", LEAD + cut)]
        msg = self.messages()
        self.assertEqual(self.calls(msg), [("read", {"path": "a.py"})])
        self.assertEqual(msg["stop_reason"], "tool_use")
        evs = self.sse(self.messages(stream=True))
        starts = [e["content_block"] for e in evs if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use"]
        self.assertEqual(len(starts), 1)                           # the half call was held, never announced
        self.assertEqual(evs[-2]["delta"]["stop_reason"], "tool_use")

    def test_an_unterminated_call_under_greedy_is_left_as_it_was(self):
        """Its opener was already the template's: the same greedy prefix would write the same tokens."""
        cut = "<tool_call>\n<function=read>\n<parameter=path>\na.p"
        self.engine.rules = [(tcr.OPENER, REST), ("", LEAD + cut)]
        msg = self.messages(temperature=0)
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertEqual(msg["strata_rejected_tool_calls"][0]["reason"], "malformed")

    def test_without_holding_calls_an_announced_half_call_is_only_counted(self):
        self.svc.toolcall_hold_calls = False
        cut = "<tool_call>\n<function=read>\n<parameter=path>\na.p"
        self.engine.rules = [(tcr.OPENER, REST), ("", LEAD + cut)]
        self.messages(stream=True)
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertEqual(self.totals().get("tool_calls_malformed"), 1)

    def test_a_chat_control_token_in_the_answer_is_cut_there(self):
        self.engine.rules = [("", "All done.<|im_start|>user\nthanks")]
        msg = self.messages()
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertEqual(self.texts(msg), "All done.")
        self.assertEqual(msg["stop_reason"], "end_turn")
        t = self.totals()
        self.assertEqual((t.get("toolcall_cuts"), t.get("toolcall_retry_ok")), (1, 1))

    def test_prose_and_code_that_name_the_format_are_not_touched(self):
        for answer in ("Use a <tool_call> block to call a tool.",
                       "Example:\n```\n<tool_call>\n<function=read>\n```\nThat is the format.",
                       "Inline `<tool_call><function=read>` is quoted.",
                       "if a < b and b > c then a <b> tag"):
            with self.subTest(answer=answer):
                self.engine.rules = [("", answer)]
                self.engine.prompts.clear()
                msg = self.messages()
                self.assertEqual(len(self.engine.prompts), 1)
                self.assertEqual(self.texts(msg), answer)
                self.assertEqual(msg["stop_reason"], "end_turn")
        self.assertNotIn("toolcall_retries", self.totals())

    def test_a_good_call_after_text_with_a_lt_sign_streams_unchanged(self):
        self.engine.rules = [("", "Since a < b:\n\n" + CALL)]
        evs = self.sse(self.messages(stream=True))
        text = "".join(e["delta"].get("text", "") for e in evs if e["type"] == "content_block_delta")
        self.assertEqual(text, "Since a < b:")
        self.assertEqual(evs[-2]["delta"]["stop_reason"], "tool_use")
        self.assertEqual(len(self.engine.prompts), 1)

    def test_with_thinking_the_retry_keeps_the_thinking_and_its_prefix(self):
        thought = "I should read the file."
        self.engine.rules = [(tcr.OPENER, REST), ("", thought + "</think>\n\n" + LEAD + BROKEN)]
        msg = self.messages(thinking=True)
        self.assertEqual(self.calls(msg), [("read", {"path": "a.py"})])
        self.assertEqual("".join(c.get("thinking", "") for c in msg["content"] if c["type"] == "thinking"), thought)
        first, second = self.engine.prompts
        self.assertEqual(second, first + self.tok.encode(thought + "</think>\n\n" + LEAD, parse_special=True)
                         + self.tok.encode(tcr.OPENER, parse_special=True))

    def test_after_the_budget_wrap_up_the_retry_continues_the_second_pass(self):
        thought = "Let me think step by step about the file. " * 3
        self.engine.rules = [(tcr.OPENER, REST), ("</think>\n\n", LEAD + BROKEN), ("", thought + "</think>\n\nnever")]
        msg = self.messages(thinking=True, reasoning_budget_tokens=20)
        self.assertEqual(self.calls(msg), [("read", {"path": "a.py"})])
        first, second, third = self.engine.prompts
        self.assertEqual(second, first + self.tok.encode(thought[:20]) + self.tok.encode(REASONING_WRAP_UP, parse_special=True))
        self.assertEqual(third, second + self.tok.encode(LEAD) + self.tok.encode(tcr.OPENER, parse_special=True))

    def test_text_with_a_lt_sign_before_the_broken_call_stays_as_sent(self):
        lead = "Since a < b:\n\n"
        self.engine.rules = [(tcr.OPENER, REST), ("", lead + BROKEN)]
        evs = self.sse(self.messages(stream=True))
        text = "".join(e["delta"].get("text", "") for e in evs if e["type"] == "content_block_delta")
        self.assertEqual(text, "Since a < b:")
        self.assertEqual(evs[-2]["delta"]["stop_reason"], "tool_use")
        first, second = self.engine.prompts
        self.assertEqual(second, first + self.tok.encode(lead) + self.tok.encode(tcr.OPENER, parse_special=True))

    def test_after_the_close_retry_a_broken_call_is_regenerated_too(self):
        """#1053's continuation (the thinking closed by the server) and then a broken call: both passes are kept."""
        self.svc.reasoning_close_retry = True
        thought = "I should read it."
        self.engine.rules = [(tcr.OPENER, REST), ("</think>\n\n", LEAD + BROKEN), ("", thought)]
        msg = self.messages(thinking=True)
        self.assertEqual(self.calls(msg), [("read", {"path": "a.py"})])
        self.assertEqual(len(self.engine.prompts), 3)
        self.assertEqual(self.engine.prompts[2], self.engine.prompts[1] + self.tok.encode(LEAD)
                         + self.tok.encode(tcr.OPENER, parse_special=True))

    def test_a_second_call_cut_after_a_complete_first_one_is_regenerated(self):
        cut = "<tool_call>\n<function=read>\n<parameter=path>\nb."
        self.engine.rules = [(tcr.OPENER, REST), ("", LEAD + CALL + "\n" + cut)]
        msg = self.messages()
        self.assertEqual(self.calls(msg), [("read", {"path": "a.py"}), ("read", {"path": "a.py"})])
        self.assertEqual(msg["stop_reason"], "tool_use")

    def test_an_opener_with_nothing_after_it_is_regenerated(self):
        for dangling in ("<tool_call>", "<tool_call>\n<function", "<tool_call>\n<function=re"):
            with self.subTest(dangling=dangling):
                self.engine.rules = [(tcr.OPENER, REST), ("", LEAD + dangling)]
                self.engine.prompts.clear()
                msg = self.messages(stream=False)
                self.assertEqual(self.calls(msg), [("read", {"path": "a.py"})])
                self.assertEqual(self.texts(msg), LEAD.strip())

    def test_a_call_whose_wrapper_closer_is_left_out_is_not_retried(self):
        """#4427: `</function>` and then the end of the turn is a whole call (the parser's end reads it)."""
        self.engine.rules = [(tcr.OPENER, REST), ("", LEAD + CALL[:-len("\n</tool_call>")])]
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.engine.prompts.clear()
                out = self.messages(stream=stream)
                self.assertEqual(len(self.engine.prompts), 1)
                stop = self.sse(out)[-2]["delta"]["stop_reason"] if stream else out["stop_reason"]
                self.assertEqual(stop, "tool_use")
        self.assertNotIn("tool_calls_malformed", self.totals())

    def test_a_stop_sequence_request_is_left_alone(self):
        msg = self.messages(stop_sequences=["zzz"])
        self.assertEqual(len(self.engine.prompts), 1)
        self.assertIn("<tool_call>", self.texts(msg))

    def test_openai_route_too(self):
        code, b = self.post("/v1/chat/completions", {
            "model": "m", "max_tokens": 600, "reasoning_effort": "none",
            "tools": [{"type": "function", "function": {"name": "read", "parameters": TOOLS[0]["input_schema"]}}],
            "messages": [{"role": "user", "content": "read a.py"}]})
        self.assertEqual(code, 200, b)
        c = b["choices"][0]
        self.assertEqual(c["finish_reason"], "tool_calls")
        self.assertEqual(json.loads(c["message"]["tool_calls"][0]["function"]["arguments"]), {"path": "a.py"})


REAL_TOKENIZER = Path(os.environ.get("STRATA_TEST_TOKENIZER") or "<set STRATA_TEST_TOKENIZER>")


def real_tokenizer():
    """The pack's own BPE tokenizer, loaded the way serve/server.py main() loads it (CPU only, ~10 MB of files)."""
    import strata_tokenizer as ST
    vocab = json.loads((REAL_TOKENIZER / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (REAL_TOKENIZER / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((REAL_TOKENIZER / "token_type.json").read_text())
    tcfg = json.loads((REAL_TOKENIZER / "tokenizer.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)


@unittest.skipUnless((REAL_TOKENIZER / "vocab.json").exists(), "the pack tokenizer is not on this machine")
class RealTokenizer(Retry):
    """The same retries with the real Qwen vocabulary: <tool_call> and <|im_start|> are single special tokens and the
    detokenizer is the incremental UTF-8 one."""
    _tok = None

    def setUp(self):
        if RealTokenizer._tok is None:
            RealTokenizer._tok = real_tokenizer()
        self.tok = RealTokenizer._tok
        self.engine = ScriptEngine(self.tok, list(self.RULES))
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.toolcall_retry = 2
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def test_the_markers_are_single_tokens(self):
        for tag in ("<tool_call>", "<|im_start|>", "<|im_end|>"):
            self.assertEqual(len(self.tok.encode(tag, parse_special=True)), 1, tag)

    def test_utf8_before_the_broken_call(self):
        lead = "Смотрю файл — ✓.\n\n"
        self.engine.rules = [(tcr.OPENER, REST), ("", lead + BROKEN)]
        msg = self.messages()
        self.assertEqual(self.texts(msg), lead.strip())
        self.assertEqual(self.calls(msg), [("read", {"path": "a.py"})])

    # ByteTokenizer-specific token counts do not hold here
    def test_the_broken_ending_is_regenerated_with_the_template_opener(self):
        msg = self.messages()
        self.assertEqual(self.calls(msg), [("read", {"path": "a.py"})])
        self.assertEqual(self.texts(msg), LEAD.strip())
        first, second = self.engine.prompts
        self.assertEqual(second[:len(first)], first)
        self.assertEqual(self.tok.decode(second[len(first):]), LEAD + tcr.OPENER)

    def test_after_the_budget_wrap_up_the_retry_continues_the_second_pass(self):
        thought = "Let me think step by step about the file. " * 3
        self.engine.rules = [(tcr.OPENER, REST), ("</think>\n\n", LEAD + BROKEN), ("", thought + "</think>\n\nnever")]
        msg = self.messages(thinking=True, reasoning_budget_tokens=8)
        self.assertEqual(self.calls(msg), [("read", {"path": "a.py"})])
        first, second, third = self.engine.prompts
        self.assertEqual(second[:len(first)], first)
        self.assertTrue(self.tok.decode(second[len(first):]).endswith(REASONING_WRAP_UP))
        self.assertEqual(third[:len(second)], second)
        self.assertEqual(self.tok.decode(third[len(second):]), LEAD + tcr.OPENER)


class Markup(unittest.TestCase):
    def test_call_shaped(self):
        for text, kind in (("x <tool_call><|im_start|>function>", "call"), ("x\n<tool_call>", "call"),
                           ("x\n<function=read>\n<parameter=path>", "call"), ("ok<|im_start|>user\nhi", "control"),
                           ("ok<|im_start|>function=read>", "call"), ("x\n<invoke name=\"read\">", "call")):
            with self.subTest(text=text):
                self.assertEqual(tcr.call_markup(text)[1], kind)
        for text in ("a <tool_call> block", "```\n<tool_call>\n<function=x>\n```", "`<function=x>`", "a < b", "<b>bold</b>",
                     "```python\nprint('<tool_call>')\n"):
            with self.subTest(text=text):
                self.assertIsNone(tcr.call_markup(text))

    def test_hold_releases_when_no_marker_can_form(self):
        hold = tcr.Hold(True)
        self.assertEqual(hold.push([Event("content", "a <")]), [])           # may become <tool_call>
        out = hold.push([Event("content", " b")])
        self.assertEqual("".join(e.text for e in out), "a < b")
        self.assertEqual(hold.push([Event("content", "<tool")]), [])
        self.assertEqual(hold.push([Event("content", "box>")]), [Event("content", "<tool"), Event("content", "box>")])

    def test_hold_keeps_a_streamed_call_until_it_completes(self):
        hold = tcr.Hold(True)
        call = ToolCall("read", {"path": "a"})
        self.assertEqual(hold.push([Event("tool_start", call=call), Event("tool_args", '{"path": ')]), [])
        out = hold.push([Event("tool_args", '"a"}'), Event("tool_call", call=call)])
        self.assertEqual([e.kind for e in out], ["tool_start", "tool_args", "tool_args", "tool_call"])
        self.assertEqual((hold.calls, hold.sent_calls, hold.sent_starts), (1, 1, 1))

    def test_off_passes_everything_through(self):
        hold = tcr.Hold(False)
        evs = [Event("content", "<tool_call>"), Event("tool_start", call=ToolCall("x", {}))]
        self.assertEqual(hold.push(evs), evs)
        self.assertEqual(hold.flush(), [])
        self.assertEqual(hold.content(), "<tool_call>")

    def test_env_values(self):
        with mock.patch.dict(os.environ, {"STRATA_TOOLCALL_RETRY": "x", "STRATA_REASONING_CAP_FRAC": "1.5"}):
            self.assertEqual(tcr.env_int("STRATA_TOOLCALL_RETRY", 0), 0)
            self.assertIsNone(tcr.env_frac("STRATA_REASONING_CAP_FRAC"))
        with mock.patch.dict(os.environ, {"STRATA_TOOLCALL_RETRY": "2", "STRATA_REASONING_CAP_FRAC": "0.6"}):
            self.assertEqual(tcr.env_int("STRATA_TOOLCALL_RETRY", 0), 2)
            self.assertEqual(tcr.env_frac("STRATA_REASONING_CAP_FRAC"), 0.6)
        with mock.patch.dict(os.environ, {"STRATA_TOOLCALL_RETRY": "3", "STRATA_REASONING_CAP_FRAC": "0.5",
                                          "STRATA_MAX_OUT_MIN": "4096", "STRATA_TOOLCALL_RETRY_HOLD_CALLS": "0"}):
            svc = Service(MockEngine(ByteTokenizer(), "x"), ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertEqual((svc.toolcall_retry, svc.reasoning_cap_frac, svc.max_out_min, svc.toolcall_hold_calls),
                         (3, 0.5, 4096, False))


class LongThinker(ScriptEngine):
    """Thinks THOUGHT (one token per byte) unless the prompt already closed the thinking."""
    THOUGHT = "Let me think step by step about two plus two. " * 60       # 2,760 reasoning tokens
    ANSWER = "The answer is 4."

    def __init__(self, tok):
        super().__init__(tok, [("</think>\n\n", self.ANSWER), ("", self.THOUGHT + "</think>\n\n" + self.ANSWER)])
        self.max_context = 32768


class ThinkingShare(Base):
    def setUp(self):
        super().setUp()
        self.engine = self.svc.engine = LongThinker(self.tok)

    def ask(self, max_tokens, **extra):
        body = {"model": "m", "max_tokens": max_tokens, "messages": [{"role": "user", "content": "2+2?"}], **extra}
        code, b = self.post("/v1/messages", body)
        self.assertEqual(code, 200, b)
        thinking = "".join(c.get("thinking", "") for c in b["content"] if c["type"] == "thinking")
        return b, thinking

    def test_off_the_built_in_answer_room_keeps_its_quarter(self):
        b, thinking = self.ask(2048)
        self.assertEqual(self.texts(b), LongThinker.ANSWER)
        wrap = len(REASONING_WRAP_UP.split("</think>")[0])
        self.assertEqual(len(thinking) - wrap, 2048 - 512)          # #984 / answer_room: max_tokens - max(512, max/4)

    def test_the_audit_case_8192_with_a_16384_budget_is_already_capped_at_three_quarters(self):
        """Claude Code sends max_tokens 8192 and thinking {type: adaptive}; the presets set reasoning_budget_tokens 16384.
        #984 already closes the thinking at max_tokens - max(512, max/4) = 6144, so the answer keeps 2048."""
        self.engine.rules[1] = ("", "Let me think step by step about two plus two. " * 200 + "</think>\n\n" + LongThinker.ANSWER)
        self.svc.reasoning_budget_tokens = 16384
        b, thinking = self.ask(8192, thinking={"type": "adaptive"}, output_config={"effort": "high"})
        self.assertEqual(self.texts(b), LongThinker.ANSWER)
        wrap = len(REASONING_WRAP_UP.split("</think>")[0])
        self.assertEqual(len(thinking) - wrap, 6144)
        self.svc.reasoning_cap_frac = 0.6                          # STRATA_REASONING_CAP_FRAC=0.6: 4915
        b, thinking = self.ask(8192, thinking={"type": "adaptive"})
        self.assertEqual(len(thinking) - wrap, 4915)

    def test_the_cap_frac_lowers_the_thinking_share(self):
        self.svc.reasoning_cap_frac = 0.5
        b, thinking = self.ask(2048, thinking={"type": "enabled", "budget_tokens": 16384})
        self.assertEqual(self.texts(b), LongThinker.ANSWER)
        wrap = len(REASONING_WRAP_UP.split("</think>")[0])
        self.assertEqual(len(thinking) - wrap, 1024)
        self.assertEqual(self.totals().get("reasoning_cap_hits"), 1)

    def test_the_cap_frac_never_raises_a_smaller_budget(self):
        self.svc.reasoning_cap_frac = 0.9
        b, thinking = self.ask(2048, thinking={"type": "enabled", "budget_tokens": 100})
        wrap = len(REASONING_WRAP_UP.split("</think>")[0])
        self.assertEqual(len(thinking) - wrap, 100)
        self.assertNotIn("reasoning_cap_hits", self.totals())

    def test_max_out_min_off_a_small_cap_is_eaten_by_the_thinking(self):
        b, _ = self.ask(300)
        self.assertEqual(b["stop_reason"], "max_tokens")
        self.assertFalse(self.texts(b))

    def test_max_out_min_raises_a_small_cap_for_a_thinking_request(self):
        self.svc.max_out_min = 4096
        b, thinking = self.ask(300)
        self.assertEqual(b["stop_reason"], "end_turn")
        self.assertEqual(self.texts(b), LongThinker.ANSWER)
        self.assertGreater(b["usage"]["output_tokens"], 300)         # the documented contract break

    def test_max_out_min_leaves_a_request_without_thinking_alone(self):
        self.svc.max_out_min = 4096
        self.engine.rules = [("", "abcdefghij" * 100)]
        b, _ = self.ask(300, thinking={"type": "disabled"})
        self.assertEqual(b["stop_reason"], "max_tokens")
        self.assertEqual(b["usage"]["output_tokens"], 300)


if __name__ == "__main__":
    unittest.main()
