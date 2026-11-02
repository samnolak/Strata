"""ctl55-robust: opt-in robustness for agent turns (Claude Code over /v1/messages).  All off by default.

STRATA_TOOLCALL_RETRY=N (default 0: off, the server behaves exactly as before)
    A turn that ENDS (a natural stop, no stop sequence) on a broken tool call -- `<tool_call><|im_start|>function>`,
    a `<tool_call>` whose call never closes, a bare `<function=...` the parser could not read -- is regenerated up to
    N times before the client sees it.  The answer's events are held from the first byte that may be tool-call
    markup (and, with STRATA_TOOLCALL_RETRY_HOLD_CALLS=1, the default, a streamed call from its tool_start to its
    tool_call) until the turn shows what it was; everything before that streams as usual.  A retry goes back to the
    token before the broken opener (the parser and the detokenizer are rebuilt by replaying the kept tokens), writes
    the template's own opener `<tool_call>\\n<function=` there and lets the model continue: a different, deterministic
    prefix, so a greedy request does not repeat the same tokens; the engine reuses the prefix it already holds.  A
    retry that produced exactly the tokens of the one before stops retrying (greedy and nothing left to change); a
    request with a seed gets seed+attempt from the second attempt on.  A chat-control token in the answer that is not
    the start of a call (`Done.<|im_start|>user ...`) is cut there instead, and the turn ends ("stop").
    When nothing can be fixed, the held events go out unchanged (the same response as without the flag).
STRATA_TOOLCALL_RETRY_HOLD_CALLS=0|1 (default 1)
    0: a streamed call is not held, so a call that the output ended inside after its name was sent is only counted.
STRATA_REASONING_CAP_FRAC=f (0 < f <= 1; default unset: off)
    With thinking on, the thinking is closed (the budget's wrap-up, then `</think>`) after at most f * max_tokens
    reasoning tokens, whatever the budget (request, config, #984 / answer_room).  It only lowers the cap.
STRATA_MAX_OUT_MIN=n (default 0: off)
    An Anthropic request with thinking on and 0 < max_tokens < n runs with max_tokens n (still fitted to the context).
    NOTE: usage.output_tokens may then exceed the request's max_tokens (Anthropic counts thinking in max_tokens).

/metrics totals: tool_calls_malformed (turns that ended on broken call markup, counted with or without the flag),
toolcall_retries, toolcall_retry_ok (a retry that ended on a valid call or a clean answer), toolcall_retry_tokens
(generated tokens dropped by retries), toolcall_cuts (control tokens cut), reasoning_cap_hits.
"""
from __future__ import annotations

import bisect
import copy
import os
import re

from serve.frontend import CALL_START, FUNC_START, Event

# what a broken call can start with in the answer (a whole one switches the hold to "until the end of the turn")
MARKERS = ("<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "<|im_start|>", "<|im_end|>",
           "<|endoftext|>", "<invoke", "</invoke", "<tool_use>", "</tool_use>")
CONTROL = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
OPENER = CALL_START + "\n" + FUNC_START            # the template's opener: `<tool_call>\n<function=` NAME `>`
# call-shaped markup in answer text (code masked out): an opener followed by a tag or the end, a call tag at the start
# of a line, a named `<function=NAME>` anywhere, or a chat-control token.  Prose that names the format ("use a
# <tool_call> block") is not call-shaped, the same rule the parser applies.
_CALL_SHAPED = re.compile(r"<tool_call>\s*(?:<|$)|<function=[^<>\s]*>|(?m:^[ \t]*<(?:function=|parameter=|invoke\b|tool_use>))"
                          r"|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>")
# after a control token: the model was writing a call (`<tool_call><|im_start|>function>`, `<|im_start|>function=X>`)
_CALLISH_AFTER = re.compile(r"\s*(?:function|tool_call|parameter|invoke|<)")
_FENCE = re.compile(r"(?m)^[ \t]*(?:```|~~~)[^\n]*$")
_INLINE = re.compile(r"`[^`\n]*`")


def env_int(name: str, default: int = 0) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        print(f"[strata] {name}={raw!r} is not a whole number: ignored ({default})", flush=True)
        return default


def env_frac(name: str) -> float | None:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        v = -1.0
    if not 0 < v <= 1:
        print(f"[strata] {name}={raw!r}: expected a fraction in (0, 1]; ignored", flush=True)
        return None
    return v


def mask_code(text: str) -> str:
    """`text` with fenced blocks and inline code replaced by spaces (same length: positions still match)."""
    chars = list(text)
    fences = [m for m in _FENCE.finditer(text)]
    for k in range(0, len(fences), 2):              # an unclosed fence runs to the end
        a = fences[k].start()
        b = fences[k + 1].end() if k + 1 < len(fences) else len(text)
        chars[a:b] = " " * (b - a)
    masked = "".join(chars)
    return _INLINE.sub(lambda m: " " * len(m.group(0)), masked)


def call_markup(text: str, start: int = 0):
    """The first call-shaped markup at or after `start` in answer text, outside code -> (position, kind, snippet)
    with kind "call" (regenerate with the template's opener) or "control" (a chat-control token that does not start a
    call: cut there); None when there is none."""
    masked = mask_code(text)
    m = _CALL_SHAPED.search(masked, start)
    if m is None:
        return None
    pos, tag = m.start(), m.group(0)
    kind = "call"
    if tag in CONTROL and not _CALLISH_AFTER.match(text, m.end()):
        kind = "control"
    return pos, kind, text[pos:pos + 60]


def _maybe_marker(tail: str) -> bool:
    """A suffix of the held text that starts with `<` may still grow into a marker."""
    i = tail.rfind("<", max(0, len(tail) - 16))
    while i >= 0:
        s = tail[i:]
        if any(m.startswith(s) for m in MARKERS):
            return True
        i = tail.rfind("<", max(0, len(tail) - 16), i)
    return False


class Hold:
    """The answer's events, held from the first `<` that may start tool-call markup (and a streamed call from its
    tool_start) until the turn shows what it is.  Disabled (`on` False) it passes everything through and only keeps
    the account (the metrics read it).  `safe` is the generated-text position before which everything was sent."""

    def __init__(self, on: bool, hold_calls: bool = True):
        self.on, self.hold_calls = on, hold_calls
        self.events: list = []
        self.text = ""                  # the held content text
        self.hard = False               # a whole marker is held: kept until the end of the turn
        self.open_calls = 0             # held tool_start events without their tool_call yet
        self.sent_content: list[str] = []
        self.sent_reasoning: list[str] = []
        self.sent_starts = 0
        self.sent_calls = 0
        self.calls = 0                  # complete calls in the turn (sent or held)
        self.safe = 0

    def _account(self, ev):
        if ev.kind == "content":
            self.sent_content.append(ev.text or "")
        elif ev.kind == "reasoning":
            self.sent_reasoning.append(ev.text or "")
        elif ev.kind == "tool_start":
            self.sent_starts += 1
        elif ev.kind == "tool_call":
            self.sent_calls += 1

    def push(self, evs, boundary: int | None = None) -> list:
        """Events to send now.  `boundary`: the generated-text position the parser has consumed up to."""
        out = []
        for ev in evs:
            if ev.kind == "tool_call":
                self.calls += 1
            if not self.on:
                self._account(ev)
                out.append(ev)
                continue
            if not self.events:
                starts = (ev.kind == "content" and "<" in (ev.text or "")) or (ev.kind == "tool_start" and self.hold_calls)
                if not starts:
                    self._account(ev)
                    out.append(ev)
                    continue
            self.events.append(ev)
            if ev.kind == "content":
                self.text += ev.text or ""
                if not self.hard and any(m in self.text for m in MARKERS):
                    self.hard = True
            elif ev.kind == "tool_start":
                self.open_calls += 1
            elif ev.kind == "tool_call" and self.open_calls and any(e.kind == "tool_start" and e.call is not None
                                                                   and e.call.id == ev.call.id for e in self.events):
                self.open_calls -= 1
        if self.events and not self.hard and not self.open_calls and not _maybe_marker(self.text):
            out += self.flush()
        if self.on and not self.events and boundary is not None:
            self.safe = boundary
        return out

    def flush(self) -> list:
        out, self.events = self.events, []
        for ev in out:
            self._account(ev)
        self.text, self.hard, self.open_calls = "", False, 0
        return out

    def discard(self) -> int:
        """Drop the held events (a retry replaces them); the calls among them no longer count."""
        dropped = self.events
        self.calls -= sum(1 for e in dropped if e.kind == "tool_call")
        self.events, self.text, self.hard, self.open_calls = [], "", False, 0
        return len(dropped)

    def content(self) -> str:
        return "".join(self.sent_content) + self.text


class Tape:
    """Every token fed to the parser in this run (generated or written by the server), with the generated-text
    length and the output-token count after it, and the passes (each pass's prompt and the tape position it starts
    at), so a retry can go back to any token: the prompt is that pass's prompt plus the tape up to there."""

    def __init__(self, prompt, answer_at=None):
        self.ids: list[int] = []
        self.ends: list[int] = []
        self.ns: list[int] = []
        self.pieces: list[str] = []
        self.size = 0                   # len(self.text), kept without joining the pieces on every token
        self._text = ""
        self.passes = [(list(prompt), 0)]
        self.answer_at = answer_at      # where the answer (the content state) starts in self.text

    @property
    def text(self) -> str:
        if len(self._text) != self.size:
            self._text = "".join(self.pieces)
        return self._text

    def add(self, t: int, piece: str, n: int):
        self.ids.append(t)
        self.pieces.append(piece)
        self.size += len(piece)
        self.ends.append(self.size)
        self.ns.append(n)

    def new_pass(self, prompt):
        self.passes.append((list(prompt), len(self.ids)))

    def token_before(self, pos: int) -> int:
        """The number of tokens whose text ends at or before `pos`."""
        return bisect.bisect_right(self.ends, pos)

    def prompt_at(self, k: int) -> list[int]:
        for prompt, start in reversed(self.passes):
            if start <= k:
                return prompt + self.ids[start:k]
        return self.passes[0][0] + self.ids[:k]

    def cut(self, k: int):
        del self.ids[k:], self.ends[k:], self.ns[k:], self.pieces[k:]
        self.size = self.ends[-1] if self.ends else 0
        self._text = "".join(self.pieces)
        self.passes = [(p, s) for p, s in self.passes if s <= k]
        if self.answer_at is not None and self.answer_at > self.size:
            self.answer_at = None


def broken_at(parser, tape: Tape, hold: Hold):
    """Where the turn's broken call starts in tape.text -> (position, kind, snippet), or None when the turn is not
    broken.  Only for a turn whose content shows no complete call (an unterminated call counts whatever came before:
    its half-sent tool_use would end the turn as end_turn)."""
    if parser.state in ("call", "json"):
        if finish_completes(parser):
            return None                 # the parser's own end reads it as a call (a `</tool_call>` left out, #4427)
        lead = FUNC_START if getattr(parser, "bare", False) else CALL_START
        pos = len(tape.text) - len(parser.buf) - (0 if lead == FUNC_START else len(CALL_START))
        if pos < 0 or not tape.text.startswith(lead, pos):
            pos = tape.text.rfind(lead)
        if pos < 0:
            return None
        return pos, "call", tape.text[pos:pos + 60]
    if hold.calls:
        return None
    base = tape.answer_at or 0
    found = None
    if hold.hard:                       # held answer text with a whole marker in it
        found = call_markup(tape.text[base:], max(hold.safe, base) - base)
    if found is None and parser.state == "content" and parser.buf:
        # text the parser still holds (`<tool_call>` whose follower never came) goes out as content at its end
        at = tape.size - len(parser.buf)
        if at >= base and tape.text.startswith(parser.buf, at):
            found = call_markup(tape.text[base:], at - base)
    if found is None:
        return None
    return base + found[0], found[1], found[2]


def finish_completes(parser) -> bool:
    """A turn that stops inside a call: does the parser's end (on a copy) still deliver it as a call?"""
    memo = {id(v): v for v in (getattr(parser, "schemas", None), getattr(parser, "aliases", None)) if v is not None}
    try:
        probe = copy.deepcopy(parser, memo)
        evs = probe.finish("stop")
    except Exception:               # a parser that cannot be copied or finished twice: take the call as broken
        return False
    return any(e.kind == "tool_call" for e in evs) and call_markup(
        "".join(e.text or "" for e in evs if e.kind == "content")) is None


def replay(new_parser, detok_cls, tok, ids, hold: Hold):
    """A fresh parser and detokenizer fed `ids` -> (parser, detok, events to send now, tail) or None when the replay
    does not agree with what was already sent (then nothing is retried)."""
    p, d = new_parser(), detok_cls(tok)
    evs, text = [], ""
    for t in ids:
        piece = d.push(t)
        text += piece
        evs += p.feed(piece)
    content = "".join(e.text or "" for e in evs if e.kind == "content")
    reasoning = "".join(e.text or "" for e in evs if e.kind == "reasoning")
    sent_content = "".join(hold.sent_content)
    if reasoning != "".join(hold.sent_reasoning) or not content.startswith(sent_content):
        return None
    if sum(e.kind == "tool_call" for e in evs) != hold.sent_calls or sum(e.kind == "tool_start" for e in evs) != hold.sent_starts:
        return None
    if p.state not in ("content", "reasoning"):
        return None
    extra = content[len(sent_content):]
    return p, d, ([Event("content", extra)] if extra else []), text[-2:]
