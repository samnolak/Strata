"""serve/test_fair_users.py - fair sharing of the batch slots between USERS (STRATA_FAIR_USERS, opt-in): who a request
belongs to (tenant_of, the prompt-start fallback), the gate's order (fewest requests in the engine, then fewest tokens
served, short first within a user, the max-wait valve), its bookkeeping (cancel, deadline, an early stop, a user coming
back) and the real StrataEngine + Service on the fake engine of serve/test_parallel.py.  No GPU.

    python -m unittest serve.test_fair_users -v
"""
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve.server import StrataEngine, tenant_of  # noqa: E402
from serve import test_parallel as tp  # noqa: E402


def engine(n=2, env=None):
    """A StrataEngine with n slots and no process; generate_batched is replaced per test."""
    e = StrataEngine("missing-executable", ["--batch", str(n)], lazy=True)
    e.batch = n
    e.slot_busy = [False] * n
    e.slot_held = [[] for _ in range(n)]
    e.slot_cv = threading.Condition()
    e._swarm_config(env if env is not None else {"STRATA_FAIR_USERS": "1"})
    e.alive = lambda: True
    e.__dict__.setdefault("_tl", threading.local())
    return e


class Who(unittest.TestCase):
    def test_header_then_user_then_metadata(self):
        self.assertEqual(tenant_of({"X-Strata-User": " alice "}, {"user": "bob"}), "u:alice")
        self.assertEqual(tenant_of({}, {"user": "bob"}), "u:bob")
        self.assertEqual(tenant_of({}, {"safety_identifier": "carol"}), "u:carol")
        self.assertEqual(tenant_of({}, {"metadata": {"user_id": "dave"}}), "u:dave")
        self.assertIsNone(tenant_of({}, {"user": "", "metadata": {"user_id": 7}}))
        self.assertIsNone(tenant_of({}, {"messages": []}))
        self.assertIsNone(tenant_of(None, None))
        self.assertEqual(tenant_of({"X-Strata-User": "x" * 500}, {}), "u:" + "x" * 128)

    def test_one_api_key_per_user_comes_first(self):
        keys = "key-alice,key-bob"
        self.assertEqual(tenant_of({"Authorization": "Bearer key-bob", "X-Strata-User": "alice"}, {"user": "x"}, keys),
                         "k:2")                                      # the key decides: a name cannot claim a share
        self.assertEqual(tenant_of({"x-api-key": "key-alice"}, {}, ["key-alice", "key-bob"]), "k:1")
        self.assertEqual(tenant_of({"Authorization": "Bearer key-alice"}, {"user": "x"}, "only-one-key"), "u:x")
        self.assertEqual(tenant_of({"Authorization": "Bearer wrong"}, {"user": "x"}, keys), "u:x")

    def test_claude_code_sessions_of_one_account_are_one_user(self):
        a = "user_ab12_account_1111-2222_session_aaaa"
        b = "user_ab12_account_1111-2222_session_bbbb"
        self.assertEqual(tenant_of({}, {"metadata": {"user_id": a}}), tenant_of({}, {"metadata": {"user_id": b}}))
        self.assertEqual(tenant_of({}, {"metadata": {"user_id": a}}), "u:user_ab12_account_1111-2222")
        j = json.dumps({"device_id": "dev1", "account_uuid": "acc1", "session_id": "s1"})
        self.assertEqual(tenant_of({}, {"metadata": {"user_id": j}}), "u:acc1")

    def test_the_prompt_start_when_nobody_says(self):
        e = engine(env={"STRATA_FAIR_USERS": "1", "STRATA_FAIR_PREFIX_TOKENS": "100"})
        repo_x = list(range(1000, 1100))
        repo_y = list(range(5000, 5100))
        a1, a2 = e._fair_tenant(repo_x + [1, 2, 3]), e._fair_tenant(repo_x + [9, 9])
        self.assertEqual(a1, a2)                                    # one swarm on one repository: one user
        self.assertNotEqual(a1, e._fair_tenant(repo_y + [1, 2, 3]))
        self.assertTrue(a1.startswith("p:"))
        e._tl.tenant = "u:alice"
        self.assertEqual(e._fair_tenant(repo_y), "u:alice")         # what the client said comes first
        e._tl.tenant = None
        self.assertEqual(e._fair_prefix(), 100)


class Switch(unittest.TestCase):
    def test_off_by_default_and_from_the_config(self):
        self.assertFalse(server.FAIR_USERS)
        e = engine(env={})
        self.assertFalse(e._fair_on())
        self.assertIsNone(e.fair_view())
        self.assertIsNone(e.__dict__.get("_fair"))
        e._swarm_config({"STRATA_FAIR_USERS": "1"})
        self.assertTrue(e._fair_on())
        self.assertEqual(e._fair_prefix(), server.FAIR_PREFIX_TOKENS)
        e.batch = 0                                                  # no --batch: nothing to share
        self.assertFalse(e._fair_on())

    def test_generate_takes_the_gate_only_when_on(self):
        e = engine(env={})
        seen = []
        e.generate_batched = lambda *a, **k: iter([seen.append("plain") or 7])
        e._fair_generate = lambda *a, **k: iter([seen.append("fair") or 8])
        with mock.patch.object(server.bias_api, "validate_request", lambda *a: None):
            self.assertEqual(list(e.generate([1, 2], 4, {}, threading.Event())), [7])
            e._swarm_config({"STRATA_FAIR_USERS": "1"})
            self.assertEqual(list(e.generate([1, 2], 4, {}, threading.Event())), [8])
        self.assertEqual(seen, ["plain", "fair"])


class Order(unittest.TestCase):
    def entry(self, t, arr, plen=100):
        return {"t": t, "arr": arr, "plen": plen}

    def test_fewest_in_the_engine_then_fewest_served(self):
        e = engine()
        f = e._fair_state()
        a3, a4, b1 = self.entry("u:a", 1.0), self.entry("u:a", 2.0), self.entry("u:b", 3.0)
        f["waiting"] = [a3, a4, b1]
        f["inflight"] = {"u:a": 2}
        self.assertIs(e._fair_next(f, 4.0), b1)                     # b has nothing in the engine: it goes first
        f["inflight"] = {"u:a": 1, "u:b": 1}
        f["served"] = {"u:a": 500.0, "u:b": 100.0}
        self.assertIs(e._fair_next(f, 4.0), b1)                     # level: the one served less
        f["served"] = {"u:a": 100.0, "u:b": 500.0}
        self.assertIs(e._fair_next(f, 4.0), a3)                     # ... and in arrival order within a user

    def test_short_first_within_a_user(self):
        e = engine(env={"STRATA_FAIR_USERS": "1", "STRATA_ADMIT_UNCACHED": "1", "STRATA_ADMIT_SHORT_FIRST": "1000"})
        f = e._fair_state()
        cold, back = self.entry("u:a", 1.0, plen=30_000), self.entry("u:a", 2.0, plen=600)
        f["waiting"] = [cold, back]
        self.assertIs(e._fair_next(f, 3.0), back)
        self.assertIs(e._fair_next(f, 1.0 + server.ADMIT_MAX_WAIT_S + 1), cold)   # waited long enough: short too

    def test_a_long_wait_goes_first_whoever_it_is(self):
        e = engine()
        f = e._fair_state()
        old, new = self.entry("u:a", 0.0), self.entry("u:b", 50.0)
        f["waiting"] = [old, new]
        f["inflight"] = {"u:a": 3}
        self.assertIs(e._fair_next(f, 60.0), new)
        self.assertIs(e._fair_next(f, server.FAIR_MAX_WAIT_S + 1.0), old)


class Gate(unittest.TestCase):
    """_fair_generate with a stand-in generate_batched: each request yields its id, then waits for its release."""

    def setUp(self):
        self.e = engine(2)
        self.release = {}
        self.inner_closed = []

        def inner(ids, max_new, sampling, cancel, embeddings=None):
            try:
                yield ids[0]
                self.release[ids[0]].wait(10)
                yield ids[0]
            finally:
                self.inner_closed.append(ids[0])
        self.e.generate_batched = inner
        self.out = {}

    def go(self, rid, user, cancel=None, plen=10):
        self.release[rid] = threading.Event()
        cancel = cancel or threading.Event()

        def run():
            self.e._tl.tenant = user
            try:
                self.out[rid] = list(self.e._fair_generate([rid] + [0] * (plen - 1), 8, {}, cancel))
            except Exception as ex:   # noqa: BLE001
                self.out[rid] = ex
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    def wait_for(self, cond, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            with self.e.slot_cv:
                if cond(self.e._fair_state()):
                    return True
            time.sleep(0.01)
        self.fail("condition not met")

    def test_another_user_gets_the_next_free_place(self):
        th = [self.go(1, "u:a"), self.go(2, "u:a")]
        self.wait_for(lambda f: f["total"] == 2)
        th += [self.go(3, "u:a"), self.go(4, "u:a")]
        self.wait_for(lambda f: len(f["waiting"]) == 2)
        th.append(self.go(5, "u:b"))
        self.wait_for(lambda f: len(f["waiting"]) == 3)
        self.release[1].set()
        self.wait_for(lambda f: list(f["order"])[2:] == ["u:b"])    # b before a's third and fourth
        for r in (2, 3, 4, 5):
            self.release[r].set()
        for t in th:
            t.join(10)
        self.assertEqual(list(self.e._fair_state()["order"]), ["u:a", "u:a", "u:b", "u:a", "u:a"])
        self.assertEqual({k: v for k, v in self.out.items()}, {r: [r, r] for r in (1, 2, 3, 4, 5)})
        f = self.e._fair_state()
        self.assertEqual((f["total"], f["inflight"], f["waiting"]), (0, {}, []))
        v = self.e.fair_view()
        self.assertEqual((v["users"]["u:a"]["admitted"], v["users"]["u:b"]["admitted"]), (4, 1))
        self.assertGreater(v["users"]["u:a"]["served"], v["users"]["u:b"]["served"])

    def test_cancel_and_deadline_while_waiting(self):
        th = [self.go(1, "u:a"), self.go(2, "u:a")]
        self.wait_for(lambda f: f["total"] == 2)
        cancel = threading.Event()
        th.append(self.go(3, "u:b", cancel=cancel))
        late = threading.Event()
        late._strata_queue_deadline = time.monotonic() + 0.3
        th.append(self.go(4, "u:c", cancel=late))
        self.wait_for(lambda f: len(f["waiting"]) == 2)
        cancel.set()
        th[2].join(5)
        th[3].join(5)
        self.assertEqual(self.out[3], [])                            # cancelled: ends with nothing, as before
        self.assertIsInstance(self.out[4], server.QueueWaitExpired)
        self.wait_for(lambda f: f["waiting"] == [])
        for r in (1, 2):
            self.release[r].set()
        for t in th[:2]:
            t.join(5)
        self.assertEqual(self.e._fair_state()["total"], 0)

    def test_a_consumer_that_stops_early_gives_the_place_back(self):
        self.release[9] = threading.Event()
        self.e._tl.tenant = "u:a"
        g = self.e._fair_generate([9, 0, 0], 8, {}, threading.Event())
        self.assertEqual(next(g), 9)
        self.assertEqual(self.e._fair_state()["total"], 1)
        g.close()
        self.assertEqual(self.inner_closed, [9])                     # the slot's request is stopped at once
        self.assertEqual((self.e._fair_state()["total"], self.e._fair_state()["inflight"]), (0, {}))

    def test_a_dead_engine_ends_the_wait(self):
        th = [self.go(1, "u:a"), self.go(2, "u:a")]
        self.wait_for(lambda f: f["total"] == 2)
        th.append(self.go(3, "u:b"))
        self.wait_for(lambda f: len(f["waiting"]) == 1)
        self.e.alive = lambda: False
        th[2].join(5)
        self.assertIsInstance(self.out[3], server.EngineDied)
        self.e.alive = lambda: True
        for r in (1, 2):
            self.release[r].set()
        for t in th[:2]:
            t.join(5)

    def test_a_user_coming_back_starts_level_not_ahead(self):
        f = self.e._fair_state()
        th = [self.go(1, "u:a")]
        self.wait_for(lambda f: f["total"] == 1)
        with self.e.slot_cv:
            f["served"]["u:a"] = 5000.0                              # a has been served a lot meanwhile
            f["served"]["u:b"] = 10.0                                # b was away (and had little): no credit for it
            f["served"]["u:c"] = 99_999.0                            # c had much: not held against it
        th.append(self.go(2, "u:b"))
        self.wait_for(lambda f: f["total"] == 2)
        self.assertEqual(f["served"]["u:b"], 5000.0 + 10 / server.FAIR_PREFILL_DIV)
        th.append(self.go(3, "u:c"))
        self.wait_for(lambda f: len(f["waiting"]) == 1)
        self.assertEqual(f["served"]["u:c"], 5000.0)                 # level with the least served active user
        for r in (1, 2, 3):
            self.release[r].set()
        for t in th:
            t.join(5)

    def test_reserve_keeps_a_place_for_a_user_between_turns(self):
        self.e._swarm_config({"STRATA_FAIR_USERS": "1", "STRATA_FAIR_RESERVE_S": "0.8"})
        f = self.e._fair_state()
        th = [self.go(1, "u:b")]
        self.wait_for(lambda f: f["total"] == 1)
        self.release[1].set()
        th[0].join(5)
        self.assertIn("u:b", f["idle_since"])                      # b's agent runs its tools now
        th += [self.go(2, "u:a"), self.go(3, "u:a")]
        self.wait_for(lambda f: f["total"] == 1 and len(f["waiting"]) == 1)   # a's second waits: b's place is kept
        time.sleep(0.3)
        self.assertEqual((f["total"], len(f["waiting"])), (1, 1))
        th.append(self.go(4, "u:b"))                                 # b's next turn: in at once
        self.wait_for(lambda f: f["total"] == 2 and list(f["order"])[-1] == "u:b")
        t0 = time.monotonic()
        self.release[4].set()
        th[-1].join(5)
        self.wait_for(lambda f: f["total"] == 2)                    # b away again: a gets the place after 0.8 s
        self.assertGreater(time.monotonic() - t0, 0.6)
        self.assertGreater(f["reserve_holds"], 0)
        self.assertEqual(self.e.fair_view()["reserve_s"], 0.8)
        for r in (2, 3):
            self.release[r].set()
        for t in th:
            t.join(5)
        self.assertEqual(f["total"], 0)

    def test_no_reserve_for_a_prompt_start_user_or_when_off(self):
        f = self.e._fair_state()
        th = [self.go(1, "u:b")]
        self.wait_for(lambda f: f["total"] == 1)
        self.release[1].set()
        th[0].join(5)
        self.assertEqual(f["idle_since"], {})                       # off: nothing kept
        self.e._swarm_config({"STRATA_FAIR_USERS": "1", "STRATA_FAIR_RESERVE_S": "5"})
        th.append(self.go(2, "p:abc"))
        self.wait_for(lambda f: f["total"] == 1)
        self.release[2].set()
        th[-1].join(5)
        self.assertEqual(f["idle_since"], {})                       # a prompt-start hash is no named user
        for t in th:
            t.join(5)


class FairService(tp.FakeEngineCase):
    """The real StrataEngine and Service with STRATA_FAIR_USERS on, the fake engine behind them (2 slots)."""

    def post(self, text, user=None, header=None, max_tokens=64):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "reasoning_effort": "none"}
        if user:
            body["user"] = user
        h = {"Content-Type": "application/json"}
        if header:
            h["X-Strata-User"] = header
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers=h)
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def test_off_nothing_shows(self):
        self.start(2)
        self.assertEqual(self.post("hello", user="alice")["choices"][0]["message"]["content"], "ok, done.")
        self.assertNotIn("fair", self.get("/metrics")["live"])
        self.assertIsNone(self.engine.__dict__.get("_fair"))

    def test_a_second_user_is_not_queued_behind_a_swarm(self):
        with mock.patch.object(server, "FAIR_USERS", True):
            self.start(2)
            res, th = {}, []

            def go(k, text, **kw):
                t0 = time.time()
                r = self.post(text, **kw)
                res[k] = (r["choices"][0]["message"]["content"], time.time() - t0)
            for i in range(4):                                       # alice's swarm: 4 long replies on 2 slots
                th.append(threading.Thread(target=go, args=(f"a{i}", f"LONGREPLY agent {i}"), kwargs={"user": "alice"}))
                th[-1].start()
            deadline = time.time() + 10
            while time.time() < deadline and self.engine.fair_view()["waiting"] < 2:
                time.sleep(0.01)
            th.append(threading.Thread(target=go, args=("b", "LONGREPLY chat"), kwargs={"header": "bob"}))
            th[-1].start()
            for t in th:
                t.join(60)
            self.assertEqual(sorted(res), ["a0", "a1", "a2", "a3", "b"])
            self.assertTrue(all(v[0].startswith("ok, ") for v in res.values()), res)
            fair = self.get("/metrics")["live"]["fair"]
            self.assertEqual(fair["users"]["u:alice"]["admitted"], 4)
            self.assertEqual(fair["users"]["u:bob"]["admitted"], 1)
            self.assertEqual(fair["order"][:3], ["u:alice", "u:alice", "u:bob"], fair)   # bob: the first free slot
            self.assertEqual((fair["inflight"], fair["waiting"]), (0, 0))
            self.assertFalse(any(self.engine.slot_busy))

    def test_anthropic_metadata_and_the_prompt_start(self):
        with mock.patch.object(server, "FAIR_USERS", True):
            self.start(2)
            body = {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}],
                    "metadata": {"user_id": "user_ab_account_cd_session_ef"}}
            req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=60) as r:
                self.assertEqual(r.status, 200)
            self.post("hello there")                                 # nobody named: the prompt's start
            users = self.get("/metrics")["live"]["fair"]["users"]
            self.assertEqual(users["u:user_ab_account_cd"]["admitted"], 1)
            self.assertEqual(sum(1 for u in users if u.startswith("p:")), 1, users)


if __name__ == "__main__":
    unittest.main()
