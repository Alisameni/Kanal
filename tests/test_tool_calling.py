"""Offline tests for the tool-calling loop added to main.py.

The OpenAI client and the search HTTP layer are always replaced with mocks, so
no request ever leaves the process and these tests pass with the network
unplugged. The key used here is an obvious fake placeholder; no value is read
from .env.
"""

import json
import os
import socket
import unittest
import urllib.error
from types import SimpleNamespace
from unittest import mock

from openai import APIConnectionError, APIStatusError

import main
import web_search

FAKE_KEY = "test-key-not-a-real-secret"

# Patchers that make any socket or HTTP use fail loudly; see setUpModule().
_GUARDS = []


def setUpModule():
    """Keep the whole module offline, not just the tests that remember to mock.

    These tests replace the SDK client and the Tavily HTTP layer, so a real
    request would mean a bug in a test. Patching the low-level entry points turns
    such a bug into a clear failure instead of a live call.
    """
    for target in (
        "socket.socket",
        "socket.create_connection",
        "urllib.request.urlopen",
    ):
        patcher = mock.patch(target, side_effect=AssertionError(f"{target} was used"))
        patcher.start()
        _GUARDS.append(patcher)


def tearDownModule():
    for patcher in reversed(_GUARDS):
        patcher.stop()
    _GUARDS.clear()


def completion(content=None, tool_calls=None):
    """A minimal stand-in for a ChatCompletion response."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_call(
    call_id="call_1",
    name=web_search.TOOL_NAME,
    arguments='{"query": "tehran weather"}',
):
    """A minimal stand-in for one ChatCompletionMessageToolCall."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def fake_client(*responses):
    """A fake client that replays the given responses in order.

    Every call's messages are snapshotted, because the loop keeps appending to
    the list it passes in; without the snapshot an earlier call would appear to
    contain the later tool messages too.

    ``with_options`` is what the time-boxed tool path uses to set a per-request
    timeout and drop retries. The real SDK returns a new client view that shares
    the HTTP client; here the overrides are recorded and the same object is
    returned, so every call lands in one log.
    """
    client = mock.Mock()
    client.calls = []
    client.overrides = []
    pending = list(responses)

    def create(**kwargs):
        snapshot = [dict(message) for message in kwargs["messages"]]
        client.calls.append(dict(kwargs, messages=snapshot))
        if not pending:
            raise AssertionError("the model was called more often than expected")
        return pending.pop(0)

    def with_options(**overrides):
        client.overrides.append(dict(overrides))
        return client

    client.chat.completions.create.side_effect = create
    client.with_options.side_effect = with_options
    return client


def sent_messages(client, index=-1):
    """The messages of one recorded call."""
    return client.calls[index]["messages"]


def last_kwargs(client):
    return client.chat.completions.create.call_args_list[-1].kwargs


def http_error(case, code, message="boom"):
    """Build an HTTPError whose body is closed once the test finishes."""
    error = urllib.error.HTTPError(
        web_search.TAVILY_SEARCH_URL, code, message, {}, None
    )
    case.addCleanup(error.close)
    return error


def provider_error(code=400, message="tools are not supported"):
    """An APIStatusError like the one a provider sends for an unsupported field."""
    return APIStatusError(
        message,
        response=mock.Mock(status_code=code, headers={}),
        body=None,
    )


class FakeClock:
    """A monotonic-style clock that only moves when a test moves it.

    The time budget is tested by advancing this clock inside a mocked step, so no
    test ever sleeps and every result is deterministic.
    """

    def __init__(self, start=0.0):
        self.time = start

    def __call__(self):
        return self.time

    def advance(self, seconds):
        self.time += seconds


def slow_client(client, clock, elapsed, times=1):
    """Make the fake client's first ``times`` calls spend ``elapsed`` seconds."""
    original = client.chat.completions.create.side_effect
    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        if len(seen) <= times:
            clock.advance(elapsed)
        return original(**kwargs)

    client.chat.completions.create.side_effect = create
    return client


def spend_search_time(clock, elapsed):
    """Patch execute_tool so that one search spends ``elapsed`` seconds."""

    def run(*args, **kwargs):
        clock.advance(elapsed)
        return '{"results": []}'

    return mock.patch.object(main.web_search, "execute_tool", side_effect=run)


class SearchEnabledTestCase(unittest.TestCase):
    """Base case: a key is configured, so web_search is offered to the model."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {web_search.SEARCH_API_KEY_ENV: FAKE_KEY})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assertTrue(web_search.is_available())


class PlainReplyTests(SearchEnabledTestCase):
    def test_a_plain_reply_needs_no_tool_and_one_call(self):
        client = fake_client(completion(content="سلام!"))

        reply = main.get_reply_with_tools(
            client, "test-model", [{"role": "user", "content": "hi"}]
        )

        self.assertEqual(reply, "سلام!")
        self.assertEqual(client.chat.completions.create.call_count, 1)
        tools = last_kwargs(client)["tools"]
        self.assertEqual([tool["function"]["name"] for tool in tools], ["web_search"])
        self.assertEqual(
            [message["role"] for message in sent_messages(client)], ["user"]
        )

    def test_an_empty_model_answer_still_returns_none(self):
        client = fake_client(completion(content=None))

        reply = main.get_reply_with_tools(
            client, "test-model", [{"role": "user", "content": "hi"}]
        )

        self.assertIsNone(reply)
        self.assertEqual(client.chat.completions.create.call_count, 1)


class ToolExecutionTests(SearchEnabledTestCase):
    def test_a_valid_tool_call_is_executed_with_its_validated_arguments(self):
        client = fake_client(
            completion(tool_calls=[tool_call()]), completion(content="پاسخ نهایی")
        )
        tool_output = json.dumps({"query": "tehran weather", "results": []})

        with mock.patch.object(
            main.web_search, "execute_tool", return_value=tool_output
        ) as execute:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

        self.assertEqual(reply, "پاسخ نهایی")
        execute.assert_called_once_with(
            web_search.TOOL_NAME,
            '{"query": "tehran weather"}',
            # The search gets the module cap: there is plenty of budget left.
            timeout=web_search.REQUEST_TIMEOUT_SECONDS,
        )

    def test_the_tool_result_is_sent_back_and_the_final_reply_returned(self):
        client = fake_client(
            completion(tool_calls=[tool_call()]), completion(content="پاسخ نهایی")
        )
        tool_output = json.dumps(
            {
                "query": "tehran weather",
                "results": [
                    {"title": "t", "url": "https://example.com", "snippet": "s"}
                ],
            }
        )

        with mock.patch.object(
            main.web_search, "execute_tool", return_value=tool_output
        ):
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

        self.assertEqual(reply, "پاسخ نهایی")
        self.assertEqual(client.chat.completions.create.call_count, 2)

        second_call = sent_messages(client)
        assistant_turn = [m for m in second_call if m["role"] == "assistant"][-1]
        self.assertEqual(assistant_turn["tool_calls"][0]["id"], "call_1")
        self.assertEqual(
            assistant_turn["tool_calls"][0]["function"]["name"], web_search.TOOL_NAME
        )
        self.assertEqual(
            assistant_turn["tool_calls"][0]["function"]["arguments"],
            '{"query": "tehran weather"}',
        )

        tool_turn = second_call[-1]
        self.assertEqual(tool_turn["role"], "tool")
        self.assertEqual(tool_turn["tool_call_id"], "call_1")
        self.assertEqual(tool_turn["content"], tool_output)

    def test_a_second_round_can_run_another_search(self):
        client = fake_client(
            completion(tool_calls=[tool_call("call_1")]),
            completion(
                tool_calls=[tool_call("call_2", arguments='{"query": "more"}')]
            ),
            completion(content="خلاصه"),
        )

        with mock.patch.object(
            main.web_search, "execute_tool", return_value='{"results": []}'
        ) as execute:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

        self.assertEqual(reply, "خلاصه")
        self.assertEqual(execute.call_count, 2)
        tool_turns = [m for m in sent_messages(client) if m["role"] == "tool"]
        self.assertEqual(
            [m["tool_call_id"] for m in tool_turns], ["call_1", "call_2"]
        )


class InvalidToolCallTests(SearchEnabledTestCase):
    def test_invalid_json_arguments_are_reported_to_the_model(self):
        client = fake_client(
            completion(tool_calls=[tool_call(arguments="{not json")]),
            completion(content="بدون جستوجو"),
        )

        with mock.patch.object(main.web_search, "post_json") as post:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

        post.assert_not_called()
        self.assertEqual(reply, "بدون جستوجو")
        payload = json.loads(sent_messages(client)[-1]["content"])
        self.assertIn("error", payload)
        self.assertNotIn(FAKE_KEY, json.dumps(payload))

    def test_an_unknown_tool_name_is_never_executed(self):
        client = fake_client(
            completion(tool_calls=[tool_call(name="run_shell")]),
            completion(content="نمیتوانم"),
        )

        with mock.patch.object(main.web_search, "search") as search:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

        search.assert_not_called()
        self.assertEqual(reply, "نمیتوانم")
        payload = json.loads(sent_messages(client)[-1]["content"])
        self.assertIn("unknown tool", payload["error"])

    def test_search_failures_become_a_controlled_message_for_the_model(self):
        failures = {
            "timeout": TimeoutError("slow"),
            "http": http_error(self, 503, "down"),
            "network": urllib.error.URLError(socket.gaierror(-2, "no dns")),
            "unreadable": None,
        }

        for case, error in failures.items():
            with self.subTest(case=case):
                client = fake_client(
                    completion(tool_calls=[tool_call()]),
                    completion(content="بدون جستوجو"),
                )
                if error is None:
                    # A body the service should never send.
                    post = mock.patch.object(
                        main.web_search, "post_json", return_value="not a mapping"
                    )
                else:
                    post = mock.patch.object(
                        main.web_search, "post_json", side_effect=error
                    )

                with post:
                    reply = main.get_reply_with_tools(
                        client, "test-model", [{"role": "user", "content": "?"}]
                    )

                self.assertEqual(reply, "بدون جستوجو")
                content = sent_messages(client)[-1]["content"]
                self.assertIn("error", json.loads(content))
                self.assertNotIn(FAKE_KEY, content)

    def test_a_broken_tool_does_not_break_the_conversation(self):
        client = fake_client(
            completion(tool_calls=[tool_call()]), completion(content="ادامه میدهم")
        )

        with mock.patch.object(
            main.web_search, "execute_tool", side_effect=RuntimeError("boom")
        ):
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

        self.assertEqual(reply, "ادامه میدهم")
        payload = json.loads(sent_messages(client)[-1]["content"])
        self.assertIn("error", payload)


class ToolBudgetTests(SearchEnabledTestCase):
    def test_the_round_budget_ends_with_a_tool_free_answer(self):
        client = fake_client(
            completion(tool_calls=[tool_call("call_1")]),
            completion(tool_calls=[tool_call("call_2")]),
            completion(content="پاسخ بدون ابزار"),
        )

        with mock.patch.object(
            main.web_search, "execute_tool", return_value='{"results": []}'
        ) as execute:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

        self.assertEqual(main.MAX_TOOL_ROUNDS, 2)
        self.assertEqual(reply, "پاسخ بدون ابزار")
        self.assertEqual(execute.call_count, main.MAX_TOOL_ROUNDS)
        self.assertEqual(
            client.chat.completions.create.call_count, main.MAX_TOOL_ROUNDS + 1
        )
        # The final turn must not offer tools, or the model could ask again.
        self.assertNotIn("tools", last_kwargs(client))

    def test_an_endless_tool_loop_still_terminates(self):
        client = mock.Mock()
        client.chat.completions.create.return_value = completion(
            tool_calls=[tool_call()]
        )
        # The time-boxed path asks for a client view; a bare mock returns itself,
        # so the same response is replayed until a cap stops the loop.
        client.with_options.return_value = client

        with mock.patch.object(
            main.web_search, "execute_tool", return_value='{"results": []}'
        ) as execute:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

        self.assertIsNone(reply)
        self.assertEqual(execute.call_count, main.MAX_TOOL_ROUNDS)

    def test_parallel_tool_calls_are_capped_by_MAX_TOOL_CALLS(self):
        calls = [tool_call(f"call_{index}") for index in range(main.MAX_TOOL_CALLS + 2)]
        client = fake_client(completion(tool_calls=calls), completion(content="done"))

        with mock.patch.object(
            main.web_search, "execute_tool", return_value='{"results": []}'
        ) as execute:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

        self.assertEqual(reply, "done")
        self.assertEqual(execute.call_count, main.MAX_TOOL_CALLS)
        # Every requested call still gets an answer, so the API contract holds.
        tool_turns = [m for m in sent_messages(client) if m["role"] == "tool"]
        self.assertEqual(len(tool_turns), len(calls))
        self.assertIn("error", json.loads(tool_turns[-1]["content"]))


class NoToolMessagesInHistoryTests(SearchEnabledTestCase):
    def test_tool_messages_never_enter_the_public_history(self):
        client = fake_client(
            completion(tool_calls=[tool_call()]), completion(content="پاسخ نهایی")
        )

        with mock.patch.object(
            main.web_search,
            "execute_tool",
            return_value='{"results": [{"title": "t", "url": "https://e.com"}]}',
        ):
            reply, history = main.chat("هوا چطور است؟", [], client, "test-model")

        self.assertEqual(reply, "پاسخ نهایی")
        self.assertEqual(
            history,
            [
                {"role": "user", "content": "هوا چطور است؟"},
                {"role": "assistant", "content": "پاسخ نهایی"},
            ],
        )
        for turn in history:
            self.assertEqual(sorted(turn), ["content", "role"])
            self.assertIn(turn["role"], ("user", "assistant"))
        self.assertNotIn("tool_call_id", json.dumps(history))
        self.assertNotIn("https://e.com", json.dumps(history))

    def test_a_failed_tool_search_still_produces_a_normal_turn(self):
        client = fake_client(
            completion(tool_calls=[tool_call()]), completion(content="بدون جستوجو")
        )

        with mock.patch.object(
            main.web_search, "post_json", side_effect=TimeoutError("slow")
        ):
            reply, history = main.chat("سؤال", [], client, "test-model")

        self.assertEqual(reply, "بدون جستوجو")
        self.assertEqual([turn["role"] for turn in history], ["user", "assistant"])

    def test_the_history_limit_is_still_applied(self):
        client = fake_client(completion(content="ok"))
        old_history = [
            {"role": "user", "content": f"q{index}"}
            for index in range(main.HISTORY_LIMIT)
        ]

        reply, history = main.chat("new", old_history, client, "test-model")

        self.assertEqual(reply, "ok")
        self.assertEqual(len(history), main.HISTORY_LIMIT)
        self.assertEqual(history[-1], {"role": "assistant", "content": "ok"})


class SearchDisabledTests(unittest.TestCase):
    """Without SEARCH_API_KEY the chatbot must behave exactly as before."""

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {web_search.SEARCH_API_KEY_ENV: ""})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_tool_is_not_offered_and_the_original_path_runs(self):
        self.assertFalse(web_search.is_available())
        client = fake_client(completion(content="ok"))

        with mock.patch.object(main, "get_reply_with_tools") as with_tools:
            reply, history = main.chat("hi", [], client, "test-model")

        with_tools.assert_not_called()
        self.assertEqual(reply, "ok")
        self.assertEqual(
            history,
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
            ],
        )
        # No tools field and no extra request: the request is the old one.
        self.assertEqual(sorted(last_kwargs(client)), ["messages", "model"])
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_an_empty_model_answer_keeps_the_history_unchanged(self):
        client = fake_client(completion(content=None))

        reply, history = main.chat("hi", [], client, "test-model")

        self.assertIsNone(reply)
        self.assertEqual(history, [])

    def test_a_provider_error_still_propagates_unchanged(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = RuntimeError("provider down")

        with self.assertRaises(RuntimeError):
            main.chat("hi", [], client, "test-model")

    def test_the_plain_path_is_never_time_boxed(self):
        # No key: neither a budget nor a per-request timeout is applied here.
        client = fake_client(completion(content="ok"))

        with mock.patch.object(client, "with_options") as with_options:
            main.chat("hi", [], client, "test-model")

        with_options.assert_not_called()


class NoNetworkTests(SearchEnabledTestCase):
    def test_a_full_tool_flow_never_opens_a_socket(self):
        client = fake_client(
            completion(tool_calls=[tool_call()]), completion(content="پاسخ نهایی")
        )
        tavily_body = {
            "results": [{"title": "t", "url": "https://example.com", "content": "c"}]
        }

        with (
            mock.patch("socket.socket", side_effect=AssertionError("socket opened")),
            mock.patch(
                "socket.create_connection",
                side_effect=AssertionError("connection opened"),
            ),
            mock.patch(
                "urllib.request.urlopen", side_effect=AssertionError("HTTP request")
            ),
            mock.patch.object(
                web_search, "post_json", return_value=tavily_body
            ) as post,
        ):
            reply, history = main.chat("هوا؟", [], client, "test-model")

        self.assertEqual(reply, "پاسخ نهایی")
        post.assert_called_once()
        self.assertEqual(history[-1], {"role": "assistant", "content": "پاسخ نهایی"})

    def test_the_real_http_layer_works_against_a_mocked_urlopen(self):
        client = fake_client(
            completion(tool_calls=[tool_call(arguments='{"query": "tehran"}')]),
            completion(content="پاسخ نهایی"),
        )
        body = json.dumps(
            {
                "results": [
                    {
                        "title": "Tehran",
                        "url": "https://example.com/news",
                        "content": "sunny",
                    }
                ]
            }
        ).encode("utf-8")

        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = body
            reply, history = main.chat("هوا؟", [], client, "test-model")

        self.assertEqual(reply, "پاسخ نهایی")
        self.assertEqual(history[-1], {"role": "assistant", "content": "پاسخ نهایی"})

        tool_turn = [m for m in sent_messages(client) if m["role"] == "tool"][-1]
        payload = json.loads(tool_turn["content"])
        self.assertEqual(payload["query"], "tehran")
        self.assertEqual(payload["results"][0]["url"], "https://example.com/news")
        self.assertEqual(
            urlopen.call_args.args[0].full_url, web_search.TAVILY_SEARCH_URL
        )


class TimeBudgetTests(SearchEnabledTestCase):
    """The tool turn runs under one shared budget, with no real sleeping."""

    def test_the_constants_leave_room_under_the_frontend_timeout(self):
        # frontend/src/api/chat.ts aborts POST /chat after 45s.
        self.assertLess(main.TOOL_DEADLINE_SECONDS, 45)
        self.assertLess(main.TOOL_ANSWER_RESERVE_SECONDS, main.TOOL_DEADLINE_SECONDS)
        self.assertLess(main.TOOL_STEP_TIMEOUT_SECONDS, 20)  # build_client default
        self.assertGreater(main.TOOL_MIN_STEP_SECONDS, 0)

    def test_a_step_timeout_is_the_smaller_of_the_time_left_and_the_cap(self):
        clock = FakeClock()
        deadline = main.TOOL_DEADLINE_SECONDS
        cap = main.TOOL_STEP_TIMEOUT_SECONDS

        # Plenty of time left: the per-step cap wins.
        self.assertEqual(main.step_timeout(deadline, clock, cap), cap)

        # Time runs short: what is left wins.
        clock.advance(deadline - 3.0)
        self.assertEqual(main.step_timeout(deadline, clock, cap), 3.0)

        # The reserve for the final answer is held back.
        self.assertIsNone(main.step_timeout(deadline, clock, cap, reserve=3.0))

        # The deadline has passed: no step is started at all.
        clock.advance(5.0)
        self.assertIsNone(main.step_timeout(deadline, clock, cap))

    def test_a_model_call_is_bounded_by_the_time_left(self):
        clock = FakeClock()
        client = fake_client(
            completion(tool_calls=[tool_call()]), completion(content="پاسخ")
        )

        with spend_search_time(clock, 14.0):
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}], now=clock
            )

        self.assertEqual(reply, "پاسخ")
        self.assertEqual(
            client.overrides,
            [
                {"timeout": main.TOOL_STEP_TIMEOUT_SECONDS, "max_retries": 0},
                # 30 - 8 (reserved) - 14 = 8s left, not another full 12s step.
                {"timeout": 8.0, "max_retries": 0},
            ],
        )
        self.assertEqual(client.chat.completions.create.call_count, 2)


    def test_a_search_is_bounded_by_the_time_left(self):
        clock = FakeClock()
        client = fake_client(
            completion(tool_calls=[tool_call()]), completion(content="پاسخ")
        )
        # The first model call itself spends most of the round budget.
        slow_client(client, clock, 19.0)

        with mock.patch.object(
            main.web_search, "execute_tool", return_value='{"results": []}'
        ) as execute:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}], now=clock
            )

        self.assertEqual(reply, "پاسخ")
        # 30 - 8 (reserved) - 19 = 3s are left for the search.
        self.assertEqual(execute.call_args.kwargs["timeout"], 3.0)
        self.assertLessEqual(
            execute.call_args.kwargs["timeout"], web_search.REQUEST_TIMEOUT_SECONDS
        )

    def test_a_search_left_without_time_is_skipped_not_attempted(self):
        clock = FakeClock()
        client = fake_client(
            completion(tool_calls=[tool_call()]), completion(content="پاسخ")
        )
        # The model call leaves 1s of the round budget: not enough to search.
        spent = main.TOOL_DEADLINE_SECONDS - main.TOOL_ANSWER_RESERVE_SECONDS - 1
        slow_client(client, clock, spent)

        with spend_search_time(clock, 0.0) as execute:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}], now=clock
            )

        execute.assert_not_called()
        self.assertEqual(reply, "پاسخ")
        # The model still gets an answer for every call it asked for.
        tool_turn = [m for m in sent_messages(client) if m["role"] == "tool"][-1]
        self.assertIn("time budget", json.loads(tool_turn["content"])["error"])

    def test_the_loop_stops_when_the_whole_budget_is_gone(self):
        clock = FakeClock()
        # Only one response is queued: a second model call would fail with the
        # fake's own error instead of a deadline, so this also proves no round
        # was started after the budget ran out.
        client = fake_client(completion(tool_calls=[tool_call()]))

        with spend_search_time(clock, main.TOOL_DEADLINE_SECONDS + 5):
            with self.assertRaises(main.ToolDeadlineExceeded):
                main.get_reply_with_tools(
                    client, "test-model", [{"role": "user", "content": "?"}], now=clock
                )

        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_no_further_round_starts_but_the_answer_gets_the_rest(self):
        clock = FakeClock()
        client = fake_client(
            completion(tool_calls=[tool_call()]), completion(content="پاسخ آخر")
        )

        with spend_search_time(clock, main.TOOL_DEADLINE_SECONDS - 3.0):
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}], now=clock
            )

        # The round budget is spent, so only the reserved final answer is asked
        # for, and it gets the 3s that are left instead of a fresh step cap.
        self.assertEqual(reply, "پاسخ آخر")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertNotIn("tools", last_kwargs(client))
        self.assertEqual(client.overrides[-1], {"timeout": 3.0, "max_retries": 0})

    def test_the_round_and_tool_caps_still_apply_while_time_is_left(self):
        clock = FakeClock()
        client = fake_client(
            completion(tool_calls=[tool_call("call_1"), tool_call("call_2")]),
            completion(tool_calls=[tool_call("call_3"), tool_call("call_4")]),
            completion(content="پاسخ بدون ابزار"),
        )

        with mock.patch.object(
            main.web_search, "execute_tool", return_value='{"results": []}'
        ) as execute:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}], now=clock
            )

        # The clock never moved: the caps, not the budget, ended the loop.
        self.assertEqual(clock(), 0.0)
        self.assertEqual(reply, "پاسخ بدون ابزار")
        self.assertEqual(execute.call_count, main.MAX_TOOL_CALLS)
        self.assertEqual(
            client.chat.completions.create.call_count, main.MAX_TOOL_ROUNDS + 1
        )

    def test_a_deadline_error_is_reported_like_the_other_failures(self):
        message = main.describe_error(main.ToolDeadlineExceeded("out of time"))

        self.assertIn("took too long", message)
        self.assertNotIn(FAKE_KEY, message)


class ProviderToolRejectionTests(SearchEnabledTestCase):
    """A provider that refuses tool calling must fail loudly, never silently."""

    def test_a_rejected_tools_parameter_is_not_swallowed(self):
        client = fake_client()
        client.chat.completions.create.side_effect = provider_error()

        with self.assertRaises(APIStatusError):
            main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

        # One attempt: no retry and no silent fallback request without tools.
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_chat_lets_a_provider_error_reach_its_caller(self):
        client = fake_client()
        client.chat.completions.create.side_effect = provider_error(429)

        with self.assertRaises(APIStatusError):
            main.chat("hi", [], client, "test-model")

        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_the_deadline_error_is_not_a_provider_error(self):
        # It must not be caught by the provider branches in describe_error() or
        # in server.py, which is why it is a plain Exception.
        self.assertFalse(issubclass(main.ToolDeadlineExceeded, APIStatusError))
        self.assertFalse(issubclass(main.ToolDeadlineExceeded, APIConnectionError))

    def test_the_client_defaults_are_only_overridden_inside_the_tool_path(self):
        client = fake_client(completion(content="ok"))

        main.chat("hi", [], client, "test-model")

        self.assertEqual(
            client.overrides,
            [{"timeout": main.TOOL_STEP_TIMEOUT_SECONDS, "max_retries": 0}],
        )
        # The request itself is still the plain one plus the tool list.
        self.assertEqual(sorted(last_kwargs(client)), ["messages", "model", "tools"])


if __name__ == "__main__":
    unittest.main()




