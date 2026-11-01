"""Offline tests for the tool-calling loop added to main.py.

The OpenAI client and the search HTTP layer are always replaced with mocks, so
no request ever leaves the process and these tests pass with the network
unplugged. The key used here is an obvious fake placeholder; no value is read
from .env.

The save-research tests in this file only ever build a *proposal*: every case
that exercises it patches save_tool.write_proposal so that reaching the one
function that writes a file becomes a test failure.
"""

import ast
import contextvars
import inspect
import json
import os
import socket
import threading
import unittest
import urllib.error
from types import SimpleNamespace
from unittest import mock

from openai import APIConnectionError, APIStatusError, APITimeoutError

import main
import save_tool
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

    An entry of ``responses`` may also be an exception, which is raised instead of
    returned; that is how a call that times out mid-turn is simulated.
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
        response = pending.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

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


# One real URL, exactly as web_search would return it. The save tests use it as
# the only source a proposal is allowed to cite.
REAL_URL = "https://example.com/weather"


def search_call(call_id="call_search", query="tehran weather"):
    """A web_search tool call with a valid query."""
    return tool_call(call_id, arguments=json.dumps({"query": query}))


def search_result(urls=(REAL_URL,), query="tehran weather"):
    """The JSON string web_search returns for a successful search."""
    return json.dumps(
        {
            "query": query,
            "results": [
                {"title": f"Source {index}", "url": url, "snippet": "snippet"}
                for index, url in enumerate(urls)
            ],
        },
        ensure_ascii=False,
    )


def distinct_search_call(index, query=None):
    """A web_search call with its own query, so a repeat cannot be mistaken for one.

    The loop searches an identical query only once per turn, so a test that needs
    several real searches has to give each call its own query. Each call also gets
    its own id, because the tool answers have to stay tellable apart.
    """
    return search_call(f"call_{index}", query=query or f"tehran weather {index}")


def save_arguments(**overrides):
    """Valid save_research arguments, with ``overrides`` applied on top."""
    arguments = {
        "title": "Tehran weather",
        "summary": "خلاصه یافتهها",
        "analysis": "تحلیل کامل هوا",
        "sources": [{"title": "Example", "url": REAL_URL}],
    }
    arguments.update(overrides)
    return arguments


def save_call(call_id="call_save", **overrides):
    """A save_research tool call whose arguments are valid by default."""
    return tool_call(
        call_id,
        name=save_tool.TOOL_NAME,
        arguments=json.dumps(save_arguments(**overrides), ensure_ascii=False),
    )


def tool_turns(client):
    """The tool messages of the last recorded call, in the order they were sent."""
    return [message for message in sent_messages(client) if message["role"] == "tool"]


def offered_tools(client, index):
    """The tool names offered in one recorded model call."""
    tools = client.calls[index].get("tools") or []
    return [tool["function"]["name"] for tool in tools]


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
    def test_the_round_budget_ends_with_an_answer_that_still_offers_the_tools(self):
        client = fake_client(
            completion(tool_calls=[distinct_search_call(1)]),
            completion(tool_calls=[distinct_search_call(2)]),
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
        # The answer round keeps offering the tools. Taking them away while the
        # system prompt still allows save_research is what made the model write its
        # call into the message text instead of using message.tool_calls.
        self.assertEqual(offered_tools(client, -1), [web_search.TOOL_NAME])

    def test_an_endless_tool_loop_still_terminates(self):
        client = mock.Mock()
        # Every round asks for a fresh query, so it is the search budget that runs
        # out and not the "already searched" rule, and the model never stops asking.
        rounds = {"count": 0}

        def create(**kwargs):
            rounds["count"] += 1
            return completion(tool_calls=[distinct_search_call(rounds["count"])])

        client.chat.completions.create.side_effect = create
        # The time-boxed path asks for a client view; a bare mock returns itself,
        # so the same kind of response is replayed until a cap stops the loop.
        client.with_options.return_value = client

        with mock.patch.object(
            main.web_search, "execute_tool", return_value='{"results": []}'
        ) as execute:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

        self.assertIsNone(reply)
        # Both caps hold: the searches stop at their own budget, and the model calls
        # stop at the work rounds plus the answer rounds, so no loop can run forever.
        self.assertEqual(execute.call_count, main.MAX_TOOL_CALLS)
        self.assertEqual(
            client.chat.completions.create.call_count,
            main.MAX_TOOL_ROUNDS + main.MAX_ANSWER_ROUNDS,
        )

    def test_parallel_tool_calls_are_capped_by_MAX_TOOL_CALLS(self):
        calls = [
            distinct_search_call(index) for index in range(main.MAX_TOOL_CALLS + 2)
        ]
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
        self.assertEqual(
            json.loads(tool_turns[-1]["content"])["error"], main.SEARCH_BUDGET_SPENT
        )


# A reply of the shape the provider really produced: its own tool-call syntax
# written into message.content instead of a structured message.tool_calls.
MARKUP_REPLY = (
    "<||DSML|| calls>"
    '<||DSML|| invoke name="save_research">'
    '<||DSML|| parameter name="title" string="true">Report</||DSML|| parameter>'
    '<||DSML|| parameter name="sources" string="true">['
    + REAL_URL
    + "]</||DSML|| parameter>"
    "</||DSML|| invoke>"
    "</||DSML|| calls>"
)


class ToolCallMarkupTests(SearchEnabledTestCase):
    """The model's textual tool-call markup is never an answer and never a call.

    The tool-call syntax belongs in message.tool_calls. When the provider writes it
    into message.content instead, the loop must never hand that text to the user,
    must never read it back as an executable call, and must still terminate.
    """

    def run_turn(self, client):
        with mock.patch.object(
            main.web_search, "execute_tool", return_value=search_result()
        ):
            return main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

    def markup_only_client(self):
        """A client whose every round answers with nothing but the markup."""
        return fake_client(
            *(
                completion(content=MARKUP_REPLY)
                for _ in range(main.MAX_TOOL_ROUNDS + main.MAX_ANSWER_ROUNDS)
            )
        )

    def test_the_helpers_recognize_the_markup(self):
        self.assertTrue(main.is_tool_call_markup(MARKUP_REPLY))
        self.assertTrue(main.is_tool_call_markup("some text " + MARKUP_REPLY))
        self.assertFalse(main.is_tool_call_markup("a normal answer"))
        self.assertFalse(main.is_tool_call_markup(""))
        self.assertFalse(main.is_tool_call_markup(None))

    def test_the_markup_is_never_returned_to_the_caller(self):
        self.assertEqual(main.reply_text(MARKUP_REPLY), main.TOOL_MARKUP_ERROR)
        self.assertNotIn(main.TOOL_CALL_MARKUP, main.reply_text(MARKUP_REPLY))
        self.assertEqual(main.reply_text("a normal answer"), "a normal answer")
        self.assertIsNone(main.reply_text(None))

    def test_a_reply_that_is_only_markup_never_reaches_the_user(self):
        client = self.markup_only_client()

        with main.collect_saves() as proposals:
            reply = self.run_turn(client)

        # A short, readable message instead of the markup.
        self.assertEqual(reply, main.TOOL_MARKUP_ERROR)
        self.assertNotIn(main.TOOL_CALL_MARKUP, reply)
        self.assertEqual(proposals, [])

    def test_the_markup_recovery_is_bounded(self):
        client = self.markup_only_client()

        with main.collect_saves():
            self.run_turn(client)

        # fake_client() fails the turn if the loop asks for one more response, so
        # this also proves the retries stop at the two round budgets.
        self.assertEqual(
            client.chat.completions.create.call_count,
            main.MAX_TOOL_ROUNDS + main.MAX_ANSWER_ROUNDS,
        )

    def test_the_markup_is_never_parsed_into_a_call(self):
        client = self.markup_only_client()

        with mock.patch.object(main, "execute_save_tool") as save:
            with mock.patch.object(
                main.web_search, "execute_tool", return_value=search_result()
            ) as search:
                with main.collect_saves() as proposals:
                    self.run_turn(client)

        # The markup names save_research and even carries a real source, and still
        # nothing is executed and no proposal is recorded: text is never read back
        # as a call.
        save.assert_not_called()
        search.assert_not_called()
        self.assertEqual(proposals, [])

    def test_a_round_after_the_markup_can_still_answer(self):
        client = fake_client(
            completion(content=MARKUP_REPLY), completion(content="final answer")
        )

        with main.collect_saves() as proposals:
            reply = self.run_turn(client)

        self.assertEqual(reply, "final answer")
        self.assertEqual(proposals, [])
        # The retry is an ordinary round, so the tools are offered again: that is
        # what lets the model repeat its request on the structured channel.
        self.assertEqual(offered_tools(client, -1), [web_search.TOOL_NAME])

    def test_a_real_call_after_the_markup_is_executed_normally(self):
        client = fake_client(
            completion(content=MARKUP_REPLY),
            completion(tool_calls=[search_call()]),
            completion(content="final answer"),
        )

        # Patched here rather than through run_turn(), so this mock is the one the
        # loop actually calls.
        with mock.patch.object(
            main.web_search, "execute_tool", return_value=search_result()
        ) as execute:
            with main.collect_saves():
                reply = main.get_reply_with_tools(
                    client, "test-model", [{"role": "user", "content": "?"}]
                )

        self.assertEqual(reply, "final answer")
        execute.assert_called_once()


class SearchQueryDedupeTests(SearchEnabledTestCase):
    """An identical query is searched once per turn; different ones all run."""

    def run_turn(self, client):
        with mock.patch.object(
            main.web_search, "execute_tool", return_value=search_result()
        ) as execute:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )
        return reply, execute

    def test_the_query_key_normalizes_whitespace_and_case(self):
        self.assertEqual(
            main.search_query_key('{"query": "  Tehran   weather "}'),
            "tehran weather",
        )
        self.assertEqual(
            main.search_query_key({"query": "TEHRAN WEATHER"}), "tehran weather"
        )
        self.assertEqual(
            main.search_query_key(b'{"query": "tehran weather"}'), "tehran weather"
        )

    def test_an_unreadable_query_yields_no_key(self):
        # None means "pass the call to the tool unchanged", never "block it".
        self.assertIsNone(main.search_query_key("{not json"))
        self.assertIsNone(main.search_query_key('{"query": ""}'))
        self.assertIsNone(main.search_query_key('{"other": 1}'))
        self.assertIsNone(main.search_query_key(["not", "a", "mapping"]))
        self.assertIsNone(main.search_query_key(None))

    def test_an_identical_query_is_searched_once_but_still_answered(self):
        client = fake_client(
            completion(tool_calls=[search_call("call_1"), search_call("call_2")]),
            completion(content="done"),
        )

        reply, execute = self.run_turn(client)

        self.assertEqual(reply, "done")
        execute.assert_called_once()
        answers = [json.loads(turn["content"]) for turn in tool_turns(client)]
        self.assertEqual(len(answers), 2)
        self.assertIn("already searched", answers[-1]["error"])

    def test_a_repeat_costs_no_search_budget(self):
        client = fake_client(
            completion(tool_calls=[search_call("call_1"), search_call("call_2")]),
            completion(tool_calls=[distinct_search_call(3)]),
            completion(content="done"),
        )

        reply, execute = self.run_turn(client)

        self.assertEqual(reply, "done")
        # The repeat spent nothing, so the later, genuinely different search ran.
        self.assertEqual(execute.call_count, 2)

    def test_whitespace_and_case_cannot_hide_a_repeat(self):
        first = search_call("call_1", query="Tehran weather")
        second = search_call("call_2", query="tehran   WEATHER")
        client = fake_client(
            completion(tool_calls=[first, second]), completion(content="done")
        )

        reply, execute = self.run_turn(client)

        self.assertEqual(reply, "done")
        execute.assert_called_once()

    def test_a_search_past_the_budget_names_the_search_tool(self):
        calls = [
            distinct_search_call(index) for index in range(main.MAX_TOOL_CALLS + 1)
        ]
        client = fake_client(completion(tool_calls=calls), completion(content="done"))

        reply, execute = self.run_turn(client)

        self.assertEqual(reply, "done")
        self.assertEqual(execute.call_count, main.MAX_TOOL_CALLS)
        answers = [json.loads(turn["content"]) for turn in tool_turns(client)]
        self.assertEqual(answers[-1]["error"], main.SEARCH_BUDGET_SPENT)


class ToolOfferingTests(SearchEnabledTestCase):
    """No round of a tool turn is ever tool-free."""

    def test_every_round_offers_the_tools(self):
        client = fake_client(
            completion(tool_calls=[distinct_search_call(1)]),
            completion(tool_calls=[distinct_search_call(2)]),
            completion(tool_calls=[distinct_search_call(3)]),
            completion(content="done"),
        )

        with main.collect_saves():
            with mock.patch.object(
                main.web_search, "execute_tool", return_value=search_result()
            ):
                reply = main.get_reply_with_tools(
                    client, "test-model", [{"role": "user", "content": "?"}]
                )

        self.assertEqual(reply, "done")
        # The system prompt tells the model it may call save_research, so no round
        # may take the tool channel away: that is what made the model write its call
        # into the text instead of using message.tool_calls.
        for index in range(len(client.calls)):
            self.assertIn("tools", client.calls[index])
        self.assertEqual(offered_tools(client, 0), [web_search.TOOL_NAME])
        self.assertEqual(
            offered_tools(client, -1), [web_search.TOOL_NAME, save_tool.TOOL_NAME]
        )


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
        # frontend/src/api/chat.ts aborts POST /chat after 130s.
        self.assertLess(main.TOOL_DEADLINE_SECONDS, 130)
        self.assertLess(main.TOOL_ANSWER_RESERVE_SECONDS, main.TOOL_DEADLINE_SECONDS)
        self.assertGreater(main.TOOL_MIN_STEP_SECONDS, 0)

    def test_the_caps_cover_a_long_slow_answer(self):
        # Sized from the reported failure, not from taste: an analytical request
        # ("search for X and analyse the results") was answered with ~8700
        # characters that took 24.9s to write, and a slow round needed over 25s.
        # A cap under that cuts the user's only answer off and reports a timeout.
        self.assertGreaterEqual(main.TOOL_ANSWER_TIMEOUT_SECONDS, 30)
        self.assertGreaterEqual(main.TOOL_STEP_TIMEOUT_SECONDS, 25)

    def test_the_rounds_still_get_their_share_of_the_budget(self):
        # The reserve is taken out of the same budget, so the rounds must be left
        # with more than one round cap; otherwise a search turn could never reach
        # its second round (or its search).
        left_for_rounds = (
            main.TOOL_DEADLINE_SECONDS - main.TOOL_ANSWER_RESERVE_SECONDS
        )
        self.assertGreater(left_for_rounds, main.TOOL_STEP_TIMEOUT_SECONDS)

    def test_the_answer_reserve_covers_the_answer_cap(self):
        # A reserve smaller than the cap would not protect the answer it is held
        # back for: the answer would still be cut off by its own cap.
        self.assertLessEqual(
            main.TOOL_ANSWER_TIMEOUT_SECONDS, main.TOOL_ANSWER_RESERVE_SECONDS
        )

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
        # The search spends all but 8s of the round budget, so the second call has
        # to live on the time left instead of a fresh round cap.
        spent = main.TOOL_DEADLINE_SECONDS - main.TOOL_ANSWER_RESERVE_SECONDS - 8.0

        with spend_search_time(clock, spent):
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}], now=clock
            )

        self.assertEqual(reply, "پاسخ")
        self.assertEqual(
            client.overrides,
            [
                {"timeout": main.TOOL_STEP_TIMEOUT_SECONDS, "max_retries": 0},
                # Only 8s are left by then, not another full round cap.
                {"timeout": 8.0, "max_retries": 0},
            ],
        )
        self.assertEqual(client.chat.completions.create.call_count, 2)


    def test_a_search_is_bounded_by_the_time_left(self):
        clock = FakeClock()
        client = fake_client(
            completion(tool_calls=[tool_call()]), completion(content="پاسخ")
        )
        # The first model call itself spends all but 3s of the round budget.
        spent = main.TOOL_DEADLINE_SECONDS - main.TOOL_ANSWER_RESERVE_SECONDS - 3.0
        slow_client(client, clock, spent)

        with mock.patch.object(
            main.web_search, "execute_tool", return_value='{"results": []}'
        ) as execute:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}], now=clock
            )

        self.assertEqual(reply, "پاسخ")
        # 3s are left for the search, below its own 8s cap.
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
        # The answer round still offers the tools: the request must never take away
        # a tool the system prompt tells the model it may call.
        self.assertEqual(offered_tools(client, -1), [web_search.TOOL_NAME])
        self.assertEqual(client.overrides[-1], {"timeout": 3.0, "max_retries": 0})

    def test_the_final_answer_gets_the_answer_cap_not_the_round_cap(self):
        clock = FakeClock()
        client = fake_client(
            completion(tool_calls=[tool_call("call_1")]),
            completion(tool_calls=[tool_call("call_2")]),
            completion(content="پاسخ آخر"),
        )

        with mock.patch.object(
            main.web_search, "execute_tool", return_value='{"results": []}'
        ):
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}], now=clock
            )

        # The clock never moved, so the work rounds are cut off by their own cap and
        # the answer round is the only call that reaches the answer cap.
        # That cap is what a slow reply needs: being clipped to the round cap here
        # is what made the turn fail with a timeout instead of answering.
        self.assertEqual(reply, "پاسخ آخر")
        self.assertEqual(offered_tools(client, -1), [web_search.TOOL_NAME])
        self.assertGreaterEqual(
            main.TOOL_ANSWER_TIMEOUT_SECONDS, main.TOOL_STEP_TIMEOUT_SECONDS
        )
        self.assertEqual(
            client.overrides,
            [
                {"timeout": main.TOOL_STEP_TIMEOUT_SECONDS, "max_retries": 0},
                {"timeout": main.TOOL_STEP_TIMEOUT_SECONDS, "max_retries": 0},
                {"timeout": main.TOOL_ANSWER_TIMEOUT_SECONDS, "max_retries": 0},
            ],
        )

    def test_a_round_that_times_out_does_not_fail_the_turn(self):
        # A round can be genuinely slow on this provider: one measured round needed
        # over 25s. Losing the whole turn, and with it the search the earlier round
        # already gathered, would be worse than dropping that one round.
        client = fake_client(
            completion(tool_calls=[tool_call("call_1")]),
            APITimeoutError(request=mock.Mock()),
            completion(content="پاسخ نهایی"),
        )

        with mock.patch.object(
            main.web_search, "execute_tool", return_value='{"results": []}'
        ):
            reply = main.get_reply_with_tools(
                client,
                "test-model",
                [{"role": "user", "content": "?"}],
                now=FakeClock(),
            )

        # The timed-out round is skipped, and the final answer still receives the
        # tool result of the round that did finish.
        self.assertEqual(reply, "پاسخ نهایی")
        self.assertEqual(client.chat.completions.create.call_count, 3)
        # The round the timeout freed up is spent on the answer, which still offers
        # the tools.
        self.assertEqual(offered_tools(client, -1), [web_search.TOOL_NAME])
        tool_turns = [m for m in sent_messages(client) if m["role"] == "tool"]
        self.assertEqual(len(tool_turns), 1)

    def test_an_unreachable_provider_still_fails_the_turn(self):
        # Only a timeout is survivable. A provider that was never reached is a real
        # failure and must reach the caller (server.py maps it to 503).
        client = fake_client(APIConnectionError(request=mock.Mock()))

        with self.assertRaises(APIConnectionError):
            main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

    def test_the_round_and_tool_caps_still_apply_while_time_is_left(self):
        clock = FakeClock()
        client = fake_client(
            completion(
                tool_calls=[distinct_search_call(1), distinct_search_call(2)]
            ),
            completion(
                tool_calls=[distinct_search_call(3), distinct_search_call(4)]
            ),
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


class SaveCollectorTests(unittest.TestCase):
    """The save collector is per-request state, never module state."""

    def test_there_is_no_collector_by_default(self):
        # The None default is part of the contract: no collector means no place
        # to put a proposal, so save_research is never offered either.
        self.assertIsNone(main._PENDING_SAVES.get())
        self.assertFalse(main.record_proposal(save_arguments()))
        self.assertFalse(main.save_tool_offered([REAL_URL]))

    def test_the_context_manager_yields_its_own_list(self):
        with main.collect_saves() as proposals:
            self.assertEqual(proposals, [])
            self.assertIs(main._PENDING_SAVES.get(), proposals)

    def test_a_proposal_is_recorded_while_the_collector_is_active(self):
        proposal = save_arguments()

        with main.collect_saves() as proposals:
            self.assertTrue(main.record_proposal(proposal))
            self.assertEqual(proposals, [proposal])

    def test_a_proposal_is_dropped_without_a_collector(self):
        self.assertFalse(main.record_proposal(save_arguments()))

    def test_the_previous_value_is_restored_after_the_context(self):
        with main.collect_saves() as proposals:
            main.record_proposal(save_arguments(title="mine"))

        self.assertEqual([item["title"] for item in proposals], ["mine"])
        self.assertIsNone(main._PENDING_SAVES.get())
        self.assertFalse(main.record_proposal(save_arguments(title="later")))

    def test_a_nested_context_restores_the_outer_collector(self):
        with main.collect_saves() as outer:
            with main.collect_saves() as inner:
                main.record_proposal(save_arguments(title="inner"))
            self.assertIs(main._PENDING_SAVES.get(), outer)
            main.record_proposal(save_arguments(title="outer"))

        self.assertEqual([item["title"] for item in outer], ["outer"])
        self.assertEqual([item["title"] for item in inner], ["inner"])

    def test_an_exception_still_resets_the_collector(self):
        proposals = None

        with self.assertRaises(RuntimeError):
            with main.collect_saves() as collected:
                proposals = collected
                main.record_proposal(save_arguments(title="kept"))
                raise RuntimeError("boom")

        self.assertEqual([item["title"] for item in proposals], ["kept"])
        self.assertIsNone(main._PENDING_SAVES.get())
        self.assertFalse(main.record_proposal(save_arguments(title="after")))

    def test_only_a_proposal_mapping_is_recorded(self):
        with main.collect_saves() as proposals:
            for junk in (None, 5, "proposal", ["proposal"]):
                with self.subTest(junk=junk):
                    self.assertFalse(main.record_proposal(junk))
            self.assertEqual(proposals, [])

    def test_concurrent_collectors_never_share_proposals(self):
        results = {}
        both_open = threading.Barrier(2, timeout=5)

        def work(name):
            try:
                with main.collect_saves() as proposals:
                    # Both collectors are open at the same time here, so a leak
                    # between requests would show up in the assertions below.
                    both_open.wait()
                    main.record_proposal(save_arguments(title=name))
                    results[name] = [item["title"] for item in proposals]
            except Exception as error:
                results[name] = error

        threads = [
            threading.Thread(target=work, args=(name,)) for name in ("first", "second")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(results["first"], ["first"])
        self.assertEqual(results["second"], ["second"])
        # The parent context never had a collector and must not gain one.
        self.assertIsNone(main._PENDING_SAVES.get())

    def test_a_sibling_context_cannot_see_this_collector(self):
        sibling = contextvars.copy_context()

        with main.collect_saves() as proposals:
            main.record_proposal(save_arguments(title="mine"))
            # The copy was taken before this collector existed, so code running
            # inside it has no collector: it can neither read nor extend ours.
            leaked = sibling.run(
                lambda: main.record_proposal(save_arguments(title="other"))
            )
            self.assertFalse(leaked)

        self.assertEqual([item["title"] for item in proposals], ["mine"])


class SaveToolAvailabilityTests(unittest.TestCase):
    """Only real, searched URLs, and only inside a collector, enable the tool."""

    def names(self, tools):
        return [tool["function"]["name"] for tool in tools]

    def test_real_urls_are_extracted_and_validated(self):
        content = json.dumps(
            {
                "query": "q",
                "results": [
                    {"url": "https://one.example/a"},
                    {"url": "https://one.example/a"},
                    {"url": "javascript:alert(1)"},
                    {"url": "ftp://two.example/b"},
                    {"url": "https://exa mple.com/a"},
                    {"title": "no url at all"},
                    "not a mapping",
                ],
            }
        )

        self.assertEqual(main.known_source_urls(content), ["https://one.example/a"])

    def test_a_result_without_usable_urls_yields_nothing(self):
        unusable = (
            "not json",
            '{"error": "could not reach the search service"}',
            '{"results": []}',
            '{"results": "not a list"}',
            "[]",
            None,
            5,
        )
        for content in unusable:
            with self.subTest(content=content):
                self.assertEqual(main.known_source_urls(content), [])

    def test_urls_accumulate_across_searches_without_duplicates(self):
        known = []

        main.remember_source_urls(known, search_result(["https://a.example/1"]))
        main.remember_source_urls(
            known, search_result(["https://a.example/1", "https://b.example/2"])
        )
        main.remember_source_urls(
            known, web_search.error_content("the search request timed out")
        )

        self.assertEqual(known, ["https://a.example/1", "https://b.example/2"])

    def test_the_save_tool_needs_both_a_collector_and_a_real_url(self):
        with main.collect_saves():
            self.assertFalse(main.save_tool_offered([]))
            self.assertTrue(main.save_tool_offered([REAL_URL]))

        # The same real URL, but no collector for this request.
        self.assertFalse(main.save_tool_offered([REAL_URL]))

    def test_build_tools_adds_the_save_tool_only_when_allowed(self):
        search_only = [web_search.TOOL_NAME]
        both = [web_search.TOOL_NAME, save_tool.TOOL_NAME]

        self.assertEqual(self.names(main.build_tools()), search_only)
        self.assertEqual(self.names(main.build_tools([])), search_only)
        self.assertEqual(self.names(main.build_tools([REAL_URL])), search_only)

        with main.collect_saves():
            self.assertEqual(self.names(main.build_tools([REAL_URL])), both)

        # Leaving the context must take the extra tool away again.
        self.assertEqual(self.names(main.build_tools([REAL_URL])), search_only)


class SystemPromptTests(unittest.TestCase):
    """The prompt must stop the model from claiming a file was saved."""

    def test_ready_is_explained_as_a_draft_not_a_written_file(self):
        prompt = main.SYSTEM_PROMPT.lower()

        self.assertIn(main.SAVE_READY_STATUS, prompt)
        self.assertIn("not that a file was saved", prompt)

    def test_an_explicit_confirmation_is_required(self):
        prompt = main.SYSTEM_PROMPT.lower()

        self.assertIn("confirm", prompt)
        self.assertIn("unambiguous", prompt)

    def test_the_prompt_is_the_one_sent_as_the_system_message(self):
        messages = main.build_prompt([], "?")

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], main.SYSTEM_PROMPT)


class SaveToolRegistryTests(SearchEnabledTestCase):
    """The tool list is rebuilt per round and follows what really happened."""

    def run_turn(self, client, search=None):
        """Run one tool turn with the search HTTP layer replaced."""
        with mock.patch.object(
            main.web_search, "execute_tool", return_value=search or search_result()
        ) as execute:
            reply = main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )
        return reply, execute

    def test_without_a_collector_only_web_search_is_ever_offered(self):
        client = fake_client(
            completion(tool_calls=[search_call()]), completion(content="پاسخ")
        )

        reply, execute = self.run_turn(client)

        self.assertEqual(reply, "پاسخ")
        execute.assert_called_once()
        self.assertEqual(offered_tools(client, 0), [web_search.TOOL_NAME])
        self.assertEqual(offered_tools(client, 1), [web_search.TOOL_NAME])

    def test_a_successful_search_offers_the_save_tool_in_the_next_round(self):
        client = fake_client(
            completion(tool_calls=[search_call()]), completion(content="پاسخ")
        )

        with main.collect_saves():
            self.run_turn(client)

        self.assertEqual(offered_tools(client, 0), [web_search.TOOL_NAME])
        self.assertEqual(
            offered_tools(client, 1), [web_search.TOOL_NAME, save_tool.TOOL_NAME]
        )

    def test_a_search_without_a_real_url_never_offers_the_save_tool(self):
        unusable = (
            web_search.error_content("could not reach the search service"),
            search_result([]),  # it worked, but found nothing
            search_result(["javascript:alert(1)"]),  # it worked, no usable URL
            "not json at all",
        )
        for search in unusable:
            with self.subTest(search=search[:40]):
                client = fake_client(
                    completion(tool_calls=[search_call()]), completion(content="پاسخ")
                )

                with main.collect_saves():
                    self.run_turn(client, search=search)

                self.assertEqual(offered_tools(client, 1), [web_search.TOOL_NAME])

    def test_a_save_call_before_any_real_source_is_refused(self):
        client = fake_client(
            completion(tool_calls=[save_call()]), completion(content="پاسخ")
        )

        with main.collect_saves() as proposals:
            reply, _ = self.run_turn(client)

        self.assertEqual(reply, "پاسخ")
        # It was never offered in that round, and calling it anyway changes nothing.
        self.assertEqual(offered_tools(client, 0), [web_search.TOOL_NAME])
        self.assertEqual(proposals, [])
        error = json.loads(tool_turns(client)[0]["content"])
        self.assertEqual(error["status"], "error")
        self.assertIn("no source", error["error"])


class CliToolWiringTests(SearchEnabledTestCase):
    """python main.py keeps the old behaviour: no collector, no save tool."""

    def test_the_terminal_loop_never_opens_a_collector(self):
        client = mock.Mock()

        with (
            mock.patch.object(main, "collect_saves") as collect,
            mock.patch.object(main, "load_config", return_value=("k", "u", "m")),
            mock.patch.object(main, "build_client", return_value=client),
            mock.patch.object(main, "chat", return_value=("ok", [])) as chat,
            mock.patch("builtins.input", side_effect=["hi", "exit"]),
            mock.patch("builtins.print"),
        ):
            main.main()

        self.assertEqual(chat.call_count, 1)
        collect.assert_not_called()

    def test_a_turn_without_a_collector_only_ever_gets_web_search(self):
        # This is the path the terminal loop and every plain caller use.
        client = fake_client(
            completion(tool_calls=[search_call()]), completion(content="پاسخ")
        )

        with mock.patch.object(
            main.web_search, "execute_tool", return_value=search_result()
        ):
            reply, history = main.chat("هوا؟", [], client, "test-model")

        self.assertEqual(reply, "پاسخ")
        self.assertEqual(offered_tools(client, 0), [web_search.TOOL_NAME])
        self.assertEqual(offered_tools(client, 1), [web_search.TOOL_NAME])
        self.assertEqual([turn["role"] for turn in history], ["user", "assistant"])


class SaveFlowTestCase(SearchEnabledTestCase):
    """Base case for the save flow: reaching the only writer fails the test."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(
            main.save_tool,
            "write_proposal",
            side_effect=AssertionError("write_proposal() must not be called"),
        )
        self.writer = patcher.start()
        self.addCleanup(patcher.stop)

    def run_turn(self, client, search=None):
        """Run one tool turn with the search HTTP layer replaced."""
        with mock.patch.object(
            main.web_search, "execute_tool", return_value=search or search_result()
        ):
            return main.get_reply_with_tools(
                client, "test-model", [{"role": "user", "content": "?"}]
            )

    def save_answer(self, client):
        """The JSON object the save tool answered with."""
        return json.loads(tool_turns(client)[-1]["content"])

    def two_round_client(self, save):
        """A client that searches, then asks to save, then answers.

        ``save`` is one tool call, or a list of them for a parallel round.
        """
        saved = save if isinstance(save, list) else [save]
        return fake_client(
            completion(tool_calls=[search_call()]),
            completion(tool_calls=saved),
            completion(content="پاسخ نهایی"),
        )


class SaveFlowTests(SaveFlowTestCase):
    """A finished search plus a save call: one proposal, and no file."""

    def test_a_valid_save_call_records_one_proposal(self):
        client = self.two_round_client(save_call())

        with main.collect_saves() as proposals:
            reply = self.run_turn(client)

        self.assertEqual(reply, "پاسخ نهایی")
        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(proposal["title"], "Tehran weather")
        self.assertEqual([source["url"] for source in proposal["sources"]], [REAL_URL])
        self.assertIn("# Tehran weather", proposal["content"])
        self.assertTrue(proposal["filename"].endswith(save_tool.FILE_EXTENSION))
        self.assertEqual(len(proposal["content"].encode("utf-8")), proposal["bytes"])

    def test_the_model_gets_a_small_path_free_acknowledgement(self):
        client = self.two_round_client(save_call())

        with main.collect_saves() as proposals:
            self.run_turn(client)

        ack = self.save_answer(client)
        self.assertEqual(ack["status"], main.SAVE_READY_STATUS)
        self.assertEqual(sorted(ack), ["bytes", "file", "sources", "status", "title"])
        self.assertEqual(ack["file"], proposals[0]["filename"])
        self.assertEqual(ack["title"], "Tehran weather")
        self.assertEqual(ack["sources"], 1)
        self.assertEqual(ack["bytes"], proposals[0]["bytes"])
        self.assertNotIn(os.sep, ack["file"])

    def test_the_flow_fits_the_existing_budgets(self):
        client = self.two_round_client(save_call())

        with main.collect_saves():
            self.run_turn(client)

        # Round 1 searches, round 2 saves, then the reserved answer round.
        self.assertEqual(main.MAX_TOOL_ROUNDS, 2)
        self.assertEqual(main.MAX_TOOL_CALLS, 3)
        self.assertEqual(client.chat.completions.create.call_count, 3)
        self.assertEqual(offered_tools(client, 0), [web_search.TOOL_NAME])
        self.assertEqual(
            offered_tools(client, 1), [web_search.TOOL_NAME, save_tool.TOOL_NAME]
        )
        # The answer round keeps the tools as well: a model that still wants to save
        # can say so on the structured channel instead of writing markup into text.
        self.assertEqual(
            offered_tools(client, 2), [web_search.TOOL_NAME, save_tool.TOOL_NAME]
        )

    def test_a_source_the_search_never_returned_is_dropped(self):
        invented = "https://invented.example/fake"
        client = self.two_round_client(
            save_call(
                sources=[
                    {"title": "Real", "url": REAL_URL},
                    {"title": "Invented", "url": invented},
                ]
            )
        )

        with main.collect_saves() as proposals:
            self.run_turn(client)

        sources = [source["url"] for source in proposals[0]["sources"]]
        self.assertEqual(sources, [REAL_URL])
        self.assertNotIn(invented, json.dumps(proposals[0], ensure_ascii=False))

    def test_a_search_without_usable_urls_records_nothing(self):
        client = self.two_round_client(save_call())

        with main.collect_saves() as proposals:
            reply = self.run_turn(client, search=search_result([]))

        self.assertEqual(reply, "پاسخ نهایی")
        self.assertEqual(proposals, [])
        error = self.save_answer(client)
        self.assertEqual(error["status"], "error")
        self.assertIn("no source", error["error"])

    def test_a_direct_call_without_a_collector_is_refused(self):
        answer = json.loads(
            main.execute_save_tool(json.dumps(save_arguments()), [REAL_URL])
        )

        self.assertEqual(answer["status"], "error")
        self.assertIn("not available", answer["error"])
        self.writer.assert_not_called()

    def test_a_direct_call_with_an_empty_url_list_is_refused(self):
        with main.collect_saves() as proposals:
            raw = json.dumps(save_arguments())
            answer = json.loads(main.execute_save_tool(raw, []))

        self.assertEqual(answer["status"], "error")
        self.assertIn("no source", answer["error"])
        self.assertEqual(proposals, [])
        self.writer.assert_not_called()

    def test_invalid_arguments_come_back_as_an_error_object(self):
        client = self.two_round_client(save_call(summary="   "))

        with main.collect_saves() as proposals:
            reply = self.run_turn(client)

        # The model still gets an answer and can correct its arguments.
        self.assertEqual(reply, "پاسخ نهایی")
        self.assertEqual(proposals, [])
        error = self.save_answer(client)
        self.assertEqual(error["status"], "error")
        self.assertIn("'summary'", error["error"])
        self.writer.assert_not_called()

    def test_unreadable_arguments_come_back_as_an_error_object(self):
        broken = tool_call("call_save", name=save_tool.TOOL_NAME, arguments="{not json")
        client = self.two_round_client(broken)

        with main.collect_saves() as proposals:
            reply = self.run_turn(client)

        self.assertEqual(reply, "پاسخ نهایی")
        self.assertEqual(proposals, [])
        self.assertIn("error", self.save_answer(client))
        self.writer.assert_not_called()

    def test_a_broken_save_tool_does_not_break_the_conversation(self):
        client = self.two_round_client(save_call())

        with mock.patch.object(
            main.save_tool, "build_proposal", side_effect=RuntimeError("boom")
        ):
            with main.collect_saves() as proposals:
                reply = self.run_turn(client)

        self.assertEqual(reply, "پاسخ نهایی")
        self.assertEqual(proposals, [])
        self.assertEqual(self.save_answer(client)["error"], "the save tool failed")
        self.writer.assert_not_called()

    def test_repeated_save_calls_still_write_nothing(self):
        two_saves = [save_call("call_save_1"), save_call("call_save_2")]
        client = self.two_round_client(two_saves)

        with main.collect_saves() as proposals:
            self.run_turn(client)

        answers = [json.loads(turn["content"]) for turn in tool_turns(client)[1:]]
        self.assertEqual([answer["status"] for answer in answers], ["ready"] * 2)
        self.assertEqual(len(proposals), 2)
        self.assertNotEqual(proposals[0]["filename"], proposals[1]["filename"])
        self.writer.assert_not_called()

    def test_a_full_search_budget_does_not_use_up_the_save_budget(self):
        searches = [
            distinct_search_call(index) for index in range(main.MAX_TOOL_CALLS)
        ]
        client = fake_client(
            completion(tool_calls=searches),
            completion(tool_calls=[save_call()]),
            completion(content="پاسخ بدون ذخیره"),
        )

        with main.collect_saves() as proposals:
            reply = self.run_turn(client)

        # The three searches used the whole search budget, but the save has its own,
        # so the call in the next round is executed and one proposal is recorded.
        self.assertEqual(reply, "پاسخ بدون ذخیره")
        self.assertEqual(len(proposals), 1)
        answer = self.save_answer(client)
        self.assertEqual(answer["status"], main.SAVE_READY_STATUS)
        self.assertEqual(client.chat.completions.create.call_count, 3)
        self.writer.assert_not_called()


    def test_the_save_budget_caps_the_proposals_on_its_own(self):
        saves = [
            save_call(f"call_save_{index}") for index in range(main.MAX_SAVE_CALLS + 1)
        ]
        client = fake_client(
            completion(tool_calls=[search_call()]),
            completion(tool_calls=saves),
            completion(content="done"),
        )

        with main.collect_saves() as proposals:
            reply = self.run_turn(client)

        self.assertEqual(reply, "done")
        self.assertEqual(len(proposals), main.MAX_SAVE_CALLS)
        # Every requested call is still answered, and the one past the save budget
        # is told so, without touching the file.
        self.assertEqual(self.save_answer(client)["error"], main.SAVE_BUDGET_SPENT)
        self.writer.assert_not_called()


class SaveFlowEndToEndTests(SaveFlowTestCase):
    """The whole flow through chat(): the real search layer, mocked transport."""

    def test_search_then_save_through_chat(self):
        client = self.two_round_client(save_call())
        body = json.dumps(
            {"results": [{"title": "Tehran", "url": REAL_URL, "content": "sunny"}]}
        ).encode("utf-8")

        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = body
            with main.collect_saves() as proposals:
                reply, history = main.chat("هوا؟", [], client, "test-model")

        self.assertEqual(reply, "پاسخ نهایی")
        self.assertEqual(len(proposals), 1)
        sources = [source["url"] for source in proposals[0]["sources"]]
        self.assertEqual(sources, [REAL_URL])
        self.assertEqual(
            urlopen.call_args.args[0].full_url, web_search.TAVILY_SEARCH_URL
        )
        # Neither the tool traffic nor the proposal may enter the public history.
        self.assertEqual([turn["role"] for turn in history], ["user", "assistant"])
        for turn in history:
            self.assertEqual(sorted(turn), ["content", "role"])
        self.assertNotIn(REAL_URL, json.dumps(history))
        self.assertNotIn(proposals[0]["filename"], json.dumps(history))
        self.writer.assert_not_called()


class NoWritePathTests(SaveFlowTestCase):
    """There is no execution path from this module to the file writer."""

    def test_the_writer_is_never_referenced_in_the_code(self):
        # Parsed rather than grepped: comments may name the writer, but no
        # attribute access such as save_tool.write_proposal(...) may exist.
        with open(main.__file__, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())

        attributes = [
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        ]
        self.assertNotIn("write_proposal", attributes)

    def test_the_public_entry_points_are_unchanged(self):
        self.assertEqual(
            list(inspect.signature(main.chat).parameters),
            ["message", "history", "client", "model_name"],
        )
        self.assertEqual(
            list(inspect.signature(main.get_reply).parameters),
            ["client", "model_name", "messages"],
        )

    def test_the_new_helpers_are_side_effect_free(self):
        # None of them may create the output folder either: one proposal is
        # recorded by hand, the second one by the tool itself.
        with mock.patch("os.makedirs", side_effect=AssertionError("makedirs()")):
            with mock.patch("os.remove", side_effect=AssertionError("remove()")):
                with main.collect_saves() as proposals:
                    main.record_proposal(save_arguments())
                    answer = main.execute_save_tool(
                        json.dumps(save_arguments()), [REAL_URL]
                    )
                self.assertEqual(len(proposals), 2)
                self.assertEqual(json.loads(answer)["status"], "ready")
        self.writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()




