"""Offline tests for web_search.py, the Tavily-backed search tool.

The HTTP layer (``post_json``/``urllib.request.urlopen``) is always replaced,
so no socket is ever opened and the tests pass with the network unplugged. The
key used here is an obvious fake placeholder; no value is read from .env.
"""

import json
import os
import socket
import unittest
import urllib.error
from unittest import mock

import web_search

# Clearly fake: never a real credential and never printed by the code.
FAKE_KEY = "test-key-not-a-real-secret"

# Patchers that make any socket or HTTP use fail loudly; see setUpModule().
_GUARDS = []


def setUpModule():
    """Keep the whole module offline, not just the tests that remember to mock.

    The search HTTP layer is replaced everywhere below, so a real request would
    mean a bug in a test. Patching the low-level entry points turns such a bug
    into a clear failure instead of a live call.
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


def oversized_response():
    """A Tavily-shaped body with more results and longer text than allowed."""
    return {
        "query": "anything",
        "results": [
            {
                "title": f"Result {index} " + "T" * 400,
                "url": f"https://example.com/{index}",
                "content": "C" * 2000,
                "score": 0.9,
            }
            for index in range(20)
        ],
    }


def http_error(case, code, message="boom"):
    """Build an HTTPError whose body is closed once the test finishes."""
    error = urllib.error.HTTPError(
        web_search.TAVILY_SEARCH_URL, code, message, {}, None
    )
    case.addCleanup(error.close)
    return error


class ToolSpecTests(unittest.TestCase):
    """The model must only ever be offered the one, narrow tool."""

    def test_only_web_search_is_offered(self):
        tools = web_search.build_tools()

        self.assertEqual(web_search.TOOL_NAME, "web_search")
        self.assertEqual(len(tools), 1)
        spec = tools[0]
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], web_search.TOOL_NAME)

        parameters = spec["function"]["parameters"]
        self.assertEqual(parameters["type"], "object")
        self.assertEqual(sorted(parameters["properties"]), ["query"])
        self.assertEqual(parameters["properties"]["query"]["type"], "string")
        self.assertEqual(parameters["required"], ["query"])
        self.assertFalse(parameters["additionalProperties"])

    def test_tool_budgets_are_small(self):
        self.assertEqual(web_search.TAVILY_SEARCH_URL, "https://api.tavily.com/search")
        self.assertEqual(web_search.SEARCH_API_KEY_ENV, "SEARCH_API_KEY")
        self.assertEqual(web_search.MAX_RESULTS, 5)
        self.assertGreater(web_search.REQUEST_TIMEOUT_SECONDS, 0)
        self.assertLessEqual(web_search.REQUEST_TIMEOUT_SECONDS, 20)


class AvailabilityTests(unittest.TestCase):
    """No key means no search: the tool stays off."""

    def test_missing_key_disables_the_tool(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(web_search.SEARCH_API_KEY_ENV, None)
            self.assertIsNone(web_search.get_api_key())
            self.assertFalse(web_search.is_available())

    def test_blank_key_counts_as_missing(self):
        with mock.patch.dict(os.environ, {web_search.SEARCH_API_KEY_ENV: "   "}):
            self.assertFalse(web_search.is_available())

    def test_configured_key_enables_the_tool(self):
        with mock.patch.dict(os.environ, {web_search.SEARCH_API_KEY_ENV: FAKE_KEY}):
            self.assertTrue(web_search.is_available())

    def test_search_refuses_to_run_without_a_key(self):
        with mock.patch.dict(os.environ, {web_search.SEARCH_API_KEY_ENV: ""}):
            with mock.patch.object(web_search, "post_json") as post:
                with self.assertRaises(web_search.SearchError) as caught:
                    web_search.search("anything")

        post.assert_not_called()
        self.assertNotIn(FAKE_KEY, str(caught.exception))


class QueryValidationTests(unittest.TestCase):
    """Bad input is rejected before anything touches the network."""

    def test_arguments_must_be_a_json_object(self):
        for raw in ("not json", "", "   ", "[1, 2]", "42", '"text"', None, 5, 1.5):
            with self.subTest(raw=raw):
                with self.assertRaises(web_search.SearchError):
                    web_search.parse_arguments(raw)

    def test_query_is_required_and_must_be_text(self):
        for raw in ("{}", '{"query": "   "}', '{"query": 12}', '{"other": "x"}'):
            with self.subTest(raw=raw):
                with self.assertRaises(web_search.SearchError):
                    web_search.parse_arguments(raw)

    def test_query_whitespace_is_collapsed(self):
        self.assertEqual(
            web_search.parse_arguments('{"query": "  tehran\\n  weather  "}'),
            {"query": "tehran weather"},
        )

    def test_overlong_query_is_rejected(self):
        raw = json.dumps({"query": "x" * (web_search.MAX_QUERY_LENGTH + 1)})
        with self.assertRaises(web_search.SearchError) as caught:
            web_search.parse_arguments(raw)
        self.assertIn("characters", str(caught.exception))

    def test_a_mapping_is_accepted(self):
        self.assertEqual(
            web_search.parse_arguments({"query": "hello"}), {"query": "hello"}
        )


class SearchTests(unittest.TestCase):
    """The request we build and the bounded answer we hand back."""

    def search_with(self, body):
        with mock.patch.dict(os.environ, {web_search.SEARCH_API_KEY_ENV: FAKE_KEY}):
            with mock.patch.object(web_search, "post_json", return_value=body) as post:
                result = web_search.search("  tehran   weather  ")
        return result, post

    def test_request_goes_to_tavily_with_a_bounded_body(self):
        result, post = self.search_with({"results": []})

        url, payload, key = post.call_args.args
        self.assertEqual(url, web_search.TAVILY_SEARCH_URL)
        self.assertEqual(key, FAKE_KEY)
        self.assertEqual(payload["query"], "tehran weather")
        self.assertEqual(payload["max_results"], web_search.MAX_RESULTS)
        # The key never travels in the body we build, only in the header.
        self.assertNotIn(FAKE_KEY, json.dumps(payload))
        self.assertEqual(result, {"query": "tehran weather", "results": []})

    def test_results_are_limited_in_number_and_size(self):
        result, _ = self.search_with(oversized_response())

        self.assertEqual(len(result["results"]), web_search.MAX_RESULTS)
        for entry in result["results"]:
            self.assertEqual(sorted(entry), ["snippet", "title", "url"])
            self.assertLessEqual(len(entry["title"]), web_search.MAX_TITLE_LENGTH)
            self.assertLessEqual(len(entry["url"]), web_search.MAX_URL_LENGTH)
            self.assertLessEqual(len(entry["snippet"]), web_search.MAX_SNIPPET_LENGTH)

    def test_only_title_url_and_snippet_are_returned(self):
        normalized = web_search.normalize_results(oversized_response())
        content = web_search.to_tool_content({"query": "q", "results": normalized})

        self.assertLessEqual(len(content), web_search.MAX_OUTPUT_LENGTH)
        decoded = json.loads(content)
        self.assertEqual(decoded["query"], "q")
        self.assertNotIn("score", content)
        self.assertNotIn(FAKE_KEY, content)

    def test_non_http_urls_are_dropped(self):
        body = {
            "results": [
                {"title": "bad", "url": "javascript:alert(1)", "content": "x"},
                {"title": "worse", "url": None, "content": "y"},
                {"title": "good", "url": "https://example.com", "content": "y"},
            ]
        }
        result, _ = self.search_with(body)

        self.assertEqual(
            [entry["url"] for entry in result["results"]], ["https://example.com"]
        )

    def test_unexpected_response_shapes_are_rejected(self):
        for body in ([], "text", 5, {"results": "nope"}):
            with self.subTest(body=body):
                with mock.patch.dict(
                    os.environ, {web_search.SEARCH_API_KEY_ENV: FAKE_KEY}
                ):
                    with mock.patch.object(web_search, "post_json", return_value=body):
                        with self.assertRaises(web_search.SearchError):
                            web_search.search("anything")

    def test_long_error_payloads_stay_bounded(self):
        content = web_search.error_content("e" * 5000)

        self.assertLessEqual(len(content), web_search.MAX_OUTPUT_LENGTH)
        self.assertIn("error", json.loads(content))


class HttpLayerTests(unittest.TestCase):
    """HTTP failures map to short messages that never leak the key."""

    def http_error(self, code, message):
        """Build an HTTPError and close its body once the test is done."""
        return http_error(self, code, message)

    def post(self):
        return web_search.post_json(
            web_search.TAVILY_SEARCH_URL, {"query": "q"}, FAKE_KEY
        )

    def test_bearer_header_and_timeout_are_used(self):
        with mock.patch("urllib.request.urlopen") as urlopen:
            response = urlopen.return_value.__enter__.return_value
            response.read.return_value = b'{"results": []}'
            result = self.post()

        self.assertEqual(result, {"results": []})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Authorization"), f"Bearer {FAKE_KEY}")
        self.assertEqual(
            urlopen.call_args.kwargs["timeout"], web_search.REQUEST_TIMEOUT_SECONDS
        )
        self.assertNotIn(FAKE_KEY.encode(), request.data)

    def test_http_status_error(self):
        error = self.http_error(401, "unauthorized")
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(web_search.SearchError) as caught:
                self.post()

        self.assertIn("HTTP 401", str(caught.exception))
        self.assertNotIn(FAKE_KEY, str(caught.exception))

    def test_timeout(self):
        timeout = TimeoutError("slow")
        for error in (timeout, urllib.error.URLError(timeout)):
            with self.subTest(error=type(error).__name__):
                with mock.patch("urllib.request.urlopen", side_effect=error):
                    with self.assertRaises(web_search.SearchError) as caught:
                        self.post()
                self.assertIn("timed out", str(caught.exception))

    def test_network_failure(self):
        for error in (
            urllib.error.URLError(socket.gaierror(-2, "name resolution failed")),
            socket.gaierror(-2, "name resolution failed"),
        ):
            with self.subTest(error=type(error).__name__):
                with mock.patch("urllib.request.urlopen", side_effect=error):
                    with self.assertRaises(web_search.SearchError) as caught:
                        self.post()
                self.assertIn("could not reach", str(caught.exception))
                self.assertNotIn(FAKE_KEY, str(caught.exception))

    def test_unreadable_body(self):
        with mock.patch("urllib.request.urlopen") as urlopen:
            response = urlopen.return_value.__enter__.return_value
            response.read.return_value = b"<html>not json</html>"
            with self.assertRaises(web_search.SearchError) as caught:
                self.post()

        self.assertIn("unreadable", str(caught.exception))


class RequestTimeoutTests(unittest.TestCase):
    """One search never gets more time than the caller's budget allows."""

    def test_the_cap_is_the_module_default(self):
        self.assertEqual(
            web_search.bounded_timeout(None), web_search.REQUEST_TIMEOUT_SECONDS
        )
        self.assertEqual(
            web_search.bounded_timeout(30), web_search.REQUEST_TIMEOUT_SECONDS
        )

    def test_a_smaller_budget_is_used_as_it_is(self):
        self.assertEqual(web_search.bounded_timeout(2.5), 2.5)

    def test_a_useless_budget_is_refused_before_the_network(self):
        with mock.patch.object(web_search, "post_json") as post:
            with mock.patch.dict(
                os.environ, {web_search.SEARCH_API_KEY_ENV: FAKE_KEY}
            ):
                with self.assertRaises(web_search.SearchError) as caught:
                    web_search.search("q", timeout=web_search.MIN_REQUEST_SECONDS / 2)

        post.assert_not_called()
        self.assertIn("not enough time", str(caught.exception))

    def test_the_remaining_time_reaches_the_http_layer(self):
        with mock.patch.object(
            web_search, "post_json", return_value={"results": []}
        ) as post:
            with mock.patch.dict(
                os.environ, {web_search.SEARCH_API_KEY_ENV: FAKE_KEY}
            ):
                web_search.search("q", timeout=3.5)

        self.assertEqual(post.call_args.kwargs["timeout"], 3.5)

    def test_execute_tool_passes_the_remaining_time_through(self):
        found = {"query": "q", "results": []}

        with mock.patch.object(web_search, "search", return_value=found) as search:
            content = web_search.execute_tool(
                web_search.TOOL_NAME, '{"query": "q"}', timeout=2.0
            )

        self.assertEqual(search.call_args.kwargs["timeout"], 2.0)
        self.assertEqual(json.loads(content), found)

    def test_a_call_with_no_time_left_becomes_an_error_object(self):
        with mock.patch.object(web_search, "post_json") as post:
            with mock.patch.dict(
                os.environ, {web_search.SEARCH_API_KEY_ENV: FAKE_KEY}
            ):
                content = web_search.execute_tool(
                    web_search.TOOL_NAME, '{"query": "q"}', timeout=0.0
                )

        post.assert_not_called()
        self.assertIn("not enough time", json.loads(content)["error"])


class ExecuteToolTests(unittest.TestCase):
    """execute_tool() always returns a JSON string, never raises."""

    def test_unknown_tool_name_is_never_executed(self):
        with mock.patch.object(web_search, "search") as search:
            content = web_search.execute_tool("run_shell", '{"query": "x"}')

        search.assert_not_called()
        self.assertIn("unknown tool", json.loads(content)["error"])

    def test_invalid_arguments_never_reach_the_network(self):
        with mock.patch.object(web_search, "post_json") as post:
            content = web_search.execute_tool(web_search.TOOL_NAME, "not json")

        post.assert_not_called()
        self.assertIn("error", json.loads(content))

    def test_empty_arguments_are_refused(self):
        with mock.patch.object(web_search, "post_json") as post:
            content = web_search.execute_tool(web_search.TOOL_NAME, "")

        post.assert_not_called()
        self.assertIn("no arguments", json.loads(content)["error"])

    def test_successful_call_returns_only_title_url_and_snippet(self):
        with mock.patch.dict(os.environ, {web_search.SEARCH_API_KEY_ENV: FAKE_KEY}):
            with mock.patch.object(
                web_search, "post_json", return_value=oversized_response()
            ):
                content = web_search.execute_tool(
                    web_search.TOOL_NAME, '{"query": "tehran"}'
                )

        payload = json.loads(content)
        self.assertEqual(payload["query"], "tehran")
        self.assertEqual(len(payload["results"]), web_search.MAX_RESULTS)
        self.assertEqual(
            sorted(payload["results"][0]), ["snippet", "title", "url"]
        )
        self.assertNotIn(FAKE_KEY, content)

    def test_search_failures_become_a_small_error_object(self):
        failures = {
            "timeout": TimeoutError("slow"),
            "http": http_error(self, 500),
            "dns": urllib.error.URLError(socket.gaierror(-2, "no dns")),
        }
        for label, error in failures.items():
            with self.subTest(case=label):
                with mock.patch.dict(
                    os.environ, {web_search.SEARCH_API_KEY_ENV: FAKE_KEY}
                ):
                    with mock.patch("urllib.request.urlopen", side_effect=error):
                        content = web_search.execute_tool(
                            web_search.TOOL_NAME, '{"query": "x"}'
                        )

                self.assertIn("error", json.loads(content))
                self.assertNotIn(FAKE_KEY, content)
                self.assertLessEqual(len(content), web_search.MAX_OUTPUT_LENGTH)


if __name__ == "__main__":
    unittest.main()

