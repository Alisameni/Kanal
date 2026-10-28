"""Contract tests: the terminal entry point, chat() and POST /chat must not drift.

``fastapi.testclient`` runs the app in-process (no socket is opened), and the
model call itself is mocked, so these tests are fully offline.
"""

import inspect
import unittest
from unittest import mock

from fastapi.testclient import TestClient
from openai import APIStatusError

import main
import server
import web_search

client = TestClient(server.app)


class SignatureTests(unittest.TestCase):
    def test_chat_keeps_its_signature(self):
        parameters = inspect.signature(main.chat).parameters

        self.assertEqual(
            list(parameters), ["message", "history", "client", "model_name"]
        )
        for name in ("history", "client", "model_name"):
            self.assertIsNone(parameters[name].default)

    def test_get_reply_keeps_its_signature(self):
        self.assertEqual(
            list(inspect.signature(main.get_reply).parameters),
            ["client", "model_name", "messages"],
        )

    def test_the_tool_entry_points_do_not_replace_the_old_ones(self):
        # The tool loop is an addition, not a rewrite of the plain path.
        self.assertTrue(callable(main.get_reply))
        self.assertTrue(callable(main.get_reply_with_tools))
        self.assertNotEqual(main.get_reply, main.get_reply_with_tools)

    def test_limits_are_unchanged(self):
        self.assertEqual(main.HISTORY_LIMIT, 10)
        self.assertEqual(server.MAX_HISTORY_MESSAGES, main.HISTORY_LIMIT)
        self.assertEqual(server.MAX_MESSAGE_LENGTH, 4000)
        self.assertEqual(server.MAX_HISTORY_CONTENT_LENGTH, 4000)
        self.assertEqual(server.ALLOWED_ROLES, ("user", "assistant"))

    def test_the_optional_search_key_keeps_its_name(self):
        # Only the variable *name* is asserted; no value is ever read or printed.
        self.assertEqual(web_search.SEARCH_API_KEY_ENV, "SEARCH_API_KEY")
        self.assertEqual(web_search.TOOL_NAME, "web_search")


class ChatEndpointTests(unittest.TestCase):
    def test_the_contract_is_reply_and_history(self):
        history = [{"role": "assistant", "content": "پاسخ"}]

        with mock.patch.object(
            server.main, "chat", return_value=("سلام", history)
        ) as chat:
            response = client.post("/chat", json={"message": "hi", "history": []})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"reply": "سلام", "history": history})
        chat.assert_called_once_with("hi", [])

    def test_history_is_forwarded_unchanged(self):
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]

        with mock.patch.object(
            server.main, "chat", return_value=("ok", history)
        ) as chat:
            client.post("/chat", json={"message": "hi", "history": history})

        chat.assert_called_once_with("hi", history)

    def test_a_blank_message_is_rejected(self):
        response = client.post("/chat", json={"message": "   "})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"detail": "Invalid request payload."})

    def test_a_system_role_is_still_rejected(self):
        response = client.post(
            "/chat",
            json={"message": "hi", "history": [{"role": "system", "content": "x"}]},
        )

        self.assertEqual(response.status_code, 422)

    def test_tool_roles_are_not_accepted_from_clients(self):
        response = client.post(
            "/chat",
            json={"message": "hi", "history": [{"role": "tool", "content": "{}"}]},
        )

        self.assertEqual(response.status_code, 422)

    def test_the_history_limit_is_still_enforced(self):
        history = [{"role": "user", "content": "x"}] * (
            server.MAX_HISTORY_MESSAGES + 1
        )

        response = client.post("/chat", json={"message": "hi", "history": history})

        self.assertEqual(response.status_code, 422)

    def test_config_errors_still_map_to_503(self):
        with mock.patch.object(
            server.main, "chat", side_effect=main.ConfigError("missing keys")
        ):
            response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.status_code, 503)

    def test_an_empty_reply_still_maps_to_502(self):
        with mock.patch.object(server.main, "chat", return_value=(None, [])):
            response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.status_code, 502)

    def test_a_tool_deadline_maps_to_504(self):
        with mock.patch.object(
            server.main,
            "chat",
            side_effect=main.ToolDeadlineExceeded("internal detail"),
        ):
            response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.status_code, 504)
        self.assertEqual(
            response.json()["detail"],
            "The web search turn took too long to finish.",
        )
        # The internal message is never returned to the client.
        self.assertNotIn("internal detail", response.text)

    def test_a_rejected_tools_parameter_still_maps_to_502(self):
        error = APIStatusError(
            "tools are not supported",
            response=mock.Mock(status_code=400, headers={}),
            body=None,
        )

        with mock.patch.object(server.main, "chat", side_effect=error):
            response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"], "The model provider returned an error."
        )
        self.assertNotIn("tools are not supported", response.text)

    def test_status_endpoints_still_work(self):
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/health").json(), {"status": "healthy"})


if __name__ == "__main__":
    unittest.main()
