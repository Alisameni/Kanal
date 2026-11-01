"""Contract tests: the terminal entry point, chat() and POST /chat must not drift.

``fastapi.testclient`` runs the app in-process (no socket is opened), and the
model call itself is mocked, so these tests are fully offline.

The save workflow is covered here as well: the pending store on its own (with an
injected clock, so an expiring proposal never costs a real second) and the two
endpoints that consume it, always with the writer of save_tool redirected into a
``tempfile`` sandbox. No test touches the project's real ``search_results``.
"""

import inspect
import os
import shutil
import tempfile
import threading
import unittest
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError, APITimeoutError

import main
import save_tool
import server
import web_search

client = TestClient(server.app)


class FakeClock:
    """A monotonic clock the tests move by hand."""

    def __init__(self, now=1_000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def save_arguments(**overrides):
    """A minimal, valid save_research argument mapping."""
    base = {
        "title": "گزارش تحقیقاتی",
        "summary": "خلاصه یافتهها",
        "analysis": "متن تحلیل کامل",
        "sources": [{"title": "Example", "url": "https://example.com/a"}],
    }
    base.update(overrides)
    return base


def make_proposal(stamp="20260101-000000", token="abc123", **overrides):
    """A proposal with a fixed name, exactly as main passes it to the server."""
    return save_tool.build_proposal(
        save_arguments(**overrides), stamp=stamp, token=token
    )


def error_detail(response):
    """The generic message of an error response, if it has one."""
    return response.json().get("detail")


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

    def test_the_pending_store_is_bounded_and_short_lived(self):
        self.assertEqual(server.PENDING_SAVE_TTL_SECONDS, 300.0)
        self.assertGreaterEqual(server.MAX_PENDING_SAVES, 1)
        self.assertLessEqual(server.MAX_PENDING_SAVES, 64)
        self.assertIsInstance(server.PENDING_SAVES, server.PendingSaveStore)

    def test_the_pending_store_is_never_persisted(self):
        # A fresh process starts empty: nothing is written to or read from disk,
        # so a restart simply forgets every proposal that was still waiting.
        store = server.PendingSaveStore(clock=FakeClock())
        self.assertEqual(store.count(), 0)

    def test_the_save_endpoints_accept_post_only(self):
        routes = {
            route.path: route.methods
            for route in server.app.routes
            if getattr(route, "path", "").startswith("/save")
        }

        self.assertEqual(sorted(routes), ["/save", "/save/cancel"])
        for path, methods in routes.items():
            with self.subTest(path=path):
                self.assertEqual(methods, {"POST"})


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

    def test_a_provider_timeout_maps_to_504_not_503(self):
        # APITimeoutError is derived from APIConnectionError, so it must be
        # reported as a timeout: the provider *was* reached, it just did not
        # answer inside the time this server allows (see the caps in main.py).
        error = APITimeoutError(request=mock.Mock())

        with mock.patch.object(server.main, "chat", side_effect=error):
            response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.status_code, 504)
        self.assertEqual(
            response.json()["detail"],
            "The model provider took too long to answer.",
        )

    def test_an_unreachable_provider_still_maps_to_503(self):
        error = APIConnectionError(request=mock.Mock())

        with mock.patch.object(server.main, "chat", side_effect=error):
            response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"], "Could not reach the model provider."
        )

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

class PendingStoreTests(unittest.TestCase):
    """The pending store on its own: ids, lifetime, capacity and single use."""

    def setUp(self):
        self.clock = FakeClock()
        self.store = server.PendingSaveStore(clock=self.clock)
        self.proposal = make_proposal()

    def test_a_registered_proposal_returns_its_public_view(self):
        view = self.store.register(self.proposal)

        self.assertEqual(
            sorted(view), sorted(server.PUBLIC_SAVE_FIELDS + ("preview",))
        )
        self.assertIsInstance(view["save_id"], str)
        self.assertGreaterEqual(len(view["save_id"]), 32)
        self.assertEqual(view["title"], self.proposal["title"])
        self.assertEqual(view["summary"], self.proposal["summary"])
        self.assertEqual(view["filename"], self.proposal["filename"])
        self.assertEqual(
            view["sources"],
            [{"title": "Example", "url": "https://example.com/a"}],
        )
        # The preview is the exact document that would be written.
        self.assertEqual(view["preview"], self.proposal["content"])
        self.assertEqual(self.store.count(), 1)

    def test_every_proposal_gets_its_own_id(self):
        first = self.store.register(make_proposal(token="t1"))
        second = self.store.register(make_proposal(token="t2"))

        self.assertNotEqual(first["save_id"], second["save_id"])
        self.assertEqual(self.store.count(), 2)

    def test_an_unknown_id_is_never_usable(self):
        for save_id in ("", "unknown", "a" * 200, None, 5, [], {}):
            with self.subTest(save_id=save_id):
                self.assertIsNone(self.store.view(save_id))
                self.assertIsNone(self.store.begin_save(save_id))
                self.assertFalse(self.store.cancel(save_id))
                self.assertFalse(self.store.restore(save_id))
                self.assertFalse(self.store.discard(save_id))

    def test_a_proposal_expires_after_the_ttl(self):
        view = self.store.register(self.proposal)

        self.clock.advance(server.PENDING_SAVE_TTL_SECONDS - 1)
        self.assertIsNotNone(self.store.view(view["save_id"]))

        self.clock.advance(2)
        self.assertIsNone(self.store.view(view["save_id"]))
        self.assertIsNone(self.store.begin_save(view["save_id"]))
        self.assertFalse(self.store.cancel(view["save_id"]))
        self.assertEqual(self.store.count(), 0)

    def test_the_store_refuses_more_than_its_capacity(self):
        store = server.PendingSaveStore(capacity=2, clock=self.clock)

        self.assertIsNotNone(store.register(make_proposal(token="t1")))
        self.assertIsNotNone(store.register(make_proposal(token="t2")))
        self.assertIsNone(store.register(make_proposal(token="t3")))
        self.assertEqual(store.count(), 2)

    def test_the_default_capacity_is_a_real_limit(self):
        for index in range(server.MAX_PENDING_SAVES):
            with self.subTest(index=index):
                self.assertIsNotNone(
                    self.store.register(make_proposal(token=f"t{index}"))
                )

        self.assertIsNone(self.store.register(make_proposal(token="extra")))
        self.assertEqual(self.store.count(), server.MAX_PENDING_SAVES)

    def test_expired_proposals_free_capacity(self):
        store = server.PendingSaveStore(capacity=1, clock=self.clock)

        self.assertIsNotNone(store.register(make_proposal(token="t1")))
        self.assertIsNone(store.register(make_proposal(token="t2")))

        self.clock.advance(server.PENDING_SAVE_TTL_SECONDS + 1)
        self.assertIsNotNone(store.register(make_proposal(token="t3")))
        self.assertEqual(store.count(), 1)

    def test_a_broken_mapping_is_never_stored(self):
        broken = (
            None,
            "a proposal",
            [],
            {},
            {"title": "t"},
            {"title": "t", "summary": "s", "filename": "f.md", "content": ""},
            {"title": "t", "summary": "s", "filename": "", "content": "c"},
            {"title": "", "summary": "s", "filename": "f.md", "content": "c"},
            {
                "title": "t",
                "summary": "s",
                "filename": "f.md",
                "content": "c",
                "sources": "https://example.com/a",
            },
        )

        for value in broken:
            with self.subTest(value=value):
                self.assertIsNone(self.store.register(value))
        self.assertEqual(self.store.count(), 0)

    def test_the_stored_copy_cannot_be_changed_from_outside(self):
        view = self.store.register(self.proposal)
        original_sources = [{"title": "Example", "url": "https://example.com/a"}]

        # Scalars *and* the nested source mappings are copied, not shared: a
        # mutation of the caller's object after registration cannot leak in.
        self.proposal["content"] = "changed after the registration"
        self.proposal["filename"] = "other.md"
        self.proposal["sources"][0]["url"] = "https://evil.example/x"
        self.proposal["sources"][0]["title"] = "changed"
        self.proposal["sources"].append(
            {"title": "extra", "url": "https://evil.example/y"}
        )

        self.assertEqual(
            self.store.view(view["save_id"])["preview"], view["preview"]
        )
        self.assertEqual(self.store.view(view["save_id"])["sources"], original_sources)
        taken = self.store.begin_save(view["save_id"])
        self.assertEqual(taken["filename"], view["filename"])
        self.assertEqual(taken["content"], view["preview"])
        self.assertEqual(taken["sources"], original_sources)

    def test_a_taken_proposal_can_be_saved_only_once(self):
        view = self.store.register(self.proposal)

        taken = self.store.begin_save(view["save_id"])
        self.assertEqual(taken["filename"], self.proposal["filename"])
        self.assertIsNone(self.store.begin_save(view["save_id"]))

        self.assertTrue(self.store.discard(view["save_id"]))
        self.assertIsNone(self.store.begin_save(view["save_id"]))
        self.assertIsNone(self.store.view(view["save_id"]))
        self.assertEqual(self.store.count(), 0)

    def test_a_saving_proposal_cannot_be_cancelled(self):
        view = self.store.register(self.proposal)
        self.assertIsNotNone(self.store.begin_save(view["save_id"]))

        self.assertFalse(self.store.cancel(view["save_id"]))
        self.assertEqual(self.store.count(), 1)

    def test_a_cancelled_proposal_is_gone(self):
        view = self.store.register(self.proposal)

        self.assertTrue(self.store.cancel(view["save_id"]))
        self.assertFalse(self.store.cancel(view["save_id"]))
        self.assertIsNone(self.store.begin_save(view["save_id"]))
        self.assertEqual(self.store.count(), 0)

    def test_a_failed_write_can_be_retried(self):
        view = self.store.register(self.proposal)
        self.store.begin_save(view["save_id"])

        self.assertTrue(self.store.restore(view["save_id"]))
        self.assertIsNotNone(self.store.view(view["save_id"]))
        self.assertIsNotNone(self.store.begin_save(view["save_id"]))

    def test_restore_never_revives_a_proposal_that_was_not_taken(self):
        view = self.store.register(self.proposal)

        self.assertFalse(self.store.restore(view["save_id"]))
        self.store.discard(view["save_id"])
        self.assertFalse(self.store.restore(view["save_id"]))

    def test_restore_never_revives_an_expired_proposal(self):
        view = self.store.register(self.proposal)
        self.store.begin_save(view["save_id"])

        self.clock.advance(server.PENDING_SAVE_TTL_SECONDS + 1)

        self.assertFalse(self.store.restore(view["save_id"]))
        self.assertIsNone(self.store.view(view["save_id"]))
        self.assertEqual(self.store.count(), 0)

    def test_a_purge_leaves_a_proposal_that_is_being_written_alone(self):
        view = self.store.register(self.proposal)
        self.assertIsNotNone(self.store.begin_save(view["save_id"]))

        self.clock.advance(server.PENDING_SAVE_TTL_SECONDS + 1)

        # Any other request may purge (here through count() and register()), yet
        # the "saving" entry must stay: a writer still owns it.
        self.assertEqual(self.store.count(), 1)
        other = self.store.register(make_proposal(token="other"))
        self.assertIsNotNone(other)
        self.assertEqual(self.store.count(), 2)

        # It is still reserved for its writer: no second write, no cancellation.
        self.assertIsNone(self.store.begin_save(view["save_id"]))
        self.assertFalse(self.store.cancel(view["save_id"]))

        # Only the writer may retire it, and it may do so after the TTL.
        self.assertTrue(self.store.discard(view["save_id"]))
        self.assertEqual(self.store.count(), 1)

    def test_restore_still_drops_an_expired_proposal_after_a_purge(self):
        view = self.store.register(self.proposal)
        self.assertIsNotNone(self.store.begin_save(view["save_id"]))
        self.clock.advance(server.PENDING_SAVE_TTL_SECONDS + 1)

        # The purge kept the entry for its writer; the failed write now retires
        # it, so an expired proposal can never be used again.
        self.assertIsNotNone(self.store.register(make_proposal(token="other")))
        self.assertFalse(self.store.restore(view["save_id"]))
        self.assertIsNone(self.store.begin_save(view["save_id"]))
        self.assertIsNone(self.store.view(view["save_id"]))
        self.assertEqual(self.store.count(), 1)

    def test_a_document_too_large_for_a_preview_has_none(self):
        with mock.patch.object(server, "MAX_PREVIEW_LENGTH", 10):
            view = self.store.register(self.proposal)

        self.assertNotIn("preview", view)
        # The rest of the proposal is still reviewable.
        self.assertEqual(view["title"], self.proposal["title"])
        self.assertEqual(view["summary"], self.proposal["summary"])

    def test_a_document_with_markup_has_no_preview(self):
        view = self.store.register(
            make_proposal(analysis="<script>alert(1)</script>")
        )

        self.assertNotIn("preview", view)
        self.assertEqual(
            view["sources"], [{"title": "Example", "url": "https://example.com/a"}]
        )

    def test_a_harmless_comparison_keeps_its_preview(self):
        document = make_proposal(
            analysis="وقتی 5 < 10 باشد، پاسخ ساده است."
        )

        self.assertIn("5 < 10", document["content"])
        self.assertIn("preview", self.store.register(document))

    def test_only_real_links_survive_in_the_view(self):
        proposal = make_proposal(
            sources=[
                {"title": "Example", "url": "https://example.com/a"},
                {"title": "Duplicate", "url": "https://example.com/a"},
                {"title": "Bad", "url": "javascript:alert(1)"},
                {"title": "Local", "url": "file:///C:/secret.txt"},
            ]
        )

        view = self.store.register(proposal)

        self.assertEqual(
            view["sources"], [{"title": "Example", "url": "https://example.com/a"}]
        )
class PendingStoreConcurrencyTests(unittest.TestCase):
    """The store's lock: one winner per transition, and none while writing."""

    def setUp(self):
        self.clock = FakeClock()
        self.store = server.PendingSaveStore(clock=self.clock)
        self.proposal = make_proposal()

    def race(self, *actions):
        """Start every callable at the same instant and wait for all of them."""
        barrier = threading.Barrier(len(actions))
        errors = []

        def worker(action):
            try:
                barrier.wait(5)
                action()
            except BaseException as error:  # reported through ``errors``
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(action,)) for action in actions
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        for thread in threads:
            self.assertFalse(thread.is_alive(), "a racing thread never finished")
        self.assertEqual(errors, [])
        return threads

    def test_only_one_confirmation_wins(self):
        save_id = self.store.register(self.proposal)["save_id"]
        won = []

        self.race(
            *[lambda: won.append(self.store.begin_save(save_id)) for _ in range(8)]
        )

        self.assertEqual(len([entry for entry in won if entry is not None]), 1)
        self.assertEqual(len(won), 8)

    def test_only_one_cancellation_wins(self):
        save_id = self.store.register(self.proposal)["save_id"]
        results = []
        actions = [
            lambda: results.append(self.store.cancel(save_id)) for _ in range(8)
        ]

        self.race(*actions)

        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 7)

    def test_eight_confirmations_write_once(self):
        save_id = self.store.register(self.proposal)["save_id"]
        written = []

        def confirm():
            proposal = self.store.begin_save(save_id)
            if proposal is None:
                return
            written.append(proposal["filename"])
            self.store.discard(save_id)

        self.race(*[confirm] * 8)

        self.assertEqual(written, [self.proposal["filename"]])
        self.assertEqual(self.store.count(), 0)

    def test_confirmation_and_cancellation_cannot_both_win(self):
        for attempt in range(20):
            with self.subTest(attempt=attempt):
                save_id = self.store.register(
                    make_proposal(token=f"t{attempt}")
                )["save_id"]
                outcome = {}

                def confirm():
                    outcome["saved"] = self.store.begin_save(save_id) is not None

                def cancel():
                    outcome["cancelled"] = self.store.cancel(save_id)

                self.race(confirm, cancel)

                self.assertNotEqual(outcome["saved"], outcome["cancelled"])
                if outcome["saved"]:
                    # The winner holds the proposal in "saving" state.
                    self.assertFalse(self.store.cancel(save_id))
                    self.assertTrue(self.store.discard(save_id))
                self.assertEqual(self.store.count(), 0)

    def test_a_saving_proposal_stays_reserved_for_its_writer(self):
        save_id = self.store.register(self.proposal)["save_id"]
        self.assertIsNotNone(self.store.begin_save(save_id))

        # Nothing can slip in between: no second write, no cancellation.
        self.assertIsNone(self.store.begin_save(save_id))
        self.assertFalse(self.store.cancel(save_id))
        self.assertTrue(self.store.discard(save_id))
class SaveApiTestCase(unittest.TestCase):
    """Base for the endpoint tests: own store, own clock, own output folder.

    ``server.PENDING_SAVES`` is replaced for the duration of a test, so nothing
    leaks into the next one, and every real write is redirected into a
    ``tempfile`` folder that is removed again.
    """

    def setUp(self):
        self.clock = FakeClock()
        self.store = server.PendingSaveStore(clock=self.clock)
        self.proposal = None

        store_patcher = mock.patch.object(server, "PENDING_SAVES", self.store)
        store_patcher.start()
        self.addCleanup(store_patcher.stop)

        self.folder = tempfile.mkdtemp(prefix="chatbot-save-")
        self.addCleanup(shutil.rmtree, self.folder, True)

    # -- helpers ---------------------------------------------------------

    def put_in_store(self, **overrides):
        """Register a proposal the way /chat would, and return its view."""
        self.proposal = make_proposal(**overrides)
        view = self.store.register(self.proposal)
        self.assertIsNotNone(view)
        return view

    def files(self):
        """Everything that exists in the sandbox output folder."""
        return sorted(os.listdir(self.folder))

    def read_saved(self, filename):
        with open(os.path.join(self.folder, filename), encoding="utf-8") as handle:
            return handle.read()

    def write_into_folder(self):
        """Redirect save_tool's writer into the sandbox folder for the block."""
        real = save_tool.write_proposal

        def writer(proposal):
            return real(proposal, directory=self.folder)

        return mock.patch.object(save_tool, "write_proposal", side_effect=writer)

    def chat_with(self, *proposals, reply="پاسخ", history=None):
        """Mock main.chat so that the turn prepares the given proposals."""
        history = [] if history is None else history

        def fake_chat(message, sent_history):
            for proposal in proposals:
                self.assertTrue(main.record_proposal(proposal))
            return reply, sent_history

        return mock.patch.object(server.main, "chat", side_effect=fake_chat)
class ChatProposalTests(SaveApiTestCase):
    """POST /chat: the proposal is offered, nothing is written."""

    def test_a_proposal_is_returned_for_confirmation(self):
        proposal = make_proposal()

        with self.chat_with(proposal):
            response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(sorted(payload), ["history", "pending_save", "reply"])
        self.assertEqual(payload["reply"], "پاسخ")
        self.assertEqual(payload["history"], [])

        pending = payload["pending_save"]
        self.assertEqual(
            sorted(pending), sorted(server.PUBLIC_SAVE_FIELDS + ("preview",))
        )
        self.assertEqual(pending["title"], proposal["title"])
        self.assertEqual(pending["summary"], proposal["summary"])
        self.assertEqual(pending["filename"], proposal["filename"])
        self.assertEqual(
            pending["sources"],
            [{"title": "Example", "url": "https://example.com/a"}],
        )
        # The preview is the exact document, Markdown links included.
        self.assertEqual(pending["preview"], proposal["content"])
        self.assertIn("(https://example.com/a)", pending["preview"])
        self.assertEqual(self.store.count(), 1)

    def test_no_file_is_written_before_the_user_confirms(self):
        with mock.patch.object(save_tool, "write_proposal") as writer:
            with self.chat_with(make_proposal()):
                response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.status_code, 200)
        writer.assert_not_called()
        self.assertEqual(self.files(), [])

    def test_the_pending_save_carries_no_path_and_no_internal_data(self):
        with self.chat_with(make_proposal()):
            response = client.post("/chat", json={"message": "hi"})

        pending = response.json()["pending_save"]
        self.assertNotIn(self.folder, response.text)
        self.assertNotIn("search_results", response.text)
        for internal in ("analysis", "content", "bytes", "proposal", "state"):
            with self.subTest(field=internal):
                self.assertNotIn(internal, pending)

    def test_the_plain_contract_is_unchanged_without_a_proposal(self):
        history = [{"role": "user", "content": "q"}]
        with mock.patch.object(server.main, "chat", return_value=("سلام", history)):
            response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.json(), {"reply": "سلام", "history": history})
        self.assertEqual(sorted(response.json()), ["history", "reply"])
        self.assertEqual(self.store.count(), 0)

    def test_a_broken_proposal_does_not_change_the_contract(self):
        def fake_chat(message, sent_history):
            self.assertTrue(main.record_proposal({"title": "half a proposal"}))
            return "ok", []

        with mock.patch.object(server.main, "chat", side_effect=fake_chat):
            response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.json(), {"reply": "ok", "history": []})
        self.assertEqual(self.store.count(), 0)

    def test_a_full_store_does_not_change_the_contract(self):
        full = server.PendingSaveStore(capacity=0, clock=self.clock)
        with mock.patch.object(server, "PENDING_SAVES", full):
            with self.chat_with(make_proposal()):
                response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(sorted(response.json()), ["history", "reply"])

    def test_only_the_last_proposal_of_a_turn_is_registered(self):
        first = make_proposal(token="t1")
        second = make_proposal(token="t2")
        registered = []
        real_register = self.store.register

        def spy(proposal):
            registered.append(proposal)
            return real_register(proposal)

        with mock.patch.object(self.store, "register", side_effect=spy):
            with self.chat_with(first, second):
                response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(
            response.json()["pending_save"]["filename"], second["filename"]
        )
        # The earlier draft of the same turn never reaches the store.
        self.assertEqual(registered, [second])
        self.assertEqual(self.store.count(), 1)

    def test_an_invalid_last_proposal_is_not_replaced_by_an_earlier_one(self):
        valid = make_proposal(token="t1")

        with self.chat_with(valid, {"title": "half a proposal"}):
            response = client.post("/chat", json={"message": "hi"})

        # The last proposal is the one that counts; when it is unusable there is
        # no pending_save and the plain contract is left untouched.
        self.assertEqual(sorted(response.json()), ["history", "reply"])
        self.assertEqual(self.store.count(), 0)

    def test_a_proposal_with_invalid_sources_is_refused(self):
        broken_sources = (
            "https://example.com/a",
            ["https://example.com/a"],
            [{"title": "Example"}],
            [{"url": "https://example.com/a"}],
            [{"title": "", "url": "https://example.com/a"}],
            [{"title": "Example", "url": ""}],
            [{"title": "Example", "url": 5}],
            [{"title": "Example", "url": "javascript:alert(1)"}],
            [{"title": "Example", "url": " https://example.com/a "}],
            [{"title": "Example", "url": "https://example.com/a", "extra": 1}],
        )

        for sources in broken_sources:
            with self.subTest(sources=sources):
                proposal = make_proposal()
                proposal["sources"] = sources
                self.assertIsNone(self.store.register(proposal))
        self.assertEqual(self.store.count(), 0)

    def test_proposals_of_a_failed_turn_are_not_offered(self):
        def fake_chat(message, sent_history):
            main.record_proposal(make_proposal())
            raise main.ToolDeadlineExceeded("too slow")

        with mock.patch.object(server.main, "chat", side_effect=fake_chat):
            response = client.post("/chat", json={"message": "hi"})

        self.assertEqual(response.status_code, 504)
        self.assertEqual(self.store.count(), 0)

    def test_the_next_turn_does_not_inherit_a_proposal(self):
        with self.chat_with(make_proposal()):
            client.post("/chat", json={"message": "one"})

        with mock.patch.object(server.main, "chat", return_value=("ده", [])):
            response = client.post("/chat", json={"message": "two"})

        self.assertEqual(sorted(response.json()), ["history", "reply"])
        self.assertEqual(self.store.count(), 1)
class SaveEndpointTests(SaveApiTestCase):
    """POST /save: the only place where a research file is written."""

    def test_a_confirmed_proposal_is_written_once(self):
        view = self.put_in_store()

        with self.write_into_folder():
            response = client.post("/save", json={"save_id": view["save_id"]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(), {"status": "saved", "filename": view["filename"]}
        )
        self.assertEqual(self.files(), [view["filename"]])
        self.assertEqual(self.read_saved(view["filename"]), self.proposal["content"])
        # The id is used up, so the proposal is gone from the store.
        self.assertEqual(self.store.count(), 0)

    def test_the_confirmation_response_contains_no_path(self):
        view = self.put_in_store()

        with self.write_into_folder():
            response = client.post("/save", json={"save_id": view["save_id"]})

        self.assertEqual(sorted(response.json()), ["filename", "status"])
        self.assertNotIn(self.folder, response.text)
        self.assertNotIn("search_results", response.text)
        # A plain file name, not a location.
        self.assertEqual(response.json()["filename"], view["filename"])
        self.assertNotIn("\\", response.text)
        self.assertNotIn("/", response.json()["filename"])

    def test_an_unknown_id_is_a_generic_404(self):
        with mock.patch.object(save_tool, "write_proposal") as writer:
            response = client.post("/save", json={"save_id": "not-a-real-id"})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(error_detail(response), server.SAVE_NOT_FOUND_DETAIL)
        writer.assert_not_called()
        self.assertEqual(self.files(), [])

    def test_an_expired_id_is_a_generic_404(self):
        view = self.put_in_store()
        self.clock.advance(server.PENDING_SAVE_TTL_SECONDS + 1)

        with self.write_into_folder():
            response = client.post("/save", json={"save_id": view["save_id"]})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(error_detail(response), server.SAVE_NOT_FOUND_DETAIL)
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.files(), [])

    def test_an_already_used_id_is_refused(self):
        view = self.put_in_store()

        with self.write_into_folder():
            first = client.post("/save", json={"save_id": view["save_id"]})
            second = client.post("/save", json={"save_id": view["save_id"]})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 404)
        self.assertEqual(self.files(), [view["filename"]])

    def test_the_client_cannot_supply_content_or_a_location(self):
        view = self.put_in_store()
        forbidden = {
            "content": "# overwritten",
            "filename": "C:\\Windows\\evil.md",
            "path": "../../evil.md",
            "folder": "search_results",
            "directory": "..",
            "sources": [{"title": "x", "url": "https://evil.example/x"}],
        }

        with self.write_into_folder():
            for field, value in forbidden.items():
                with self.subTest(field=field):
                    response = client.post(
                        "/save", json={"save_id": view["save_id"], field: value}
                    )
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(
                        response.json(), {"detail": "Invalid request payload."}
                    )
            self.assertEqual(self.files(), [])
            # Nothing was consumed, so the real confirmation still works.
            response = client.post("/save", json={"save_id": view["save_id"]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.files(), [view["filename"]])
        self.assertEqual(self.read_saved(view["filename"]), self.proposal["content"])

    def test_a_malformed_payload_is_a_validation_error(self):
        for payload in (
            {},
            {"save_id": None},
            {"save_id": 5},
            {"save_id": ["an", "id"]},
            {"save_id": "a" * (server.MAX_SAVE_ID_LENGTH + 1)},
            {"other": "field"},
        ):
            with self.subTest(payload=payload):
                response = client.post("/save", json=payload)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(
                    response.json(), {"detail": "Invalid request payload."}
                )

    def test_an_empty_id_is_merely_unknown(self):
        # A syntactically fine but useless id is not a malformed payload.
        response = client.post("/save", json={"save_id": ""})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(error_detail(response), server.SAVE_NOT_FOUND_DETAIL)

    def test_a_filesystem_failure_is_a_generic_500_and_can_be_retried(self):
        view = self.put_in_store()
        secret = "C:\\secret\\research.md could not be saved"

        with mock.patch.object(
            save_tool, "write_proposal", side_effect=save_tool.SaveError(secret)
        ):
            failed = client.post("/save", json={"save_id": view["save_id"]})

        self.assertEqual(failed.status_code, 500)
        self.assertEqual(error_detail(failed), server.SAVE_FAILED_DETAIL)
        self.assertNotIn("secret", failed.text)
        self.assertNotIn("research.md", failed.text)
        self.assertEqual(self.files(), [])
        # The proposal waits again, so the user can simply retry.
        self.assertEqual(self.store.count(), 1)

        with self.write_into_folder():
            retried = client.post("/save", json={"save_id": view["save_id"]})

        self.assertEqual(retried.status_code, 200)
        self.assertEqual(self.files(), [view["filename"]])
        self.assertEqual(self.store.count(), 0)

    def test_an_unexpected_failure_is_also_generic(self):
        view = self.put_in_store()

        with mock.patch.object(
            save_tool, "write_proposal", side_effect=RuntimeError("boom at 12:00")
        ):
            failed = client.post("/save", json={"save_id": view["save_id"]})

        self.assertEqual(failed.status_code, 500)
        self.assertEqual(error_detail(failed), server.SAVE_FAILED_DETAIL)
        self.assertNotIn("boom", failed.text)
        self.assertEqual(self.store.count(), 1)

    def test_a_failure_after_expiry_cannot_revive_the_proposal(self):
        view = self.put_in_store()

        def expire_then_fail(proposal):
            self.clock.advance(server.PENDING_SAVE_TTL_SECONDS + 1)
            raise save_tool.SaveError("the research file could not be saved")

        with mock.patch.object(
            save_tool, "write_proposal", side_effect=expire_then_fail
        ):
            failed = client.post("/save", json={"save_id": view["save_id"]})

        self.assertEqual(failed.status_code, 500)
        self.assertEqual(self.store.count(), 0)

        with self.write_into_folder():
            retried = client.post("/save", json={"save_id": view["save_id"]})

        self.assertEqual(retried.status_code, 404)
        self.assertEqual(self.files(), [])

    def test_a_write_with_an_unusable_result_still_consumes_the_id(self):
        view = self.put_in_store()
        real_writer = save_tool.write_proposal

        def writer(proposal):
            # The document really is created, but the answer names a path.
            real_writer(proposal, directory=self.folder)
            return {"status": "saved", "file": "../escape.md"}

        with mock.patch.object(save_tool, "write_proposal", side_effect=writer):
            response = client.post("/save", json={"save_id": view["save_id"]})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(error_detail(response), server.SAVE_FAILED_DETAIL)
        self.assertNotIn("escape.md", response.text)

        # The file exists, so the id is used up: it must not go back to pending,
        # where a retry could try to write the very same file again.
        self.assertEqual(self.files(), [view["filename"]])
        self.assertEqual(self.store.count(), 0)

        with self.write_into_folder():
            retried = client.post("/save", json={"save_id": view["save_id"]})

        self.assertEqual(retried.status_code, 404)
        self.assertEqual(self.files(), [view["filename"]])

    def test_an_existing_file_is_never_overwritten(self):
        first = self.put_in_store()

        with self.write_into_folder():
            saved = client.post("/save", json={"save_id": first["save_id"]})

        self.assertEqual(saved.status_code, 200)
        before = self.read_saved(first["filename"])

        # A second proposal names the very same file.
        duplicate = self.put_in_store()
        self.assertEqual(duplicate["filename"], first["filename"])

        with self.write_into_folder():
            blocked = client.post("/save", json={"save_id": duplicate["save_id"]})

        self.assertEqual(blocked.status_code, 500)
        self.assertEqual(error_detail(blocked), server.SAVE_FAILED_DETAIL)
        self.assertEqual(self.read_saved(first["filename"]), before)
        self.assertEqual(self.files(), [first["filename"]])
        # Nothing was written, so that proposal is waiting again.
        self.assertEqual(self.store.count(), 1)

    def test_two_requests_can_never_write_the_same_proposal(self):
        view = self.put_in_store()
        started = threading.Event()
        release = threading.Event()
        calls = []
        results = []

        def slow_writer(proposal):
            calls.append(proposal["filename"])
            started.set()
            if not release.wait(10):
                raise AssertionError("the first write was never released")
            return {
                "status": "saved",
                "file": proposal["filename"],
                "title": proposal["title"],
                "sources": len(proposal["sources"]),
                "bytes": proposal["bytes"],
            }

        def confirm():
            try:
                results.append(
                    server.confirm_save(server.SaveRequest(save_id=view["save_id"]))
                )
            except HTTPException as error:
                results.append(error.status_code)

        with mock.patch.object(save_tool, "write_proposal", side_effect=slow_writer):
            worker = threading.Thread(target=confirm)
            worker.start()
            try:
                self.assertTrue(started.wait(10), "the first write never started")
                second = client.post("/save", json={"save_id": view["save_id"]})
                cancelled = client.post(
                    "/save/cancel", json={"save_id": view["save_id"]}
                )
            finally:
                release.set()
                worker.join(10)

        self.assertFalse(worker.is_alive(), "the writing thread never finished")
        self.assertEqual(second.status_code, 404)
        self.assertEqual(cancelled.status_code, 404)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            results, [{"status": "saved", "filename": view["filename"]}]
        )
        self.assertEqual(self.store.count(), 0)

    def test_the_store_lock_is_not_held_during_the_write(self):
        view = self.put_in_store()
        started = threading.Event()
        release = threading.Event()

        def slow_writer(proposal):
            started.set()
            if not release.wait(10):
                raise AssertionError("the write was never released")
            return {"status": "saved", "file": proposal["filename"]}

        def confirm():
            try:
                server.confirm_save(server.SaveRequest(save_id=view["save_id"]))
            except HTTPException:
                pass

        with mock.patch.object(save_tool, "write_proposal", side_effect=slow_writer):
            worker = threading.Thread(target=confirm)
            worker.start()
            try:
                self.assertTrue(started.wait(10), "the write never started")
                probe = threading.Thread(
                    target=lambda: self.store.register(make_proposal(token="probe"))
                )
                probe.start()
                probe.join(2)
                self.assertFalse(
                    probe.is_alive(),
                    "the store lock was held while the file system was written",
                )
            finally:
                release.set()
                worker.join(10)

        self.assertFalse(worker.is_alive(), "the writing thread never finished")
        self.assertEqual(self.store.count(), 1)

    def test_a_proposal_from_chat_can_be_confirmed_end_to_end(self):
        proposal = make_proposal()

        with self.chat_with(proposal):
            chat = client.post("/chat", json={"message": "hi"})

        self.assertEqual(chat.status_code, 200)
        save_id = chat.json()["pending_save"]["save_id"]
        self.assertEqual(self.files(), [])

        with self.write_into_folder():
            saved = client.post("/save", json={"save_id": save_id})

        self.assertEqual(saved.status_code, 200)
        self.assertEqual(saved.json()["filename"], proposal["filename"])
        self.assertEqual(self.files(), [proposal["filename"]])
        self.assertEqual(
            self.read_saved(proposal["filename"]), proposal["content"]
        )
class CancelEndpointTests(SaveApiTestCase):
    """POST /save/cancel: a waiting proposal can be dropped, once."""

    def test_a_cancelled_proposal_can_never_be_saved(self):
        view = self.put_in_store()

        response = client.post("/save/cancel", json={"save_id": view["save_id"]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "cancelled"})

        with mock.patch.object(save_tool, "write_proposal") as writer:
            saved = client.post("/save", json={"save_id": view["save_id"]})

        self.assertEqual(saved.status_code, 404)
        writer.assert_not_called()
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.files(), [])

    def test_cancelling_twice_is_a_generic_404(self):
        view = self.put_in_store()

        first = client.post("/save/cancel", json={"save_id": view["save_id"]})
        second = client.post("/save/cancel", json={"save_id": view["save_id"]})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 404)
        self.assertEqual(error_detail(second), server.SAVE_NOT_FOUND_DETAIL)

    def test_an_unknown_id_is_a_generic_404(self):
        for save_id in ("unknown", "", "a" * 40):
            with self.subTest(save_id=save_id):
                response = client.post("/save/cancel", json={"save_id": save_id})
                self.assertEqual(response.status_code, 404)
                self.assertEqual(
                    error_detail(response), server.SAVE_NOT_FOUND_DETAIL
                )

    def test_an_expired_proposal_cannot_be_cancelled(self):
        view = self.put_in_store()
        self.clock.advance(server.PENDING_SAVE_TTL_SECONDS + 1)

        response = client.post("/save/cancel", json={"save_id": view["save_id"]})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.store.count(), 0)

    def test_cancelling_never_touches_the_file_system(self):
        view = self.put_in_store()

        with mock.patch.object(save_tool, "write_proposal") as writer:
            response = client.post("/save/cancel", json={"save_id": view["save_id"]})

        self.assertEqual(response.status_code, 200)
        writer.assert_not_called()
        self.assertEqual(self.files(), [])

    def test_only_the_id_may_be_sent(self):
        view = self.put_in_store()

        for field, value in (
            ("content", "# overwritten"),
            ("filename", "evil.md"),
            ("path", "../../evil.md"),
            ("sources", []),
        ):
            with self.subTest(field=field):
                response = client.post(
                    "/save/cancel", json={"save_id": view["save_id"], field: value}
                )
                self.assertEqual(response.status_code, 422)
                self.assertEqual(
                    response.json(), {"detail": "Invalid request payload."}
                )

        # The proposal was not touched by any of those attempts.
        self.assertEqual(self.store.count(), 1)

    def test_a_proposal_being_written_cannot_be_cancelled(self):
        view = self.put_in_store()
        started = threading.Event()
        release = threading.Event()

        def slow_writer(proposal):
            started.set()
            if not release.wait(10):
                raise AssertionError("the write was never released")
            return {"status": "saved", "file": proposal["filename"]}

        def confirm():
            try:
                server.confirm_save(server.SaveRequest(save_id=view["save_id"]))
            except HTTPException:
                pass

        with mock.patch.object(save_tool, "write_proposal", side_effect=slow_writer):
            worker = threading.Thread(target=confirm)
            worker.start()
            try:
                self.assertTrue(started.wait(10), "the write never started")
                response = client.post(
                    "/save/cancel", json={"save_id": view["save_id"]}
                )
            finally:
                release.set()
                worker.join(10)

        self.assertFalse(worker.is_alive(), "the writing thread never finished")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.store.count(), 0)

if __name__ == "__main__":
    unittest.main()
