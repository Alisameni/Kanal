"""Offline tests for save_tool.py, the Markdown research-save tool.

Nothing here touches the network, the project's real ``search_results`` folder or
any file outside a ``tempfile`` sandbox: every write test is redirected with the
``directory`` argument and removed again in tearDown. The socket entry points are
patched module-wide, so a live connection would be a clear failure instead of a
silent one.
"""

import ast
import json
import os
import shutil
import socket
import sys
import tempfile
import unittest
import urllib.request
from unittest import mock

import save_tool

# Patchers that make any socket use fail loudly; see setUpModule().
_GUARDS = []


def setUpModule():
    """Keep the whole module offline, not just the tests that remember to mock."""
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


def arguments(**overrides):
    """A minimal, valid save_research argument mapping."""
    base = {
        "title": "گزارش تحقیقاتی",
        "summary": "خلاصه یافتهها",
        "analysis": "متن تحلیل کامل",
        "sources": [{"title": "Example", "url": "https://example.com/a"}],
    }
    base.update(overrides)
    return base


def as_json(payload):
    """The mapping the model would send, serialized the way providers do."""
    return json.dumps(payload, ensure_ascii=False)


def proposal(stamp="20260101-000000", token="abc123", **overrides):
    """A proposal with a fixed name, so no test depends on the clock."""
    return save_tool.build_proposal(arguments(**overrides), stamp=stamp, token=token)


class ToolSpecTests(unittest.TestCase):
    """The model must only ever be offered the one, narrow tool."""

    def test_only_save_research_is_offered(self):
        tools = save_tool.build_tools()

        self.assertEqual(save_tool.TOOL_NAME, "save_research")
        self.assertEqual(len(tools), 1)
        spec = tools[0]
        self.assertEqual(spec["type"], "function")
        self.assertEqual(spec["function"]["name"], save_tool.TOOL_NAME)

    def test_the_schema_offers_content_only(self):
        parameters = save_tool.build_tools()[0]["function"]["parameters"]

        self.assertEqual(parameters["type"], "object")
        self.assertEqual(
            sorted(parameters["properties"]),
            ["analysis", "sources", "summary", "title"],
        )
        self.assertEqual(
            parameters["required"], ["title", "summary", "analysis", "sources"]
        )
        self.assertFalse(parameters["additionalProperties"])

    def test_the_model_cannot_choose_a_location(self):
        parameters = save_tool.build_tools()[0]["function"]["parameters"]

        for forbidden in ("path", "file", "filename", "directory", "folder", "name"):
            with self.subTest(property=forbidden):
                self.assertNotIn(forbidden, parameters["properties"])

    def test_the_description_says_it_only_prepares_a_proposal(self):
        # This description is the model's only instruction for the tool, so it
        # must say the tool prepares a draft and writes nothing: the model must
        # never read "ready" as "the file was saved".
        description = save_tool.build_tools()[0]["function"]["description"].lower()

        for expected in ("proposal", "ready", "confirm", "never claim"):
            with self.subTest(phrase=expected):
                self.assertIn(expected, description)
        self.assertIn("no file", description)

    def test_source_items_are_bounded_objects(self):
        items = save_tool.build_tools()[0]["function"]["parameters"]["properties"][
            "sources"
        ]["items"]

        self.assertEqual(items["type"], "object")
        self.assertEqual(sorted(items["properties"]), ["title", "url"])
        self.assertEqual(items["required"], ["url"])
        self.assertFalse(items["additionalProperties"])

    def test_tool_budgets_are_small(self):
        self.assertEqual(save_tool.OUTPUT_DIR_NAME, "search_results")
        self.assertEqual(save_tool.FILE_EXTENSION, ".md")
        self.assertEqual(save_tool.MAX_SOURCES, 5)
        self.assertGreater(save_tool.MAX_TITLE_LENGTH, 0)
        self.assertLessEqual(save_tool.MAX_TITLE_LENGTH, 500)
        self.assertLessEqual(save_tool.MAX_SUMMARY_LENGTH, 5_000)
        self.assertLessEqual(save_tool.MAX_ANALYSIS_LENGTH, 20_000)
        self.assertLessEqual(save_tool.MAX_URL_LENGTH, 2_000)
        self.assertGreater(save_tool.MAX_FILE_BYTES, 20_000)
        self.assertLessEqual(save_tool.MAX_FILE_BYTES, 1_000_000)
        self.assertLessEqual(save_tool.MAX_ACK_LENGTH, 2_000)

    def test_only_the_standard_library_is_imported(self):
        with open(save_tool.__file__, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        self.assertTrue(imported)
        self.assertLessEqual(imported, set(sys.stdlib_module_names))


class ArgumentValidationTests(unittest.TestCase):
    """Bad input is refused before anything is rendered or written."""

    def test_arguments_must_be_a_json_object(self):
        for raw in ("not json", "", "   ", "[1, 2]", "42", '"text"', None, 5, 1.5):
            with self.subTest(raw=raw):
                with self.assertRaises(save_tool.SaveError):
                    save_tool.parse_arguments(raw)

    def test_every_field_is_required(self):
        for missing in ("title", "summary", "analysis", "sources"):
            payload = arguments()
            payload.pop(missing)
            with self.subTest(missing=missing):
                with self.assertRaises(save_tool.SaveError):
                    save_tool.parse_arguments(as_json(payload))

    def test_text_fields_must_be_strings(self):
        for field in ("title", "summary", "analysis"):
            for value in (12, 1.5, None, [], {}):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(save_tool.SaveError):
                        save_tool.parse_arguments(as_json(arguments(**{field: value})))

    def test_a_blank_field_is_refused(self):
        for field in ("title", "summary", "analysis"):
            with self.subTest(field=field):
                with self.assertRaises(save_tool.SaveError):
                    save_tool.parse_arguments(as_json(arguments(**{field: "   "})))

    def test_overlong_fields_are_refused(self):
        limits = {
            "title": save_tool.MAX_TITLE_LENGTH,
            "summary": save_tool.MAX_SUMMARY_LENGTH,
            "analysis": save_tool.MAX_ANALYSIS_LENGTH,
        }
        for field, limit in limits.items():
            with self.subTest(field=field):
                with self.assertRaises(save_tool.SaveError) as caught:
                    save_tool.parse_arguments(
                        as_json(arguments(**{field: "x" * (limit + 1)}))
                    )
                self.assertIn("characters", str(caught.exception))

    def test_values_at_the_limit_are_accepted(self):
        clean = save_tool.parse_arguments(
            as_json(arguments(title="x" * save_tool.MAX_TITLE_LENGTH))
        )

        self.assertEqual(len(clean["title"]), save_tool.MAX_TITLE_LENGTH)

    def test_a_mapping_is_accepted(self):
        clean = save_tool.parse_arguments(arguments())

        self.assertEqual(clean["title"], "گزارش تحقیقاتی")
        self.assertEqual(
            clean["sources"], [{"title": "Example", "url": "https://example.com/a"}]
        )

    def test_bytes_arguments_are_decoded(self):
        raw = as_json(arguments()).encode("utf-8")

        self.assertEqual(save_tool.parse_arguments(raw)["title"], "گزارش تحقیقاتی")

    def test_undecodable_bytes_are_refused(self):
        with self.assertRaises(save_tool.SaveError):
            save_tool.parse_arguments(b"\xff\xfe{}")

    def test_values_are_cleaned(self):
        clean = save_tool.parse_arguments(
            as_json(arguments(title="  spaced\n  title  ", analysis="a\r\nb"))
        )

        self.assertEqual(clean["title"], "spaced title")
        self.assertEqual(clean["analysis"], "a\nb")

    def test_sources_must_be_a_list(self):
        for bad in ("https://example.com/a", 5, {"url": "https://example.com/a"}):
            with self.subTest(sources=bad):
                with self.assertRaises(save_tool.SaveError):
                    save_tool.parse_arguments(as_json(arguments(sources=bad)))

    def test_null_sources_mean_no_sources(self):
        clean = save_tool.parse_arguments(as_json(arguments(sources=None)))

        self.assertEqual(clean["sources"], [])

    def test_the_cleaned_mapping_only_holds_the_four_fields(self):
        clean = save_tool.parse_arguments(
            as_json(arguments(path="C:\\Windows\\evil.md", filename="evil.md"))
        )

        self.assertEqual(sorted(clean), ["analysis", "sources", "summary", "title"])


class SourceSelectionTests(unittest.TestCase):
    """Only real, unique http(s) sources ever reach the document."""

    def test_fake_sources_are_dropped(self):
        fakes = (
            "javascript:alert(1)",
            "ftp://example.com/a",
            "file:///etc/passwd",
            "not a url",
            "//example.com/a",
            "http://",
            "https://",
            "https://exa mple.com/a",
            "https://user:pass@example.com/a",
            "C:\\local\\file.md",
            "/etc/passwd",
            "",
            None,
            5,
            {},
            [],
        )
        for fake in fakes:
            with self.subTest(fake=fake):
                self.assertEqual(save_tool.select_sources([fake]), [])

    def test_invalid_and_duplicate_urls_are_dropped(self):
        selected = save_tool.select_sources(
            [
                {"title": "One", "url": "https://one.example/a"},
                {"title": "One again", "url": "https://one.example/a"},
                "https://two.example/b",
                "ftp://three.example/c",
            ]
        )

        self.assertEqual(
            [entry["url"] for entry in selected],
            ["https://one.example/a", "https://two.example/b"],
        )

    def test_sources_are_capped(self):
        many = [
            f"https://example.com/{index}"
            for index in range(save_tool.MAX_SOURCES + 3)
        ]

        self.assertEqual(len(save_tool.select_sources(many)), save_tool.MAX_SOURCES)

    def test_urls_the_search_never_returned_are_dropped(self):
        selected = save_tool.select_sources(
            ["https://real.example/a", "https://invented.example/b"],
            known_urls=["https://real.example/a"],
        )

        self.assertEqual(
            [entry["url"] for entry in selected], ["https://real.example/a"]
        )

    def test_without_known_urls_no_grounding_check_is_done(self):
        self.assertEqual(len(save_tool.select_sources(["https://any.example/a"])), 1)

    def test_an_empty_known_url_list_drops_everything(self):
        self.assertEqual(
            save_tool.select_sources(["https://any.example/a"], known_urls=[]), []
        )

    def test_a_source_title_falls_back_to_its_url(self):
        selected = save_tool.select_sources(["https://example.com/a"])

        self.assertEqual(
            selected,
            [{"title": "https://example.com/a", "url": "https://example.com/a"}],
        )

    def test_source_titles_are_bounded(self):
        selected = save_tool.select_sources(
            [{"title": "T" * 5000, "url": "https://example.com/a"}]
        )

        self.assertEqual(len(selected[0]["title"]), save_tool.MAX_SOURCE_TITLE_LENGTH)

    def test_an_overlong_url_is_dropped(self):
        long_url = "https://example.com/" + "x" * save_tool.MAX_URL_LENGTH

        self.assertEqual(save_tool.select_sources([long_url]), [])

    def test_a_non_collection_of_known_urls_is_reported(self):
        with self.assertRaises(save_tool.SaveError):
            save_tool.select_sources([], known_urls=5)


class MarkdownTests(unittest.TestCase):
    """render_markdown() produces one bounded, UTF-8 section per field."""

    def render(self, **overrides):
        return save_tool.render_markdown(
            save_tool.normalize_arguments(arguments(**overrides))
        )

    def test_every_field_gets_a_section(self):
        text = self.render()

        self.assertTrue(text.startswith("# گزارش تحقیقاتی"))
        self.assertIn("## خلاصه\n\nخلاصه یافتهها", text)
        self.assertIn("## تحلیل\n\nمتن تحلیل کامل", text)
        self.assertIn("## منابع", text)

    def test_sources_become_a_numbered_link_list(self):
        text = self.render(
            sources=[
                {"title": "First", "url": "https://one.example/a"},
                {"title": "Second", "url": "https://two.example/b"},
            ]
        )

        self.assertIn("1. [First](https://one.example/a)", text)
        self.assertIn("2. [Second](https://two.example/b)", text)

    def test_without_sources_the_document_says_so(self):
        self.assertIn("منبعی ثبت نشده است.", self.render(sources=[]))

    def test_the_output_is_valid_utf8(self):
        text = self.render()

        self.assertEqual(text.encode("utf-8").decode("utf-8"), text)
        self.assertTrue(text.endswith("\n"))

    def test_line_breaks_are_normalized_and_control_characters_are_dropped(self):
        text = self.render(analysis="line one\r\nline two\r\x07")

        self.assertIn("line one\nline two", text)
        self.assertNotIn("\r", text)
        self.assertNotIn("\x07", text)

    def test_link_labels_and_targets_are_escaped(self):
        text = self.render(
            sources=[
                {"title": "A [bracket] title", "url": "https://ex.example/a_(b)"}
            ]
        )

        self.assertIn(
            "[A \\[bracket\\] title](https://ex.example/a_%28b%29)", text
        )


class FilenameTests(unittest.TestCase):
    """The generated name is safe, bounded and practically unique."""

    def test_the_name_has_the_expected_shape(self):
        self.assertRegex(
            save_tool.make_filename("Tehran weather"),
            r"^\d{8}-\d{6}-tehran-weather-[0-9a-f]{6}\.md$",
        )

    def test_a_fixed_stamp_and_token_make_the_name_deterministic(self):
        self.assertEqual(
            save_tool.make_filename("Report", stamp="20260101-000000", token="abc123"),
            "20260101-000000-report-abc123.md",
        )

    def test_the_name_is_filesystem_safe(self):
        name = save_tool.make_filename(
            'a/b\\c:d*e?f"g<h>i|j .. k', stamp="20260101-000000", token="abc123"
        )

        self.assertTrue(name.endswith(save_tool.FILE_EXTENSION))
        self.assertNotIn("..", name)
        for char in '<>:"/\\|?*':
            self.assertNotIn(char, name)
        self.assertFalse(
            any(ord(char) < 32 or ord(char) == 127 for char in name)
        )

    def test_only_safe_characters_survive_in_the_stem(self):
        name = save_tool.make_filename(
            "Report: 2026/01/01 — «final»!", stamp="20260101-000000", token="abc123"
        )
        stem = name[: -len(save_tool.FILE_EXTENSION)]

        self.assertTrue(
            all(char.isalnum() or char == "-" for char in stem), msg=stem
        )

    def test_the_name_stays_short(self):
        name = save_tool.make_filename("x" * 500, stamp="9" * 80, token="f" * 80)

        self.assertLessEqual(len(name), 200)
        self.assertLessEqual(len(name.encode("utf-8")), 255)

    def test_a_title_without_usable_characters_falls_back(self):
        name = save_tool.make_filename(
            "!!! ... ???", stamp="20260101-000000", token="abc123"
        )

        self.assertEqual(
            name, f"20260101-000000-{save_tool.SLUG_FALLBACK}-abc123.md"
        )

    def test_persian_titles_keep_their_words(self):
        name = save_tool.make_filename(
            "گزارش وضعیت آب و هوا", stamp="20260101-000000", token="abc123"
        )

        self.assertEqual(name, "20260101-000000-گزارش-وضعیت-آب-و-هوا-abc123.md")

    def test_two_saves_of_the_same_title_do_not_collide(self):
        with mock.patch.object(
            save_tool, "utc_stamp", return_value="20260101-000000"
        ):
            first = save_tool.make_filename("Report")
            second = save_tool.make_filename("Report")

        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("20260101-000000-report-"))

    def test_reserved_device_names_are_never_produced(self):
        for title in ("CON", "prn", "Aux", "nul", "com1", "LPT9"):
            with self.subTest(title=title):
                name = save_tool.make_filename(
                    title, stamp="20260101-000000", token="abc123"
                )
                stem = name[: -len(save_tool.FILE_EXTENSION)].lower()
                self.assertNotIn(stem, save_tool.RESERVED_NAMES)


class TargetResolutionTests(unittest.TestCase):
    """resolve_target() can only ever point inside the output folder."""

    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="save_tool_target_")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_a_plain_name_resolves_inside_the_folder(self):
        target = save_tool.resolve_target("report.md", directory=self.folder)

        self.assertEqual(
            target, os.path.join(os.path.realpath(self.folder), "report.md")
        )
        self.assertTrue(target.startswith(os.path.realpath(self.folder)))

    def test_traversal_and_absolute_names_are_refused(self):
        bad_names = (
            "../escape.md",
            "..\\escape.md",
            "sub/escape.md",
            "sub\\escape.md",
            "C:\\Windows\\escape.md",
            "\\\\server\\share\\escape.md",
            "/etc/passwd",
            ".",
            "..",
            "",
            "   ",
            "escape.md ",
            " escape.md",
            ".hidden.md",
            "escape.txt",
            "escape.md.bak",
            "escape:md.md",
            "e\x00sc.md",
            "esc\nape.md",
        )
        for name in bad_names:
            with self.subTest(name=name):
                with self.assertRaises(save_tool.SaveError):
                    save_tool.resolve_target(name, directory=self.folder)

    def test_non_string_names_are_refused(self):
        for name in (None, 5, b"escape.md", ["escape.md"]):
            with self.subTest(name=name):
                with self.assertRaises(save_tool.SaveError):
                    save_tool.resolve_target(name, directory=self.folder)

    def test_resolving_never_creates_anything(self):
        save_tool.resolve_target("report.md", directory=self.folder)

        self.assertEqual(os.listdir(self.folder), [])


class OutputFolderTests(unittest.TestCase):
    """The default folder is fixed and lives inside the project."""

    def test_the_default_folder_is_the_project_search_results_folder(self):
        project_root = os.path.dirname(os.path.abspath(save_tool.__file__))

        self.assertEqual(save_tool.OUTPUT_DIR_NAME, "search_results")
        self.assertEqual(save_tool.PROJECT_ROOT, project_root)
        self.assertEqual(
            save_tool.output_directory(),
            os.path.join(project_root, save_tool.OUTPUT_DIR_NAME),
        )

    def test_the_default_folder_is_absolute(self):
        self.assertTrue(os.path.isabs(save_tool.output_directory()))

    def test_an_override_is_only_used_when_it_is_passed(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(
                save_tool.output_directory(folder), os.path.abspath(folder)
            )
        self.assertEqual(save_tool.output_directory(), save_tool.OUTPUT_DIR)


class ProposalTests(unittest.TestCase):
    """build_proposal() only renders text; it never touches the file system."""

    def test_build_proposal_touches_no_files(self):
        with mock.patch("builtins.open", side_effect=AssertionError("open()")):
            with mock.patch("os.makedirs", side_effect=AssertionError("makedirs()")):
                with mock.patch("os.remove", side_effect=AssertionError("remove()")):
                    result = proposal()

        self.assertIn("content", result)

    def test_the_proposal_holds_content_size_and_name(self):
        result = proposal()

        self.assertEqual(
            sorted(result),
            ["analysis", "bytes", "content", "filename", "sources", "summary", "title"],
        )
        self.assertEqual(result["bytes"], len(result["content"].encode("utf-8")))
        self.assertTrue(result["filename"].endswith(save_tool.FILE_EXTENSION))
        self.assertEqual(len(result["sources"]), 1)

    def test_the_proposal_never_contains_an_absolute_path(self):
        result = proposal()

        self.assertNotIn(save_tool.OUTPUT_DIR, json.dumps(result))
        self.assertNotIn(save_tool.PROJECT_ROOT, result["filename"])
        for separator in ("/", "\\"):
            self.assertNotIn(separator, result["filename"])

    def test_a_raw_json_string_is_accepted(self):
        result = save_tool.build_proposal(
            as_json(arguments()), stamp="20260101-000000", token="abc123"
        )

        self.assertIn("# گزارش تحقیقاتی", result["content"])

    def test_invalid_arguments_raise(self):
        with self.assertRaises(save_tool.SaveError):
            save_tool.build_proposal({"title": "only a title"})

    def test_the_file_size_limit_is_enforced(self):
        with mock.patch.object(save_tool, "MAX_FILE_BYTES", 10):
            with self.assertRaises(save_tool.SaveError) as caught:
                proposal()

        self.assertIn("bytes", str(caught.exception))

    def test_a_document_at_the_field_limits_still_fits(self):
        result = proposal(
            analysis="x" * save_tool.MAX_ANALYSIS_LENGTH,
            summary="y" * save_tool.MAX_SUMMARY_LENGTH,
        )

        self.assertLessEqual(result["bytes"], save_tool.MAX_FILE_BYTES)


class ProposalCopyTests(unittest.TestCase):
    """copy_proposal(): the one validator shared by the store and the writer."""

    def test_a_build_proposal_result_is_accepted_and_copied(self):
        original = proposal()

        copied = save_tool.copy_proposal(original)

        self.assertEqual(copied, original)
        self.assertIsNot(copied, original)
        self.assertIsNot(copied["sources"], original["sources"])
        self.assertIsNot(copied["sources"][0], original["sources"][0])

    def test_the_copy_is_independent_of_the_original(self):
        original = proposal()
        copied = save_tool.copy_proposal(original)

        original["content"] = "changed after the copy"
        original["sources"][0]["url"] = "https://evil.example/x"
        original["sources"][0]["title"] = "changed"
        original["sources"].append({"title": "x", "url": "https://evil.example/y"})

        self.assertEqual(copied["content"], proposal()["content"])
        self.assertEqual(
            copied["sources"],
            [{"title": "Example", "url": "https://example.com/a"}],
        )

    def test_partial_or_foreign_mappings_are_refused(self):
        good = proposal()
        bad = (
            None,
            "a proposal",
            [],
            {},
            {"title": "t"},
            {field: good[field] for field in good if field != "analysis"},
            dict(good, title=""),
            dict(good, summary=5),
            dict(good, content=None),
            dict(good, filename=""),
            dict(good, filename="../escape.md"),
            dict(good, sources="https://example.com/a"),
            dict(good, bytes="12"),
            dict(good, bytes=-1),
            dict(good, bytes=True),
        )
        for value in bad:
            with self.subTest(value=value):
                self.assertIsNone(save_tool.copy_proposal(value))

    def test_broken_source_entries_are_refused(self):
        good = proposal()
        bad_sources = (
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
        for sources in bad_sources:
            with self.subTest(sources=sources):
                self.assertIsNone(save_tool.copy_proposal(dict(good, sources=sources)))

    def test_a_missing_size_is_recomputed(self):
        good = proposal()
        del good["bytes"]

        copied = save_tool.copy_proposal(good)

        self.assertEqual(copied["bytes"], len(copied["content"].encode("utf-8")))


class WriteProposalTests(unittest.TestCase):
    """write_proposal() is the only writer, and it never overwrites."""

    def setUp(self):
        self.sandbox = tempfile.mkdtemp(prefix="save_tool_save_")
        self.addCleanup(shutil.rmtree, self.sandbox, True)
        self.folder = os.path.join(self.sandbox, "out")

    def listing(self):
        if not os.path.isdir(self.folder):
            return []
        return sorted(os.listdir(self.folder))

    def read(self, name):
        with open(os.path.join(self.folder, name), "rb") as handle:
            return handle.read()

    def test_a_successful_save_writes_utf8_markdown(self):
        written = proposal()
        result = save_tool.write_proposal(written, directory=self.folder)

        raw = self.read(written["filename"])
        self.assertEqual(raw.decode("utf-8"), written["content"])
        self.assertEqual(len(raw), written["bytes"])
        self.assertEqual(result["status"], "saved")
        self.assertEqual(result["file"], written["filename"])
        self.assertEqual(result["title"], written["title"])
        self.assertEqual(result["sources"], 1)
        self.assertEqual(result["bytes"], written["bytes"])

    def test_persian_text_survives_a_round_trip(self):
        written = proposal(title="گزارش هواشناسی", analysis="تحلیل فارسی")
        save_tool.write_proposal(written, directory=self.folder)

        text = self.read(written["filename"]).decode("utf-8")
        self.assertIn("# گزارش هواشناسی", text)
        self.assertIn("تحلیل فارسی", text)

    def test_the_output_folder_is_created_when_missing(self):
        result = save_tool.write_proposal(proposal(), directory=self.folder)

        self.assertTrue(os.path.isfile(os.path.join(self.folder, result["file"])))

    def test_an_existing_file_is_never_overwritten(self):
        written = proposal()
        save_tool.write_proposal(written, directory=self.folder)
        before = self.read(written["filename"])

        with self.assertRaises(save_tool.SaveError) as caught:
            save_tool.write_proposal(written, directory=self.folder)

        self.assertIn("already exists", str(caught.exception))
        self.assertEqual(self.read(written["filename"]), before)
        self.assertEqual(self.listing(), [written["filename"]])

    def test_the_result_never_contains_an_absolute_path(self):
        result = save_tool.write_proposal(proposal(), directory=self.folder)

        self.assertEqual(
            sorted(result), ["bytes", "file", "sources", "status", "title"]
        )
        self.assertNotIn(self.folder, json.dumps(result))
        self.assertNotIn(save_tool.OUTPUT_DIR, json.dumps(result))

    def test_a_traversal_name_is_refused_without_writing(self):
        written = proposal()
        written["filename"] = "../escape.md"

        with self.assertRaises(save_tool.SaveError) as caught:
            save_tool.write_proposal(written, directory=self.folder)

        self.assertNotIn(self.sandbox, str(caught.exception))
        self.assertEqual(self.listing(), [])
        self.assertFalse(os.path.exists(os.path.join(self.sandbox, "escape.md")))

    def test_a_failed_write_leaves_no_partial_file(self):
        written = proposal()

        with mock.patch("save_tool.os.fsync", side_effect=OSError("disk full")):
            with self.assertRaises(save_tool.SaveError) as caught:
                save_tool.write_proposal(written, directory=self.folder)

        self.assertIn("could not be saved", str(caught.exception))
        self.assertEqual(self.listing(), [])

    def test_io_failures_become_a_path_free_error(self):
        with mock.patch("builtins.open", side_effect=PermissionError("denied")):
            with self.assertRaises(save_tool.SaveError) as caught:
                save_tool.write_proposal(proposal(), directory=self.folder)

        message = str(caught.exception)
        self.assertNotIn(self.folder, message)
        self.assertNotIn(self.sandbox, message)
        self.assertNotIn("/", message)
        self.assertNotIn("\\", message)
        self.assertEqual(self.listing(), [])

    def test_a_folder_that_cannot_be_created_is_reported(self):
        with mock.patch("os.makedirs", side_effect=OSError("denied")):
            with self.assertRaises(save_tool.SaveError) as caught:
                save_tool.write_proposal(proposal(), directory=self.folder)

        self.assertIn("could not be created", str(caught.exception))

    def test_proposals_must_come_from_build_proposal(self):
        good = proposal()
        partial = {field: good[field] for field in good if field != "analysis"}
        bad = (
            None,
            5,
            [],
            {},
            {"filename": "a.md"},
            {"content": "x"},
            partial,
            dict(good, sources=[{"title": "Example"}]),
            dict(good, filename="../escape.md"),
        )
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(save_tool.SaveError):
                    save_tool.write_proposal(value, directory=self.folder)
        self.assertEqual(self.listing(), [])

    def test_the_size_limit_is_enforced_before_writing(self):
        with mock.patch.object(save_tool, "MAX_FILE_BYTES", 10):
            with self.assertRaises(save_tool.SaveError) as caught:
                save_tool.write_proposal(proposal(), directory=self.folder)

        self.assertIn("bytes", str(caught.exception))
        self.assertEqual(self.listing(), [])

    def test_each_call_writes_exactly_one_file(self):
        first = proposal(token="t1")
        second = proposal(token="t2")
        save_tool.write_proposal(first, directory=self.folder)
        save_tool.write_proposal(second, directory=self.folder)

        self.assertEqual(
            self.listing(), sorted([first["filename"], second["filename"]])
        )


class AckTests(unittest.TestCase):
    """The acknowledgement is small, valid JSON, and leaks nothing."""

    def test_only_the_allowed_fields_survive(self):
        text = save_tool.ack_content(
            {
                "status": "saved",
                "file": "a.md",
                "title": "t",
                "sources": 2,
                "bytes": 10,
                "path": "C:\\secret\\a.md",
                "content": "hidden",
            }
        )

        payload = json.loads(text)
        self.assertEqual(
            sorted(payload), ["bytes", "file", "sources", "status", "title"]
        )
        self.assertNotIn("secret", text)
        self.assertNotIn("hidden", text)

    def test_the_result_of_a_write_is_small_and_path_free(self):
        with tempfile.TemporaryDirectory() as folder:
            written = proposal()
            text = save_tool.ack_content(
                save_tool.write_proposal(written, directory=folder)
            )

        payload = json.loads(text)
        self.assertEqual(payload["status"], "saved")
        self.assertEqual(payload["file"], written["filename"])
        self.assertNotIn(folder, text)

    def test_output_is_bounded(self):
        text = save_tool.ack_content(
            {
                "status": "saved",
                "file": "f" * 5000,
                "title": "t" * 5000,
                "sources": 3,
                "bytes": 9,
            }
        )

        payload = json.loads(text)
        self.assertLessEqual(len(text), save_tool.MAX_ACK_LENGTH)
        self.assertEqual(payload["status"], "saved")
        for field in ("file", "title"):
            self.assertLessEqual(len(payload[field]), save_tool.MAX_ACK_FIELD_LENGTH)

    def test_a_tight_limit_still_returns_json(self):
        with mock.patch.object(save_tool, "MAX_ACK_LENGTH", 10):
            text = save_tool.ack_content(
                {"status": "saved", "file": "a.md", "title": "t"}
            )

        self.assertEqual(json.loads(text)["status"], "saved")

    def test_broken_values_are_skipped_quietly(self):
        text = save_tool.ack_content(
            {"status": "saved", "sources": "many", "bytes": None}
        )

        self.assertEqual(json.loads(text), {"status": "saved"})

    def test_garbage_input_still_gives_valid_json(self):
        for value in (None, 5, "x", [], object()):
            with self.subTest(value=value):
                self.assertEqual(json.loads(save_tool.ack_content(value)), {})

    def test_error_content_is_short_and_path_free(self):
        text = save_tool.error_content("C:\\secret\\a.md could not be saved")

        payload = json.loads(text)
        self.assertEqual(payload["status"], "error")
        self.assertLessEqual(len(payload["error"]), save_tool.MAX_ERROR_LENGTH)
        self.assertLessEqual(len(text), save_tool.MAX_ACK_LENGTH)

    def test_an_empty_error_message_has_a_fallback(self):
        self.assertEqual(
            json.loads(save_tool.error_content("   "))["error"],
            "the save tool failed",
        )


if __name__ == "__main__":
    unittest.main()
