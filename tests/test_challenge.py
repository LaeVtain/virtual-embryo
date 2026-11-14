"""Offline tests: no real environment files, tokens, or API requests."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from virtual_embryo import challenge as client


SECRET = "offline-test-token-only"


def response(payload, status=200):
    result = Mock(status_code=status)
    result.json.return_value = payload
    return result


class ChallengeAPITests(unittest.TestCase):
    def test_p3_requires_explicit_confirmation_without_network(self):
        with patch.object(client, "load_token", return_value=SECRET), patch.object(client, "upload") as send:
            result = client.main(["submit", str(self.file), "--task", "T1", "--board", "T1:test",
                                  "--out", str(self.root / "p3.jsonl")])
        self.assertEqual(result, 2)
        send.assert_not_called()

    def test_p3_form_contains_split_and_confirmation(self):
        with patch.object(client, "load_token", return_value=SECRET), patch.object(client, "upload", return_value={"ok": True}) as send:
            result = client.main(["submit", str(self.file), "--task", "T1", "--board", "T1:test",
                                  "--confirm-final", "--out", str(self.root / "p3.jsonl")])
        self.assertEqual(result, 0)
        self.assertEqual(send.call_args.kwargs["fields"]["split"], "test")
        self.assertEqual(send.call_args.kwargs["fields"]["confirm_final"], "true")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.file = self.root / "fixture.h5ad"
        self.file.write_bytes(b"offline prediction fixture")
        self.sha = hashlib.sha256(self.file.read_bytes()).hexdigest()
        self.transport = Mock()
        self.api = client.ChallengeAPI(SECRET, transport=self.transport)
        self.fields = client.validate_fields("T2", "T2:embryo:val_interp", "embryo", "interp", model="fixture")

    def upload(self, operation="submit", name="attempt.jsonl", **kwargs):
        return client.upload(self.api, operation, self.file, board="T2:embryo:val_interp",
                             fields=self.fields, out=self.root / name, **kwargs)

    def accepted(self, **extra):
        return response({"id": "submission-1", "status": "queued", "issues": [],
                         "sha256": self.sha, "bundle_key": "T2:embryo:val_interp", **extra})

    def test_fixed_origin_redirects_disabled_and_multipart_boundary_not_overridden(self):
        self.transport.request.return_value = self.accepted()
        result = self.upload()
        self.assertTrue(result["ok"])
        args, kwargs = self.transport.request.call_args
        self.assertEqual(args, ("POST", client.API_BASE + "/challenge/submissions"))
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer " + SECRET)
        self.assertNotIn("Content-Type", kwargs["headers"])
        self.assertIn("file", kwargs["files"])
        self.assertNotIn(SECRET, (self.root / "attempt.jsonl").read_text())
        with self.assertRaises(client.ApiError):
            self.api.request("GET", "https://untrusted.invalid/challenge/me")

    def test_receipt_is_durable_before_post_and_existing_out_is_never_overwritten(self):
        def inspect_during_request(*args, **kwargs):
            events = client.read_receipt(self.root / "attempt.jsonl")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["idempotency_key"], kwargs["headers"]["Idempotency-Key"])
            return self.accepted()
        self.transport.request.side_effect = inspect_during_request
        self.upload()
        first = (self.root / "attempt.jsonl").read_bytes()
        with self.assertRaises(FileExistsError):
            self.upload()
        self.assertEqual((self.root / "attempt.jsonl").read_bytes(), first)
        self.assertEqual(self.transport.request.call_count, 1)

    def test_timeout_retry_reuses_key_and_refuses_changed_prediction(self):
        self.transport.request.side_effect = requests.Timeout("headers Authorization Bearer " + SECRET)
        with self.assertRaises(client.ApiError) as caught:
            self.upload()
        self.assertNotIn(SECRET, str(caught.exception))
        original = client.read_receipt(self.root / "attempt.jsonl")[0]
        self.transport.request.side_effect = None
        self.transport.request.return_value = self.accepted()
        self.upload(name="retry.jsonl", retry_from=self.root / "attempt.jsonl")
        retried = client.read_receipt(self.root / "retry.jsonl")[0]
        self.assertEqual(retried["idempotency_key"], original["idempotency_key"])
        self.file.write_bytes(b"different prediction")
        with self.assertRaises(client.ApiError):
            self.upload(name="bad-retry.jsonl", retry_from=self.root / "attempt.jsonl")
        self.assertFalse((self.root / "bad-retry.jsonl").exists())
        self.assertEqual(self.transport.request.call_count, 2)

    def test_format_only_has_no_idempotency_and_gates_exact_identity(self):
        self.transport.request.return_value = response({"ok": True, "format_only": True, "issues": []})
        self.upload("format-only", "format.jsonl")
        kwargs = self.transport.request.call_args.kwargs
        self.assertEqual(kwargs["data"]["format_only"], "true")
        self.assertNotIn("Idempotency-Key", kwargs["headers"])
        self.transport.request.return_value = self.accepted()
        self.upload(format_receipt=self.root / "format.jsonl")
        self.file.write_bytes(b"changed since format check")
        with self.assertRaises(client.ApiError):
            self.upload(name="new.jsonl", format_receipt=self.root / "format.jsonl")
        self.assertEqual(self.transport.request.call_count, 2)

    def test_http_200_rejection_and_format_issues_are_failures(self):
        self.transport.request.return_value = self.accepted(status="rejected", issues=["bad format"])
        self.assertFalse(self.upload()["ok"])
        self.transport.request.return_value = response({"ok": True, "format_only": True, "issues": ["invalid"]})
        self.assertFalse(self.upload("format-only", "format.jsonl")["ok"])
        self.transport.request.return_value = response({"ok": True, "format_only": True, "problems": ["invalid"]})
        self.assertFalse(self.upload("format-only", "format-problems.jsonl")["ok"])

    def test_http_failure_is_checked_before_json_and_redirect_is_not_followed(self):
        for code in (302, 401, 429, 500):
            with self.subTest(code=code):
                payload = response({"secret": SECRET}, status=code)
                self.transport.request.return_value = payload
                with self.assertRaises(client.ApiError):
                    self.upload(name=str(code) + ".jsonl")
                payload.json.assert_not_called()
                self.assertNotIn(SECRET, (self.root / (str(code) + ".jsonl")).read_text())

    def test_server_sha_mismatch_or_missing_and_wrong_board_stop(self):
        for name, updates in (("mismatch", {"sha256": "0" * 64}), ("missing", {"sha256": None}),
                              ("board", {"bundle_key": "T2:embryo:test_interp"})):
            self.transport.request.return_value = self.accepted(**updates)
            with self.assertRaises(client.ApiError):
                self.upload(name=name + ".jsonl")

    def test_sensitive_json_is_redacted_recursively(self):
        self.transport.request.return_value = self.accepted(
            debug={"SUBMIT_TOKEN": SECRET, SECRET: "value", "nested": ["echo " + SECRET, "Bearer other-secret"]})
        result = self.upload()
        serialized = json.dumps(result)
        self.assertNotIn(SECRET, serialized)
        self.assertNotIn("other-secret", serialized)
        self.assertIn("[REDACTED]", serialized)

    def test_env_is_explicit_and_never_sourced_or_interpolated(self):
        envfile = self.root / "fake.env"
        envfile.write_text("UNRELATED=$(do-not-execute)\nexport SUBMIT_TOKEN='offline-test-token-only'\n")
        self.assertEqual(client.load_token(envfile), SECRET)
        with patch.dict(os.environ, {"SUBMIT_TOKEN": "process-fixture"}, clear=True):
            self.assertEqual(client.load_token(), "process-fixture")
            self.assertEqual(client.load_token(envfile), SECRET)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(client.ApiError):
            client.load_token()

    def test_t2_contract_and_task_mismatch_are_rejected_before_network(self):
        for values in (("T2", "T2:embryo:val_extrap", "embryo", "extrap"),
                       ("T2", "T2:embryo:val_interp", "heart", "interp"),
                       ("T2", "T2:heart:test_interp", "heart", "interp"),
                       ("T1", "T2:embryo:val_interp", None, None),
                       ("T2", "T2:embryo:val_interp", None, None)):
            with self.assertRaises(client.ApiError):
                client.validate_fields(*values)
        self.transport.request.assert_not_called()

    def test_file_mutation_during_upload_is_not_a_successful_format_receipt(self):
        def mutate(*args, **kwargs):
            self.file.write_bytes(b"different while request in flight")
            return response({"ok": True, "format_only": True, "issues": []})
        self.transport.request.side_effect = mutate
        with self.assertRaises(client.ApiError):
            self.upload("format-only", "format.jsonl")
        events = client.read_receipt(self.root / "format.jsonl")
        self.assertEqual(events[-1]["event"], "error")
        self.assertNotIn("response", events[-1])


if __name__ == "__main__":
    unittest.main()
