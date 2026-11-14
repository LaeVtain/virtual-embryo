#!/usr/bin/env python3
"""Virtual Embryo API client. Receipts are exclusive-create, append-only JSONL.

Never changes account settings or accepts rules. Credentials are sent only to
the fixed official HTTPS origin; redirects are disabled.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

import requests


API_BASE = "https://api.virtualembryo.ai"
MAX_BYTES = 1200 * 1024 * 1024
SENSITIVE_KEYS = re.compile(r"token|secret|password|authorization|api.?key", re.I)
BOARD = re.compile(r"^(T[123]):([A-Za-z0-9_.:-]+)$")
SUBMISSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")


class ApiError(Exception):
    """A fixed, non-sensitive diagnostic suitable for CLI display."""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def redact(value, secret):
    """Sanitize remote JSON before it reaches stdout or an evidence file."""
    if isinstance(value, dict):
        return {redact(str(k), secret): "[REDACTED]" if SENSITIVE_KEYS.search(str(k)) else redact(v, secret)
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    if isinstance(value, str):
        value = value.replace(secret, "[REDACTED]") if secret else value
        return re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", value)
    return value


def load_token(env_file=None):
    """Read only the named .env or process SUBMIT_TOKEN, without sourcing code."""
    if env_file is None:
        token = os.environ.get("SUBMIT_TOKEN", "")
    else:
        token = ""
        for line in Path(env_file).read_text(encoding="utf-8-sig").splitlines():
            match = re.match(r"^\s*(?:export\s+)?SUBMIT_TOKEN\s*=\s*(.*?)\s*$", line)
            if not match:
                continue
            token = match.group(1)
            if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
                token = token[1:-1]
            else:
                token = re.split(r"\s+#", token, maxsplit=1)[0].strip()
    if not token or len(token) > 4096 or any(c.isspace() or ord(c) < 32 for c in token):
        raise ApiError("SUBMIT_TOKEN is missing or invalid; supply a process variable or explicit --env-file.")
    return token


def fingerprint(handle):
    handle.seek(0)
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        size += len(block)
        digest.update(block)
    handle.seek(0)
    return {"sha256": digest.hexdigest(), "bytes": size}


def append_event(handle, event):
    handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def read_receipt(path):
    try:
        events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not events or events[0].get("event") != "request" or events[0].get("api_base") != API_BASE:
            raise ValueError
        return events
    except (OSError, ValueError, TypeError, AttributeError):
        raise ApiError("Receipt is missing or invalid.") from None


class ChallengeAPI:
    def __init__(self, token, transport=None, timeout=180):
        self.token = token
        self.timeout = timeout
        self.transport = transport if transport is not None else requests.Session()
        if transport is None:
            self.transport.trust_env = False  # Do not inherit .netrc or proxy credentials.

    def request(self, method, path, *, fields=None, file=None, request_id=None):
        # No user-controlled URL, host, redirects, cookies, auth overrides or proxy
        # settings. An injected transport is intended solely for offline tests.
        if not path.startswith("/challenge/") or any(c in path for c in "?#\\\r\n"):
            raise ApiError("Invalid API path.")
        headers = {"Authorization": "Bearer " + self.token, "Accept": "application/json"}
        if request_id is not None:
            headers["Idempotency-Key"] = request_id
        kwargs = {"headers": headers, "timeout": (20, self.timeout), "allow_redirects": False}
        if fields is not None:
            kwargs["data"] = fields
        if file is not None:
            kwargs["files"] = {"file": (Path(file.name).name, file, "application/octet-stream")}
        try:
            response = self.transport.request(method, API_BASE + path, **kwargs)
        except (requests.RequestException, OSError):
            # Exception text can contain HTTP headers, proxy URLs or credentials.
            raise ApiError("Network failure; submission outcome may be unknown. Retry only with --retry-from this receipt.") from None
        code = response.status_code
        if code < 200 or code >= 300:
            raise ApiError("API returned HTTP %d; no response body was recorded." % code)
        try:
            payload = response.json()
        except (ValueError, TypeError):
            raise ApiError("API returned invalid JSON; submission outcome may be unknown. Preserve the receipt.") from None
        if not isinstance(payload, (dict, list)):
            raise ApiError("API returned an unexpected JSON structure; preserve the receipt.")
        return redact(payload, self.token)


def validate_fields(task, board, setting=None, mode=None, **optional):
    if not BOARD.fullmatch(board) or not board.startswith(task + ":"):
        raise ApiError("--board must identify the same task as --task.")
    if task == "T2":
        if setting not in ("heart", "embryo") or mode not in ("interp", "extrap"):
            raise ApiError("T2 requires --setting heart|embryo and --mode interp|extrap.")
        if setting == "embryo" and mode != "interp":
            raise ApiError("Embryo supports interpolation only.")
        if board not in (f"T2:{setting}:val_{mode}", f"T2:{setting}:test_{mode}"):
            raise ApiError("--board does not match the supplied T2 setting and mode.")
        if board == "T2:heart:test_interp":
            raise ApiError("Heart test submissions support extrapolation only.")
    elif setting is not None or mode is not None:
        raise ApiError("--setting and --mode apply only to T2.")
    fields = {"task": task}
    if task == "T2":
        fields.update(setting=setting, mode=mode)
    for key, value in optional.items():
        if value is not None:
            if len(value) > 300 or any(ord(c) < 32 for c in value) or SENSITIVE_KEYS.search(value):
                raise ApiError("Use short descriptive metadata without credentials or control characters.")
            fields[key] = value
    return fields


def check_format_receipt(path, identity, board):
    events = read_receipt(path)
    request = events[0]
    responses = [e for e in events if e.get("event") == "response"]
    if (request.get("operation") != "format-only" or request.get("artifact") != identity
            or request.get("board") != board or not responses or responses[-1].get("ok") is not True):
        raise ApiError("Format receipt must be successful and match the exact artifact SHA, byte size and board.")


def upload(api, operation, file_path, *, board, fields, out, format_receipt=None, retry_from=None):
    if operation not in ("submit", "format-only"):
        raise ApiError("Unknown upload operation.")
    with Path(file_path).open("rb") as prediction:
        identity = fingerprint(prediction)
        if not 0 < identity["bytes"] <= MAX_BYTES:
            raise ApiError("Prediction must be nonempty and at most 1200 MiB.")
        if api.token in str(file_path) or any(api.token in str(v) for v in fields.values()):
            raise ApiError("Credentials must not appear in file names or metadata.")
        if format_receipt is not None:
            check_format_receipt(format_receipt, identity, board)
        request_id = None
        if operation == "submit":
            if retry_from is not None:
                previous = read_receipt(retry_from)[0]
                if (previous.get("operation") != "submit" or previous.get("artifact") != identity
                        or previous.get("fields") != fields or previous.get("board") != board
                        or not SUBMISSION_ID.fullmatch(previous.get("idempotency_key", ""))):
                    raise ApiError("Retry must use the same exact file, board and metadata as the original request.")
                request_id = previous["idempotency_key"]
            else:
                request_id = "ve-" + uuid.uuid4().hex
        elif retry_from is not None:
            raise ApiError("--retry-from applies only to official submissions.")
        record = {"event": "request", "schema_version": 1, "created_utc": now(),
                  "api_base": API_BASE, "operation": operation, "board": board,
                  "fields": fields, "artifact": identity, "idempotency_key": request_id}
        # Creation precedes every POST. This journal survives unknown outcomes;
        # a retry creates a new journal but reuses the original idempotency key.
        with Path(out).open("x", encoding="utf-8") as receipt:
            append_event(receipt, record)
            try:
                sent_fields = dict(fields)
                if operation == "format-only":
                    sent_fields["format_only"] = "true"
                response = api.request("POST", "/challenge/submissions", fields=sent_fields,
                                       file=prediction, request_id=request_id)
                if fingerprint(prediction) != identity:
                    raise ApiError("Local file changed during the request; submission outcome may be unknown. Preserve the receipt.")
                if not isinstance(response, dict):
                    raise ApiError("Unexpected upload response; preserve the receipt.")
                server_sha = response.get("sha256")
                if ((operation == "submit" or server_sha is not None)
                        and server_sha != identity["sha256"]):
                    append_event(receipt, {"event": "response", "created_utc": now(), "ok": False,
                                           "response": response, "identity_matches": False})
                    raise ApiError("Server SHA-256 differs from the local file; stop and investigate this receipt.")
                if response.get("bundle_key") is not None and response["bundle_key"] != board:
                    append_event(receipt, {"event": "response", "created_utc": now(), "ok": False, "response": response})
                    raise ApiError("Server selected a different board; preserve this receipt and inspect the current phase.")
                ok = (not response.get("issues") and not response.get("problems")
                      and response.get("status") not in ("rejected", "cancelled"))
                if operation == "format-only":
                    ok = ok and response.get("ok") is True and response.get("format_only") is True
                else:
                    ok = ok and bool(response.get("id")) and response.get("status") in (
                        "queued", "scoring", "scored", "awaiting_evidence", "eligibility_review")
                append_event(receipt, {"event": "response", "created_utc": now(), "ok": ok, "response": response})
                return {"ok": ok, "receipt": str(out), "artifact": identity, "response": response}
            except ApiError as exc:
                append_event(receipt, {"event": "error", "created_utc": now(), "message": str(exc)})
                raise


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, help="explicit file containing SUBMIT_TOKEN; otherwise use process environment")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="GET current phase, boards, team, quota and submission history")
    inspect.add_argument("--out", type=Path)
    status = commands.add_parser("status", help="GET one submission; no automatic polling")
    status.add_argument("id")
    status.add_argument("--out", type=Path)
    for name in ("format-only", "submit"):
        upload_parser = commands.add_parser(name)
        upload_parser.add_argument("file", type=Path)
        upload_parser.add_argument("--task", choices=("T1", "T2", "T3"), required=True)
        upload_parser.add_argument("--board", required=True)
        upload_parser.add_argument("--setting", choices=("heart", "embryo"))
        upload_parser.add_argument("--mode", choices=("interp", "extrap"))
        upload_parser.add_argument("--split", choices=("val", "test"), help="must agree with --board")
        for field in ("model", "architecture", "agent-framework", "agent-model"):
            upload_parser.add_argument("--" + field)
        upload_parser.add_argument("--out", type=Path, required=True)
        if name == "submit":
            upload_parser.add_argument("--confirm-final", action="store_true",
                                       help="confirm spending one official P3 test attempt")
            upload_parser.add_argument("--format-receipt", type=Path)
            upload_parser.add_argument("--retry-from", type=Path)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        if args.out is not None and args.out.exists():
            raise ApiError("--out already exists; choose a new receipt path.")
        api = ChallengeAPI(load_token(args.env_file))
        if args.command in ("submit", "format-only"):
            fields = validate_fields(args.task, args.board, args.setting, args.mode,
                                     model=args.model, architecture=args.architecture,
                                     agent_framework=args.agent_framework, agent_model=args.agent_model)
            split = "test" if ":test" in args.board or args.board == "T3:bcatenin" else "val"
            if args.split is not None and args.split != split:
                raise ApiError("--split must agree with --board.")
            fields["split"] = split
            if args.command == "submit" and split == "test":
                if not args.confirm_final:
                    raise ApiError("P3 scoring requires --confirm-final to spend an official test attempt.")
                fields["confirm_final"] = "true"
            result = upload(api, args.command, args.file, board=args.board, fields=fields, out=args.out,
                            format_receipt=getattr(args, "format_receipt", None),
                            retry_from=getattr(args, "retry_from", None))
        else:
            # Reserve an output path before issuing requests; GETs never mutate
            # account state and may also be requested with stdout only.
            receipt = args.out.open("x", encoding="utf-8") if args.out is not None else None
            try:
                record = {"event": "request", "api_base": API_BASE, "operation": args.command, "created_utc": now()}
                if receipt:
                    append_event(receipt, record)
                if args.command == "inspect":
                    result = {name: api.request("GET", path) for name, path in (
                        ("phase", "/challenge/phase"), ("boards", "/challenge/boards"),
                        ("me", "/challenge/me"), ("quota", "/challenge/integration/quota"),
                        ("submissions", "/challenge/submissions"))}
                else:
                    if not SUBMISSION_ID.fullmatch(args.id):
                        raise ApiError("Invalid submission ID.")
                    result = api.request("GET", "/challenge/submissions/" + args.id)
                if receipt:
                    append_event(receipt, {"event": "response", "created_utc": now(), "response": result})
            finally:
                if receipt:
                    receipt.close()
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if isinstance(result, dict) and (not result.get("ok", True) or result.get("issues") or result.get("problems")
                                        or result.get("status") in ("rejected", "cancelled")):
            return 1
        return 0
    except ApiError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError):
        print("Could not read inputs or create the receipt; check files and choose a new output path.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
