"""Small SharedNet Room transport used by RAC episode bridges.

The Room is the communication plane only.  RAC remains the control plane: it
chooses capabilities, verifies artifacts, commits or rolls back work, and owns
all stopping decisions.  Tokens are kept in memory and never serialized into
episode metadata or coordination ledgers.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import warnings
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOM_RE = re.compile(r"\brom_[A-Za-z0-9]+\b")
INVITE_RE = re.compile(r"(?<![A-Za-z0-9_-])rit_[A-Za-z0-9_-]{43}(?![A-Za-z0-9_-])")
BASE_RE = re.compile(r"\bBASE=(\S+)")
URL_ROOM_RE = re.compile(r"https?://[^\s/]+(?:/[^\s]*)?/rooms/(rom_[A-Za-z0-9]+)")
TRAILER = "sharednet-rac: "
MAX_MESSAGE_BYTES = 32768
MAX_FIELD_BYTES = 4096
RESULT_SUMMARY_BYTES = 20000
SHAREDNET_ENV_KEYS = {"SHAREDNET_BASE_URL", "SHAREDNET_INVITE", "SHAREDNET_ROOM_ID"}


def _canonical_base(value: str) -> str:
    base = value.rstrip("/")
    return "https://www.sharednet.ai" if base == "https://sharednet.ai" else base


def load_sharednet_env(path: Path) -> dict[str, str]:
    """Read only SharedNet settings from a per-run dotenv file.

    Values are treated as literal configuration, never shell-expanded.  The
    caller keeps the file outside the evaluated workspace so member and invite
    tokens cannot become agent-visible artifacts.
    """
    if not path.is_file():
        return {}
    settings: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid dotenv assignment at {path}:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key in SHAREDNET_ENV_KEYS:
            settings[key] = value
    return settings


@dataclass(frozen=True)
class SharedNetInvite:
    base_url: str
    room_id: str
    token: str

    @classmethod
    def parse(cls, text: str, *, room_id: str, default_base: str = "https://www.sharednet.ai") -> "SharedNetInvite":
        token_match = INVITE_RE.search(text or "")
        embedded_room = ROOM_RE.search(text or "")
        if not token_match:
            raise ValueError("SHAREDNET_INVITE must contain a rit_ invite token")
        if not ROOM_RE.fullmatch(room_id):
            raise ValueError("sharednet room id must start with rom_ and contain only letters or digits")
        if embedded_room and embedded_room.group(0) != room_id:
            raise ValueError(
                f"--sharednet-room-id {room_id!r} does not match the room in SHAREDNET_INVITE"
            )
        base_match = BASE_RE.search(text or "")
        url_match = URL_ROOM_RE.search(text or "")
        if base_match:
            base = base_match.group(1)
        elif url_match:
            base = url_match.group(0).split("/api/")[0].split("/rooms/")[0]
        else:
            base = default_base
        return cls(_canonical_base(base), room_id, token_match.group(0))


@dataclass(frozen=True)
class RoomMessage:
    message_id: str
    sequence: int
    sender_id: str
    sender_name: str
    content: str
    reply_to: str | None = None

    @classmethod
    def from_json(cls, item: dict[str, Any]) -> "RoomMessage":
        sender = item.get("sender") or {}
        return cls(
            str(item["id"]),
            int(item["sequence"]),
            str(sender.get("member_id") or item.get("sender_instance_id") or ""),
            str(sender.get("name") or ""),
            str(item.get("content") or ""),
            item.get("reply_to_message_id"),
        )


Transport = Callable[[str, str, dict[str, str], dict[str, Any] | None, int], tuple[int, dict[str, Any]]]


def _urllib_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | None,
    timeout: int,
) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return response.status, json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        payload = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            parsed = {"error": {"code": "non_json_error", "message": payload[:200]}}
        return error.code, parsed


class RoomError(RuntimeError):
    pass


class RoomClient:
    def __init__(
        self,
        base_url: str,
        room_id: str,
        *,
        token: str | None = None,
        transport: Transport | None = None,
    ):
        self.base_url = _canonical_base(base_url)
        self.room_id = room_id
        self.token = token
        self.member_id = ""
        self.name = ""
        self._transport = transport or _urllib_transport

    def _call(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        token: str | None = None,
        timeout: int = 35,
    ) -> dict[str, Any]:
        auth = token or self.token
        if not auth:
            raise RoomError("SharedNet member has no authentication token")
        headers = {"authorization": f"Bearer {auth}"}
        if body is not None:
            headers["content-type"] = "application/json"
        if method == "POST" and path.endswith("/messages"):
            headers["idempotency-key"] = str(uuid.uuid4())
        status, payload = self._transport(method, f"{self.base_url}{path}", headers, body, timeout)
        if status >= 400:
            error = (payload or {}).get("error") or {}
            raise RoomError(f"SharedNet HTTP {status}: {error.get('code', 'unknown_error')}: {error.get('message', '')}")
        return payload

    def join(self, invite_token: str, name: str) -> list[RoomMessage]:
        # Matrix episode ids can exceed the API's display-name limit. Keep a
        # readable prefix and stable collision-resistant identity per role.
        if len(name) > 48:
            name = name[:35] + ":" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
        payload = self._call(
            "POST",
            f"/api/v1/rooms/{self.room_id}/join",
            {"name": name, "runtime": {"kind": "openhands"}},
            token=invite_token,
        )
        member_token = payload.get("member_token")
        if not isinstance(member_token, str):
            raise RoomError("SharedNet join response omitted member_token")
        self.token = member_token
        membership = payload.get("membership") or {}
        self.member_id = str(membership.get("member_id") or "")
        self.name = str(membership.get("name") or name)
        history_payload = payload.get("history") or {}
        history = [RoomMessage.from_json(item) for item in history_payload.get("items", [])]
        if history_payload.get("has_more"):
            history.extend(self.messages(after=history[-1].sequence if history else 0))
        return history

    def send(self, content: str, *, reply_to: str | None = None) -> RoomMessage:
        if len(content.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise ValueError(f"SharedNet message exceeds {MAX_MESSAGE_BYTES} bytes")
        body: dict[str, Any] = {"content": content}
        if reply_to:
            body["reply_to_message_id"] = reply_to
        payload = self._call("POST", f"/api/v1/rooms/{self.room_id}/messages", body)
        return RoomMessage.from_json(payload["message"])

    def messages(self, *, after: int = 0) -> list[RoomMessage]:
        collected: list[RoomMessage] = []
        cursor = after
        while True:
            query = urllib.parse.urlencode({"after": cursor, "limit": 100})
            payload = self._call("GET", f"/api/v1/rooms/{self.room_id}/messages?{query}")
            items = [RoomMessage.from_json(item) for item in payload.get("items", [])]
            collected.extend(items)
            if not items or not payload.get("has_more"):
                return collected
            cursor = items[-1].sequence


def _encode(text: str, event_type: str, fields: dict[str, Any]) -> str:
    envelope = json.dumps(
        {"type": event_type, **fields},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{text.rstrip()}\n\n{TRAILER}{envelope}"


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Keep a UTF-8-safe head and tail within an exact byte allowance."""
    if max_bytes <= 0:
        return ""
    raw = (text or "").encode("utf-8")
    if len(raw) <= max_bytes:
        return text or ""
    marker = "\n\n… content elided …\n\n".encode("utf-8")
    if len(marker) >= max_bytes:
        return raw[:max_bytes].decode("utf-8", errors="ignore")
    available = max_bytes - len(marker)
    head_bytes = (available * 3) // 4
    tail_bytes = available - head_bytes
    head = raw[:head_bytes].decode("utf-8", errors="ignore")
    tail = raw[-tail_bytes:].decode("utf-8", errors="ignore") if tail_bytes else ""
    value = f"{head}{marker.decode('utf-8')}{tail}"
    # Dropping an incomplete boundary byte can leave a small amount of spare
    # capacity, but must never make the result exceed the caller's allowance.
    while len(value.encode("utf-8")) > max_bytes:
        value = value[:-1]
    return value


def _compact_fields(fields: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    compact = dict(fields)
    changed = False
    for key, value in tuple(compact.items()):
        if isinstance(value, str) and len(value.encode("utf-8")) > MAX_FIELD_BYTES:
            compact[key] = _truncate_utf8(value, MAX_FIELD_BYTES)
            changed = True
    return compact, changed


def _encode_limited(text: str, event_type: str, fields: dict[str, Any]) -> str:
    """Encode one typed event without exceeding SharedNet's byte limit.

    Full host output remains in the episode workspace and coordination ledger;
    only the Room hand-off is compacted.  The digest lets operators correlate a
    compact message with the complete local record.
    """
    original = (text or "").strip() or "(no output)"
    encoded = _encode(original, event_type, fields)
    if len(encoded.encode("utf-8")) <= MAX_MESSAGE_BYTES:
        return encoded

    compact_fields, fields_changed = _compact_fields(fields)
    compact_fields.update(
        {
            "truncated": True,
            "fields_truncated": fields_changed,
            "original_text_bytes": len(original.encode("utf-8")),
            "text_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
        }
    )
    overhead = len(_encode("", event_type, compact_fields).encode("utf-8"))
    allowance = max(0, MAX_MESSAGE_BYTES - overhead)
    compact_text = _truncate_utf8(original, allowance)
    encoded = _encode(compact_text, event_type, compact_fields)
    if len(encoded.encode("utf-8")) <= MAX_MESSAGE_BYTES:
        return encoded

    # This should only be reachable if future fields contain large non-string
    # values.  Preserve the routing identity and digest in a minimal envelope.
    minimal = {
        key: compact_fields[key]
        for key in ("episode_id", "hop", "role", "to", "proposed_next", "status")
        if key in compact_fields
    }
    minimal.update(
        {
            "truncated": True,
            "fields_truncated": fields_changed or len(minimal) != len(fields),
            "original_text_bytes": len(original.encode("utf-8")),
            "text_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
        }
    )
    overhead = len(_encode("", event_type, minimal).encode("utf-8"))
    encoded = _encode(_truncate_utf8(original, max(0, MAX_MESSAGE_BYTES - overhead)), event_type, minimal)
    if len(encoded.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ValueError("typed SharedNet envelope cannot fit within the message limit")
    return encoded


def _decode(content: str) -> tuple[str, dict[str, Any] | None]:
    lines = content.rstrip().splitlines()
    if not lines or not lines[-1].startswith(TRAILER):
        return content.strip(), None
    try:
        envelope = json.loads(lines[-1][len(TRAILER) :])
    except json.JSONDecodeError:
        return content.strip(), None
    return "\n".join(lines[:-1]).strip(), envelope if isinstance(envelope, dict) else None


def _summarize(output: str) -> str:
    return _truncate_utf8((output or "").strip() or "(no output)", RESULT_SUMMARY_BYTES)


def _is_message_size_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in ("exceed", "too large", "payload too large", "http 413"))


class SharedNetSession:
    """One episode's role members and typed hand-offs in a caller-selected Room."""

    def __init__(
        self,
        invite: SharedNetInvite,
        episode_id: str,
        roles: tuple[str, ...],
        *,
        client_factory: Callable[[str, str], RoomClient] | None = None,
    ):
        self.invite = invite
        self.episode_id = episode_id
        self.roles = roles
        self._factory = client_factory or (lambda base, room: RoomClient(base, room))
        self.coordinator: RoomClient | None = None
        self.members: dict[str, RoomClient] = {}
        self.cursor = 0
        self.previous_result = ""
        self.previous_role = ""
        self.previous_disposition = ""
        self._requests: dict[int, str] = {}
        self.publish_warnings: list[str] = []

    @property
    def room_id(self) -> str:
        return self.invite.room_id

    def join(self) -> None:
        coordinator = self._factory(self.invite.base_url, self.invite.room_id)
        history = coordinator.join(self.invite.token, f"rac:{self.episode_id}")
        self.coordinator = coordinator
        for role in self.roles:
            member = self._factory(self.invite.base_url, self.invite.room_id)
            member.join(self.invite.token, f"{self.episode_id}:{role}")
            self.members[role] = member
        self._consume(history)

    def _consume(self, messages: list[RoomMessage]) -> list[str]:
        guidance: list[str] = []
        member_ids = {member.member_id for member in self.members.values()}
        if self.coordinator:
            member_ids.add(self.coordinator.member_id)
        for message in messages:
            self.cursor = max(self.cursor, message.sequence)
            text, envelope = _decode(message.content)
            if envelope and envelope.get("episode_id") == self.episode_id:
                if envelope.get("type") == "work.result":
                    self.previous_result = text
                    self.previous_role = str(envelope.get("role") or message.sender_name)
                continue
            if envelope is None and message.sender_id not in member_ids and text:
                guidance.append(f"{message.sender_name or message.sender_id}: {text}")
        return guidance

    def _send_typed(
        self,
        client: RoomClient,
        text: str,
        event_type: str,
        fields: dict[str, Any],
        *,
        reply_to: str | None = None,
    ) -> RoomMessage | None:
        """Publish a bounded event, degrading only message-size failures.

        Authentication, connectivity, and other Room errors remain fatal.  A
        size rejection is different: host work has already completed and is
        durably recorded locally, so a compact coordination fallback must not
        turn that completed work into an episode failure.
        """
        try:
            return client.send(
                _encode_limited(text, event_type, fields),
                reply_to=reply_to,
            )
        except Exception as error:
            if not _is_message_size_error(error):
                raise

        minimal = {
            key: value
            for key, value in fields.items()
            if key in {"episode_id", "hop", "role", "to", "proposed_next", "status"}
        }
        minimal.update(
            {
                "truncated": True,
                "publish_fallback": True,
                "original_text_bytes": len((text or "").encode("utf-8")),
                "text_sha256": hashlib.sha256((text or "").encode("utf-8")).hexdigest(),
            }
        )
        try:
            return client.send(
                _encode_limited(
                    "Full coordination text was retained in the local episode ledger; "
                    "this Room event is a size-safe fallback.",
                    event_type,
                    minimal,
                ),
                reply_to=reply_to,
            )
        except Exception as error:
            if not _is_message_size_error(error):
                raise
            warning = f"SharedNet {event_type} publication omitted after two size rejections: {error}"
            self.publish_warnings.append(warning)
            warnings.warn(warning, RuntimeWarning, stacklevel=2)
            return None

    def request(self, role: str, hop: int, prompt: str, contract_id: str | None) -> str:
        if self.coordinator is None:
            raise RuntimeError("SharedNet session has not joined its Room")
        guidance = self._consume(self.coordinator.messages(after=self.cursor))
        fields: dict[str, Any] = {
            "episode_id": self.episode_id,
            "hop": hop,
            "to": role,
        }
        if contract_id:
            fields["contract_id"] = contract_id
        request = self._send_typed(
            self.coordinator,
            f"@{role} hop {hop}\n\n{prompt}",
            "work.request",
            fields,
        )
        if request is not None:
            self._requests[hop] = request.message_id
        context: list[str] = []
        if self.previous_result:
            context.append(f"Previous team member ({self.previous_role}) reported:\n{self.previous_result}")
        if self.previous_disposition:
            context.append(f"RAC disposition of that work:\n{self.previous_disposition}")
        if guidance:
            context.append("Room guidance since the previous hand-off:\n- " + "\n- ".join(guidance))
        if not context:
            return prompt
        return f"{prompt}\n\n## SharedNet team communication\n\n" + "\n\n".join(context)

    def result(self, role: str, hop: int, output: str, proposed_next: str | None, *, error: str | None = None) -> None:
        member = self.members[role]
        fields = {
            "episode_id": self.episode_id,
            "hop": hop,
            "role": role,
            "proposed_next": proposed_next,
            "error": error,
            "status": "awaiting_verification",
        }
        summary = _summarize(output)
        self._send_typed(
            member,
            output,
            "work.result",
            fields,
            reply_to=self._requests.get(hop),
        )
        # Sending must not advance the read cursor: guidance can arrive while an
        # agent is working, before its result message receives a later sequence.
        self.previous_result = summary
        self.previous_role = role

    def disposition(self, hop: int, *, accepted: bool, reason: str, next_role: str | None) -> None:
        if self.coordinator is None:
            return
        event = "work.accepted" if accepted else "work.rejected"
        compact_reason = _truncate_utf8(reason, MAX_FIELD_BYTES)
        self.previous_disposition = f"{'accepted' if accepted else 'rejected'}: {compact_reason}"
        self._send_typed(
            self.coordinator,
            f"Hop {hop} {'accepted' if accepted else 'rejected'}: {compact_reason}",
            event,
            {
                "episode_id": self.episode_id,
                "hop": hop,
                "next": next_role,
                "reason": compact_reason,
            },
        )

    def verification(self, hop: int, *, verdict: str, reason: str, next_role: str | None) -> None:
        """Publish a non-blocking verifier result for the next selected agent."""
        if self.coordinator is None:
            return
        compact_reason = _truncate_utf8(reason, MAX_FIELD_BYTES)
        self.previous_disposition = f"verification advisory ({verdict}): {compact_reason}"
        self._send_typed(
            self.coordinator,
            f"Hop {hop} verification advisory ({verdict}): {compact_reason}",
            "work.verification",
            {
                "episode_id": self.episode_id,
                "hop": hop,
                "next": next_role,
                "verdict": verdict,
                "advisory": True,
                "reason": compact_reason,
            },
        )
