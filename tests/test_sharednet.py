import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rac_ai_scientist.bridge import HostBridge
from rac_ai_scientist.schemas import InvocationResult
from rac_ai_scientist.sharednet import RoomMessage, SharedNetInvite, SharedNetSession, _decode, load_sharednet_env


class FakeRoomClient:
    sequence = 0
    messages_log = []

    def __init__(self, base_url, room_id):
        self.base_url = base_url
        self.room_id = room_id
        self.member_id = ""
        self.name = ""

    @classmethod
    def reset(cls):
        cls.sequence = 0
        cls.messages_log = []

    def join(self, invite_token, name):
        self.member_id = f"member-{name}"
        self.name = name
        return list(self.messages_log)

    def send(self, content, *, reply_to=None):
        type(self).sequence += 1
        message = RoomMessage(
            f"message-{self.sequence}",
            self.sequence,
            self.member_id,
            self.name,
            content,
            reply_to,
        )
        type(self).messages_log.append(message)
        return message

    def messages(self, *, after=0):
        return [message for message in self.messages_log if message.sequence > after]

    @classmethod
    def external_message(cls, content, *, name="operator"):
        cls.sequence += 1
        cls.messages_log.append(
            RoomMessage(
                f"message-{cls.sequence}",
                cls.sequence,
                f"external-{name}",
                name,
                content,
            )
        )


class SharedNetTests(unittest.TestCase):
    def setUp(self):
        FakeRoomClient.reset()

    def test_explicit_room_must_match_invite(self):
        with self.assertRaises(ValueError):
            SharedNetInvite.parse(
                f"ROOM=rom_one TOKEN=rit_{'a' * 43}",
                room_id="rom_two",
            )

    def test_per_run_dotenv_loads_only_sharednet_settings(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / ".env"
            path.write_text(
                "# private per-run communication config\n"
                "SHAREDNET_ROOM_ID=rom_run3\n"
                f"SHAREDNET_INVITE='ROOM=rom_run3 TOKEN=rit_{'a' * 43}'\n"
                "SHAREDNET_BASE_URL=https://sharednet.ai # canonicalized later\n"
                "AGENT_API_KEY=must-not-be-imported\n",
                encoding="utf-8",
            )

            settings = load_sharednet_env(path)

        self.assertEqual(settings["SHAREDNET_ROOM_ID"], "rom_run3")
        self.assertIn("TOKEN=rit_", settings["SHAREDNET_INVITE"])
        self.assertEqual(settings["SHAREDNET_BASE_URL"], "https://sharednet.ai")
        self.assertNotIn("AGENT_API_KEY", settings)

    def test_roles_exchange_typed_results_and_context(self):
        invite = SharedNetInvite.parse(
            f"ROOM=rom_run1 TOKEN=rit_{'a' * 43}",
            room_id="rom_run1",
        )
        session = SharedNetSession(
            invite,
            "episode-1",
            ("researcher", "experimenter"),
            client_factory=FakeRoomClient,
        )
        session.join()

        first_prompt = session.request("researcher", 0, "frame the study", None)
        self.assertEqual(first_prompt, "frame the study")
        session.result("researcher", 0, "research plan ready", "experimenter")
        session.disposition(0, accepted=True, reason="accepted", next_role="experimenter")
        second_prompt = session.request("experimenter", 1, "run experiments", "contract-1")

        self.assertIn("research plan ready", second_prompt)
        self.assertIn("accepted: accepted", second_prompt)
        typed = [_decode(message.content)[1] for message in FakeRoomClient.messages_log]
        event_types = [item["type"] for item in typed if item]
        self.assertEqual(
            event_types,
            ["work.request", "work.result", "work.accepted", "work.request"],
        )
        self.assertTrue(all(item.get("episode_id") == "episode-1" for item in typed if item))

    def test_guidance_sent_while_agent_runs_is_not_skipped(self):
        invite = SharedNetInvite.parse(
            f"ROOM=rom_run2 TOKEN=rit_{'a' * 43}",
            room_id="rom_run2",
        )
        session = SharedNetSession(
            invite,
            "episode-2",
            ("researcher", "experimenter"),
            client_factory=FakeRoomClient,
        )
        session.join()

        session.request("researcher", 0, "frame the study", None)
        FakeRoomClient.external_message("also test the low-noise subset")
        session.result("researcher", 0, "research plan ready", "experimenter")
        next_prompt = session.request("experimenter", 1, "run experiments", None)

        self.assertIn("also test the low-noise subset", next_prompt)

    def test_verification_is_advisory_context_for_next_agent(self):
        invite = SharedNetInvite.parse(
            f"ROOM=rom_advisory TOKEN=rit_{'a' * 43}",
            room_id="rom_advisory",
        )
        session = SharedNetSession(
            invite,
            "episode-advisory",
            ("plan", "write"),
            client_factory=FakeRoomClient,
        )
        session.join()
        session.request("plan", 0, "make a plan", "contract-0")
        session.result("plan", 0, "plan attempted", "write")
        session.verification(
            0,
            verdict="refuted",
            reason="expected artifact did not change",
            next_role="write",
        )

        prompt = session.request("write", 1, "write report", "contract-1")
        self.assertIn("verification advisory (refuted)", prompt)
        self.assertIn("expected artifact did not change", prompt)
        typed = [_decode(message.content)[1] for message in FakeRoomClient.messages_log]
        self.assertIn("work.verification", [item["type"] for item in typed if item])

    def test_common_bridge_lifecycle_joins_roles_and_publishes_result(self):
        class ProbeBridge(HostBridge):
            host_id = "probe"

            def __init__(self):
                self.episode_id = "episode-probe"
                self.hop = 1
                self.cards = [SimpleNamespace(capability_id="plan"), SimpleNamespace(capability_id="write")]

            def initialize(self, **kwargs):
                pass

            def checkpoint(self):
                return SimpleNamespace()

            def native_next(self, checkpoint):
                return "write"

            def invoke(self, capability_id, contract):
                raise NotImplementedError

        bridge = ProbeBridge()
        bridge.configure_condition("R1")
        invite = f"ROOM=rom_probe TOKEN=rit_{'a' * 43}"
        with patch.dict(os.environ, {
            "SHAREDNET_ROOM_ID": "rom_probe",
            "SHAREDNET_INVITE": invite,
        }, clear=False), patch(
            "rac_ai_scientist.bridge.SharedNetSession",
            side_effect=lambda parsed, episode, roles: SharedNetSession(
                parsed, episode, roles, client_factory=FakeRoomClient
            ),
        ):
            bridge.initialize_communication()

        self.assertEqual(tuple(bridge.sharednet.members), ("plan", "write"))
        prompt = bridge.communication_prompt("plan", "make a plan", None)
        self.assertEqual(prompt, "make a plan")
        result = InvocationResult("plan", "plan ready", [], [])
        bridge.publish_invocation(result)
        self.assertEqual(result.proposed_next, "write")
        bridge.accept_invocation(result, SimpleNamespace(reason="verified"))
        event_types = [
            envelope["type"]
            for message in FakeRoomClient.messages_log
            if (envelope := _decode(message.content)[1]) is not None
        ]
        self.assertEqual(event_types, ["work.request", "work.result", "work.accepted"])


if __name__ == "__main__":
    unittest.main()
