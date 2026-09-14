import unittest

from rac_ai_scientist.bridge import BridgeContractError, validate_bridge_checkpoint
from rac_ai_scientist.schemas import Budget, CapabilityCard, Checkpoint


class BridgeTests(unittest.TestCase):
    def test_duplicate_capability_ids_are_rejected(self):
        card = CapabilityCard("same", "x", (), (), ())
        checkpoint = Checkpoint("ep", 0, "goal", "stage", [], [], Budget(1, 1, 1, 1, 1, 1), [card, card])
        with self.assertRaises(BridgeContractError):
            validate_bridge_checkpoint(checkpoint)
