"""Host adapter classes; upstream dependencies are imported only at initialization."""

from .ark import ArkBridge
from .agent_laboratory import AgentLaboratoryBridge
from .data_to_paper import DataToPaperBridge
from .ai_researcher import AIResearcherBridge
from .evo_scientist import EvoScientistBridge
from .auto_research_claw import AutoResearchClawBridge

__all__ = [
    "ArkBridge",
    "AgentLaboratoryBridge",
    "DataToPaperBridge",
    "AIResearcherBridge",
    "EvoScientistBridge",
    "AutoResearchClawBridge",
]
