"""Projection targets -- serialize rules to output formats."""

from qortex.projectors.targets._serialize import serialize_ruleset
from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget
from qortex.projectors.targets.claude_code_skill import ClaudeCodeSkillTarget
from qortex.projectors.targets.flat_json import FlatJSONTarget
from qortex.projectors.targets.flat_yaml import FlatYAMLTarget
from qortex.projectors.targets.openclaw_skill import OpenClawSkillTarget
from qortex.projectors.targets.skillipedia import SkillipediaTarget

__all__ = [
    "BuildlogSeedTarget",
    "ClaudeCodeSkillTarget",
    "FlatJSONTarget",
    "FlatYAMLTarget",
    "OpenClawSkillTarget",
    "SkillipediaTarget",
    "serialize_ruleset",
]
