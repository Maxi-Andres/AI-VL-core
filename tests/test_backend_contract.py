"""The request contract this service shares with the backend gateway — the iacore half.

Every test here names the defect it catches. Nothing runs a server, opens a socket, loads a
model, or imports `service.py`: the models are read out of the source with `ast`, so this
suite needs no Ollama, no Ultralytics, no GPU and no robot. It runs in milliseconds.

WHY THIS EXISTS. Four request models are declared **twice** — here in `service.py` and again
in `AI-VL-backend/app.py` — and the tier boundary forbids sharing a module, so the
duplication is deliberate and permanent. What was missing is the tripwire. Neither side sets
`model_config`, so pydantic v2 defaults to `extra="ignore"`: a field the gateway does not
declare is **dropped silently on the way here, with no error anywhere**. Add `seq` to
`CommandRequest` below to fix the command/stop race, forget the gateway, and the sequence
number never arrives — the race stays open and nothing tells you.

This is the mirror of `AI-VL-backend/tests/test_iacore_contract.py`. `CONTRACT` and the AST
helper are byte-identical copies on purpose: importing them across the boundary would be the
very violation these tests exist to make unnecessary.

HOW TO CHANGE THE CONTRACT. Editing a shared model is a **two-repo change**:
  1. add the field on both sides,
  2. update `CONTRACT` below **and** the identical copy in
     `AI-VL-backend/tests/test_iacore_contract.py`.
Do 1 without 2 and both suites go red; do 2 without 1 and both suites go red.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# The contract: model -> field names, IN ORDER. Keep byte-identical with the sibling copy.
#
# Types and defaults are deliberately NOT part of it. This service declares
# `scope: str = CFG.get("scope", ...)` because it owns the defaults; the gateway declares
# `scope: str | None = None` so it forwards exactly what the client sent and never invents a
# value. Asserting on types would fight that design instead of protecting it.
# --------------------------------------------------------------------------- #
CONTRACT = {
    "VlmRequest": ["image", "model", "scope", "variant", "max_tokens", "num_ctx", "think",
                   "prompt"],
    "VlmStreamRequest": ["image", "prompt", "model", "max_tokens", "num_ctx"],
    "SpeakRequest": ["text", "voice"],
    "CommandRequest": ["text", "image", "model", "robot", "num_ctx", "max_tokens"],
}

THIS_REPO = Path(__file__).resolve().parent.parent
OWN_SOURCE = THIS_REPO / "service.py"
# The sibling is cloned next to this repo in the ecosystem layout and git-ignored by the
# umbrella. A CI job that checks out only this repo will not have it — hence the skip.
SIBLING_SOURCE = THIS_REPO.parent / "AI-VL-backend" / "app.py"

MODELS = sorted(CONTRACT)


def _basemodel_fields(source: Path) -> dict[str, list[str]]:
    """{class name: [annotated field names, in source order]} for every BaseModel in a file."""
    tree = ast.parse(source.read_text(), filename=str(source))
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and any(
            isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases
        ):
            found[node.name] = [
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            ]
    return found


@pytest.fixture(scope="module")
def own() -> dict[str, list[str]]:
    return _basemodel_fields(OWN_SOURCE)


@pytest.fixture(scope="module")
def sibling() -> dict[str, list[str]]:
    if not SIBLING_SOURCE.exists():
        pytest.skip(
            f"{SIBLING_SOURCE} not checked out — the cross-repo half of this contract test "
            "only runs in the ecosystem layout. The committed CONTRACT above still guards "
            "this repo on its own, which is what protects CI."
        )
    return _basemodel_fields(SIBLING_SOURCE)


# --------------------------------------------------------------------------- #
# This repo against the contract — always runs, CI included
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", MODELS)
def test_this_services_model_matches_the_committed_contract(own, name):
    """The defect: adding or renaming a field here and forgetting the gateway, or vice versa."""
    assert name in own, f"{name} disappeared from service.py — the contract expects it"
    assert own[name] == CONTRACT[name], (
        f"service.py's {name} drifted from the contract.\n"
        f"  contract  : {CONTRACT[name]}\n"
        f"  service.py: {own[name]}\n"
        "If the change is intended, update CONTRACT here AND in "
        "AI-VL-backend/tests/test_iacore_contract.py."
    )


def test_no_shared_model_is_empty(own):
    """A model that declares no fields drops every input, because `extra="ignore"` is the
    pydantic v2 default and neither tier overrides it."""
    for name in MODELS:
        assert own[name], f"{name} declares no fields at all; it would drop every input"


# --------------------------------------------------------------------------- #
# The sibling against the same contract — the cross-repo half
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", MODELS)
def test_the_gateways_model_matches_the_committed_contract(sibling, name):
    assert name in sibling, f"{name} disappeared from AI-VL-backend/app.py"
    assert sibling[name] == CONTRACT[name], (
        f"app.py's {name} drifted from the contract.\n"
        f"  contract: {CONTRACT[name]}\n"
        f"  app.py  : {sibling[name]}\n"
        "A field added on one tier and not the other is dropped in transit."
    )


def test_the_contract_covers_every_model_the_two_tiers_share(own, sibling):
    """The defect: a NEW shared model appearing on both sides, guarded by nothing."""
    shared = set(own) & set(sibling)
    assert shared == set(CONTRACT), (
        "the two tiers share models that the contract does not list.\n"
        f"  shared but unguarded: {sorted(shared - set(CONTRACT))}\n"
        f"  listed but no longer shared: {sorted(set(CONTRACT) - shared)}"
    )
