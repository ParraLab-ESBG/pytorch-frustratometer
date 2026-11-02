"""
WARNING: This has not been run or tested.

Generate MolViewSpec (.mvsj) files for frustration visualization in Mol*.

Tube endpoints are referenced as CA atoms by (chain, residue) selector;
Mol* resolves atom coordinates from the structure loaded by the mvsj.

Format reference: MolViewSpec 1.x (https://github.com/molstar/mol-view-spec)
Viewer: Mol* 5.0+ is required for the 'tube' primitive type.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Sequence


_RED  = "#FF4040"
_GREEN = "#40C040"
_GREY  = "#808080"


def _ca_ref(chain: str, residue: int) -> dict:
    return {"label_asym_id": str(chain), "label_seq_id": int(residue), "label_atom_id": "CA"}


def write_contact_mvsj(
    contacts: Sequence[dict],
    struct_url: str,
    struct_format: str = "pdb",
    title: str = "",
    indent: int | None = None,
) -> str:
    """
    Build a MolViewSpec JSON string for pairwise contact frustration.

    Each element of *contacts* must have:
      chain1, residue1, chain2, residue2  (str / int)
      frustration  : "highly" | "minimally"   (neutral contacts are omitted upstream)
      is_water     : bool
    """
    tube_nodes = []
    for c in contacts:
        color  = _RED if c["frustration"] == "highly" else _GREEN
        params = {
            "start":  _ca_ref(c["chain1"], c["residue1"]),
            "end":    _ca_ref(c["chain2"], c["residue2"]),
            "radius": 0.15,
            "color":  color,
        }
        if c["is_water"]:
            params["dash_length"] = 0.25
        tube_nodes.append({"kind": "tube", "params": params})

    return json.dumps({
        "metadata": {
            "version":   "1",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "title":     title,
        },
        "root": {
            "kind": "root",
            "children": [
                {
                    "kind": "download",
                    "params": {"url": struct_url},
                    "children": [{
                        "kind": "parse",
                        "params": {"format": struct_format},
                        "children": [{
                            "kind": "model_structure",
                            "children": [{
                                "kind": "component",
                                "params": {"selector": "all"},
                                "children": [{
                                    "kind": "representation",
                                    "params": {
                                        "type": "cartoon",
                                        "color_theme": "uniform",
                                        "color": _GREY,
                                    },
                                }],
                            }],
                        }],
                    }],
                },
                {"kind": "primitives", "children": tube_nodes},
            ],
        },
    }, indent=indent)


def write_singleresidue_mvsj(
    struct_url: str,
    struct_format: str = "pdb",
    title: str = "",
    indent: int | None = None,
) -> str:
    """
    Build a MolViewSpec JSON string for single-residue frustration.

    The structure file must have frustration indices stored in the B-factor column.
    Mol* colors the cartoon by the uncertainty/B-factor field.
    """
    return json.dumps({
        "metadata": {
            "version":   "1",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "title":     title,
        },
        "root": {
            "kind": "root",
            "children": [{
                "kind": "download",
                "params": {"url": struct_url},
                "children": [{
                    "kind": "parse",
                    "params": {"format": struct_format},
                    "children": [{
                        "kind": "model_structure",
                        "children": [{
                            "kind": "component",
                            "params": {"selector": "all"},
                            "children": [{
                                "kind": "representation",
                                "params": {"type": "cartoon", "color_theme": "uncertainty"},
                            }],
                        }],
                    }],
                }],
            }],
        },
    }, indent=indent)
