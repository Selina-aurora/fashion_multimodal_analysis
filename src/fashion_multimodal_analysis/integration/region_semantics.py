"""Query routing and phrase checks for diagnostic region grounding.

These guards reject known wrong object phrases. They do not establish that a
returned region is visually correct or that a missed region is absent.
"""
import re

REGION_PROMPTS = {
    "collar": "collar. neckline.",
    "cuff": "sleeve cuff.",
    "hem": "hem. bottom hem.",
    "pocket": "pocket.",
    "shoulder": "shoulder.",
    "waist": "waist. waistband.",
    "pattern": "printed pattern. stripe. floral pattern.",
    "decoration": "decorative detail. button. bow. ornament.",
}
REGION_WORDS = {
    "collar": {"collar", "neckline"},
    "cuff": {"cuff"},
    "hem": {"hem"},
    "pocket": {"pocket"},
    "shoulder": {"shoulder"},
    "waist": {"waist", "waistband"},
    "pattern": {"pattern", "print", "printed", "stripe", "striped", "floral"},
    "decoration": {"decorative", "decoration", "button", "bow", "ornament"},
}
UPPER = {"top", "outerwear", "dress"}
CLOTHING = UPPER | {"pants", "skirt"}


def supports_region(category, region):
    """Route structural clothing queries; pattern/detail queries remain universal."""
    if region in {"collar", "cuff", "shoulder"}:
        return category in UPPER
    if region in {"hem", "pocket", "waist"}:
        return category in CLOTHING
    return region in {"pattern", "decoration"}


def phrase_matches(region, label):
    """Require a target word instead of accepting a generic 'garment' phrase."""
    tokens = set(re.findall(r"[a-z]+", str(label).lower()))
    tokens |= {token[:-1] for token in list(tokens) if token.endswith("s")}
    return bool(tokens & REGION_WORDS.get(region, set()))
