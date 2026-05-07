"""
retrieval/synonyms.py

Long COVID synonym expansion for sparse (keyword) search.
Applied only to the sparse path to broaden BM25-style term coverage
without polluting the dense semantic path.
"""

from __future__ import annotations

import re

# Maps a canonical term (lowercase) to a list of known aliases.
# All entries are lowercase; matching is case-insensitive.
_SYNONYM_MAP: dict[str, list[str]] = {
    "long covid": [
        "PASC", "post-acute sequelae", "post-COVID syndrome",
        "post-COVID-19 condition", "long-haul COVID", "post-acute COVID-19",
        "chronic COVID", "long hauler",
    ],
    "pasc": [
        "long COVID", "post-acute sequelae of SARS-CoV-2",
        "post-COVID syndrome", "post-acute COVID-19",
    ],
    "brain fog": [
        "cognitive impairment", "cognitive dysfunction",
        "neurocognitive symptoms", "mental fog", "cognitive fatigue",
    ],
    "fatigue": [
        "post-exertional malaise", "PEM", "chronic fatigue",
        "myalgic encephalomyelitis", "exhaustion",
    ],
    "dyspnea": ["shortness of breath", "breathlessness", "respiratory distress"],
    "dysautonomia": [
        "autonomic dysfunction", "autonomic nervous system disorder",
        "POTS", "postural orthostatic tachycardia syndrome",
    ],
    "mast cell activation": [
        "MCAS", "mast cell activation syndrome", "mast cell disorder",
    ],
    "spike protein": ["S protein", "SARS-CoV-2 spike", "spike glycoprotein"],
    "microbiome": ["gut microbiota", "intestinal microbiome", "gut flora", "dysbiosis"],
    "neurological": [
        "neurologic", "neuro", "neuropathology", "neuroinflammation",
    ],
    "inflammation": [
        "inflammatory response", "cytokine storm", "cytokine release",
        "immune dysregulation", "hyperinflammation",
    ],
    "treatment": ["therapy", "intervention", "management", "therapeutic"],
    "vaccine": ["vaccination", "immunization", "mRNA vaccine", "COVID vaccine"],
}

# Pre-compile for efficiency: {lowercase_key: compiled_pattern}
_PATTERNS: dict[str, re.Pattern] = {
    key: re.compile(r'\b' + re.escape(key) + r'\b', re.IGNORECASE)
    for key in _SYNONYM_MAP
}


def expand_query(query: str) -> str:
    """
    Append synonym terms to query for any matched Long COVID concepts.
    Returns the expanded query string (original + appended aliases).
    Only affects the sparse/keyword search path.
    """
    if not query:
        return query

    appended: list[str] = []
    lowered = query.lower()

    for key, pattern in _PATTERNS.items():
        if pattern.search(lowered):
            aliases = [a for a in _SYNONYM_MAP[key] if a.lower() not in lowered]
            if aliases:
                appended.extend(aliases)

    if not appended:
        return query

    return query + " " + " ".join(appended)
