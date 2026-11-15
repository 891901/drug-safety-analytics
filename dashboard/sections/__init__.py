"""Dashboard sections.

Each module exposes a single ``render()`` function that draws one dashboard
section from the project's real analytical outputs.  ``app.py`` imports them and
dispatches on the sidebar selection.
"""

from __future__ import annotations

from dashboard.sections import (
    cooccurrence,
    demographic,
    drugs,
    executive,
    investigation,
    machine_learning,
    network,
    reactions,
    signals,
    temporal,
)

#: Ordered (label, render callable) pairs that drive the sidebar navigation.
SECTION_MODULES = [
    ("Executive Overview", executive.render),
    ("Drug Analytics", drugs.render),
    ("Adverse Reaction Analysis", reactions.render),
    ("Drug Co-Occurrence", cooccurrence.render),
    ("Drug Network", network.render),
    ("Statistical Signal Detection", signals.render),
    ("Machine Learning", machine_learning.render),
    ("Temporal Analysis", temporal.render),
    ("Demographic Analysis", demographic.render),
    ("Drug Investigation", investigation.render),
]

__all__ = ["SECTION_MODULES"]
