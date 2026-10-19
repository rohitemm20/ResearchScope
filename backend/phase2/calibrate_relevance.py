"""
calibrate_relevance.py
----------------------

Test how the relevance score behaves using the REAL ResearchScope
embedding model and an already processed Phase 2 paper.

Usage from backend:

python -m phase2.calibrate_relevance JOB_ID "topic 1" "topic 2" "topic 3"

Example:

python -m phase2.calibrate_relevance d1ddd18cb5d3 "game theory" "Nash equilibrium" "machine learning" "protein folding"
"""

import os
import sys

from .relevance import check_relevance


# ----------------------------------------------------------------------
# Output directory
# ----------------------------------------------------------------------

OUT = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "output",
)


# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)


job_id = sys.argv[1]
topics = sys.argv[2:]


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

print()
print("=" * 90)
print("ResearchScope Relevance Calibration")
print("=" * 90)
print(f"Job ID: {job_id}")
print()


print(
    f"{'Topic':32}"
    f"{'Score':>8}"
    f"{'Raw':>10}"
    f"{'Sentences':>12}"
    f"{'Summary':>10}"
    f"{'Keywords':>10}"
)

print("-" * 90)


# ----------------------------------------------------------------------
# Score each topic
# ----------------------------------------------------------------------

for topic in topics:

    result = check_relevance(
        job_id,
        OUT,
        topic,
        [],
    )

    components = result["domain"]["components"]

    print(
        f"{topic[:32]:32}"
        f"{result['score']:>8}"
        f"{result['domain']['raw_similarity']:>10.4f}"
        f"{components['sentences']:>12.4f}"
        f"{components['summary']:>10.4f}"
        f"{components['keywords']:>10.4f}"
    )


print("-" * 90)
print()
print("Suggested interpretation:")
print("  70-100  = Strong candidate relevance")
print("  45-69   = Partial / moderate relevance")
print("  25-44   = Weak relevance")
print("   0-24   = Not relevant")
print()