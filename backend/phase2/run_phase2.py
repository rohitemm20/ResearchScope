"""
CLI:  (from backend/)   python -m phase2.run_phase2 output/<job_id>.json
                        python -m phase2.run_phase2 output/<job_id>.json --query "Nash equilibrium"
"""
import argparse
import os

from .pipeline import run_pipeline, semantic_search
from .text_cleaning import load_parsed_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parsed_json")
    ap.add_argument("--quality", choices=["fast", "best"], default="fast")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--query", help="run a semantic search after processing")
    a = ap.parse_args()

    parsed = load_parsed_json(a.parsed_json)
    job_id = parsed.get("job_id") or os.path.splitext(os.path.basename(a.parsed_json))[0]
    out_dir = os.path.dirname(os.path.abspath(a.parsed_json))

    r = run_pipeline(parsed, job_id, out_dir, a.quality, a.threshold)
    print("STATS   :", r["stats"])
    print("TIMINGS :", r["timings_sec"])
    print("\nSUMMARY :\n", r["summary"])
    print("\nKEYWORDS:", ", ".join(k["keyword"] for k in r["keywords"]))
    if a.query:
        print(f"\nSEARCH '{a.query}':")
        for h in semantic_search(job_id, out_dir, a.query):
            print(f"  {h['score']:.3f}  p{h['page']} [{h['section']}]  {h['text'][:110]}")


if __name__ == "__main__":
    main()
