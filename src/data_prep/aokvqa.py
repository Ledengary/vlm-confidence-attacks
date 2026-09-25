from __future__ import annotations
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_IDS = os.path.join(REPO_ROOT, "artifacts", "long_answer_item_ids.json")
DEFAULT_DATASET = "HuggingFaceM4/A-OKVQA"


def main():
    ap = argparse.ArgumentParser(
        description="Download A-OKVQA (Schwenk et al., ECCV 2022) and build the frozen long-answer "
                    "item pool used by Table 13. Reconstructs the pool from the shipped id list; it "
                    "does not resample.")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="HuggingFace dataset id for A-OKVQA (images plus annotations).")
    ap.add_argument("--ids", default=DEFAULT_IDS,
                    help="JSON list of the frozen aokvqa_<question_id> ids to keep.")
    ap.add_argument("--out-dir", required=True,
                    help="Target long_answer directory; writes aokvqa_items.json and images/.")
    ap.add_argument("--splits", nargs="+", default=["validation", "train"])
    args = ap.parse_args()
    from datasets import load_dataset
    keep = set(json.load(open(args.ids)))
    want = {k[len("aokvqa_"):]: k for k in keep}
    imgdir = os.path.join(args.out_dir, "images")
    os.makedirs(imgdir, exist_ok=True)
    items = {}
    for split in args.splits:
        ds = load_dataset(args.dataset, split=split)
        for r in ds:
            qid = r["question_id"]
            if qid not in want or want[qid] in items:
                continue
            iid = want[qid]
            r["image"].convert("RGB").save(os.path.join(imgdir, f"{iid}.png"), format="PNG")
            choices = list(r["choices"])
            items[iid] = {
                "item_id": iid,
                "question": r["question"],
                "choices": choices,
                "gold": choices[int(r["correct_choice_idx"])],
                "direct_answers": str(list(r.get("direct_answers", []))),
                "clean_png": os.path.join("long_answer", "images", f"{iid}.png"),
            }
    out = [items[k] for k in sorted(items)]
    json.dump(out, open(os.path.join(args.out_dir, "aokvqa_items.json"), "w"), indent=2)
    missing = sorted(k for k in keep if k not in items)
    print(f"wrote {len(out)} of {len(keep)} A-OKVQA items to {args.out_dir}")
    if missing:
        print(f"missing {len(missing)} ids (not found in splits {args.splits}); first: {missing[:3]}")


if __name__ == "__main__":
    main()
