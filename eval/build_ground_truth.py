import argparse, csv, glob, json, pathlib, re
from itertools import islice

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_JSONL = ROOT / "eval" / "gt_qa.jsonl"
OUT_CSV   = ROOT / "eval" / "gt_qa.csv"
MANIFEST  = RAW_DIR / "_manifest.json"  # optional (if you added it in the crawler)

def first_sentences(text: str, n: int = 3) -> str:
    # coarse sentence split; avoids pulling code blocks
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(list(islice((p for p in parts if p.strip()), n)))

def clean_heading(h: str) -> str:
    h = h.strip()
    h = h.replace("¶", "").replace("—", "-")
    h = re.sub(r"\s+", " ", h)
    return h.strip(":-·• ")

def is_heading_candidate(line: str) -> bool:
    line = line.strip()
    if not line or len(line) < 3:
        return False
    if len(line.split()) > 12:      # short-ish
        return False
    if line.endswith(":"):          # likely a label
        return False
    if any(tok in line for tok in ["http://","https://","{","}","=","def ","class "]):
        return False
    # many docs have a '¶' marker after headings
    if "¶" in line: 
        return True
    # otherwise use simple titlecase heuristic
    return bool(re.match(r"^[A-Z][A-Za-z0-9 \-\(\)/_]+$", line))

def window_after(lines, start_idx, span=8):
    # gather a small window of following lines until next heading-ish break
    buf = []
    for j in range(start_idx + 1, min(len(lines), start_idx + 1 + span)):
        t = lines[j].strip()
        if not t: 
            continue
        # stop if the next line also looks like a heading
        if is_heading_candidate(t):
            break
        buf.append(t)
    return "\n".join(buf)

def main(min_ans_chars: int, max_per_file: int):
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    if MANIFEST.exists():
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    rows = []
    qid = 0
    files = sorted(glob.glob(str(RAW_DIR / "*.md")))
    if not files:
        raise SystemExit(f"No raw pages found in {RAW_DIR}. Run the crawler first.")

    for fp in files:
        name = pathlib.Path(fp).name
        text = pathlib.Path(fp).read_text(encoding="utf-8")
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        # use manifest title if present, otherwise first non-empty line
        man = manifest.get(name, {})
        doc_title = man.get("title") or (lines[0] if lines else name.replace(".md",""))
        doc_title = clean_heading(doc_title)

        # collect candidate headings (title + subheadings)
        candidates = [doc_title]
        for i, line in enumerate(lines[:180]):   # scan the top part of the doc
            if is_heading_candidate(line):
                candidates.append(clean_heading(line))
        # dedupe while preserving order
        seen, heads = set(), []
        for h in candidates:
            if h and h.lower() not in seen:
                heads.append(h); seen.add(h.lower())

        # create Q&A from first few headings in this file
        created = 0
        for i, line in enumerate(lines):
            if created >= max_per_file:
                break
            if not is_heading_candidate(line):
                continue
            h = clean_heading(line)
            if not h or h.lower() not in {x.lower() for x in heads}:
                continue

            ctx = window_after(lines, i, span=8)
            ans = first_sentences(ctx, n=4)
            if len(ans) < min_ans_chars:
                continue

            # generate 2 phrasings per heading
            templates = [
                "What is {h}?",
                "How does {h} work in FastAPI?"
            ]
            for t in templates:
                q = t.format(h=h)
                rows.append({
                    "id": f"hq_{qid}",
                    "question": q,
                    "answer": ans,
                    "source": name,
                    "title": doc_title,
                    "section": h,
                    "url": man.get("url")
                })
                qid += 1
                created += 1
                if created >= max_per_file:
                    break

        # fallback: if nothing created for this file, create 1 Q&A from top paragraph
        if created == 0 and lines:
            ctx = "\n".join(lines[1:8])
            ans = first_sentences(ctx, n=3)
            if len(ans) >= min_ans_chars:
                rows.append({
                    "id": f"hq_{qid}",
                    "question": f"What does {doc_title} cover in FastAPI?",
                    "answer": ans,
                    "source": name,
                    "title": doc_title,
                    "section": doc_title,
                    "url": man.get("url")
                })
                qid += 1

    # write jsonl
    with OUT_JSONL.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    # write csv
    with OUT_CSV.open("w", encoding="utf-8", newline="") as w:
        writer = csv.DictWriter(w, fieldnames=["id","question","answer","source","title","section","url"])
        writer.writeheader(); writer.writerows(rows)

    print(f"[done] wrote {len(rows)} Q&A → {OUT_JSONL} and {OUT_CSV}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-ans-chars", type=int, default=60, help="skip answers shorter than this")
    p.add_argument("--max-per-file", type=int, default=6, help="max Q&A pairs per source file")
    args = p.parse_args()
    main(args.min_ans_chars, args.max_per_file)
