import re


def header_aware_chunks(text, max_tokens=700, overlap=120):
    # naive token proxy: words ~ tokens
    paras = re.split(r"\n{2,}", text)
    chunks, cur, count = [], [], 0
    for p in paras:
        w = p.split()
        if count + len(w) > max_tokens and cur:
            chunks.append(''.join(cur))

            # overlap: keep tail words
            tail = ' '.join(' '.join(cur).split()[-overlap:])
            cur = [tail, p]
            count = len(tail.split()) + len(w)
        else:
            cur.append(p); count += len(w)
    if cur: chunks.append(''.join(cur))
    return [c for c in chunks if c.strip()]