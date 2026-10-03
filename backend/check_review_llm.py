"""Scratch check: where does the stored (LLM-refined) review evidence differ from the rule-based one?"""
from app.data.loader import load_processed
from app.services.review_analyzer import classify_snippet


def main() -> None:
    same = diff = added = 0
    for p in load_processed():
        stored = {(r["quote"], r["aspect"]): r for r in p.review_evidence}
        rules = {(r["quote"], r["aspect"]): r for s in p.review_snippets for r in classify_snippet(s)}
        for key in sorted(set(stored) | set(rules)):
            s, r = stored.get(key), rules.get(key)
            if s and r and s["polarity"] == r["polarity"]:
                same += 1
                continue
            kind = "DIFFERENT " if (s and r) else "LLM-ADDED "
            diff += bool(s and r)
            added += not (s and r)
            print(f"{p.product_id} {key[1]:10} {kind} rules={r['polarity'] if r else '-':8} "
                  f"llm={s['polarity'] if s else '-':8} | {key[0][:70]}")
    print(f"\nagree: {same} | LLM changed polarity: {diff} | LLM added: {added}")


if __name__ == "__main__":
    main()