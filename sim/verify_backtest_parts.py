#!/usr/bin/env python3
"""The back-test's scoring and its filter port, on cases with known answers.
python3 sim/verify_backtest_parts.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_youtube import der, found, looks_hallucinated  # noqa: E402

bad = 0
def check(name, ok, detail=""):
    global bad
    bad += not ok
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")

print("DER — optimal mapping, 0.25 s collar")
ref = [(0, 10, "A"), (10.5, 20, "B"), (20.5, 30, "A"), (30.5, 40, "C")]
same_labels_swapped = [(0, 10, "x"), (10.5, 20, "y"), (20.5, 30, "x"), (30.5, 40, "z")]
d = der(ref, same_labels_swapped, 40)
check("identical segmentation, different label names -> 0%", abs(d["der"]) < 1e-9, f"DER {d['der']*100:.1f}%, mapping {d['mapping']}")
two_merged = [(0, 10, "x"), (10.5, 20, "x"), (20.5, 30, "x"), (30.5, 40, "z")]     # A and B merged
d = der(ref, two_merged, 40)
check("A and B merged into one label -> ~9.5 s confusion of 38 s", 0.20 < d["confusion"] < 0.30 and d["miss"] < 0.01, f"confusion {d['confusion']*100:.1f}%, DER {d['der']*100:.1f}%")
one_missing = [(0, 10, "x"), (10.5, 20, "y"), (30.5, 40, "z")]                      # 20.5-30 not gated
d = der(ref, one_missing, 40)
check("a 9.5 s turn not gated -> ~25% miss", 0.20 < d["miss"] < 0.30 and d["confusion"] < 0.01, f"miss {d['miss']*100:.1f}%, DER {d['der']*100:.1f}%")
extra = same_labels_swapped + [(41, 46, "w")]                                        # 5 s of speech the ref calls silence
d = der(ref, extra, 46)
check("5 s false alarm -> ~13% FA", 0.10 < d["fa"] < 0.16 and d["confusion"] < 0.01, f"fa {d['fa']*100:.1f}%")
overlap_ref = ref + [(5, 8, "B")]                                                    # B talks over A for 3 s
d = der(overlap_ref, same_labels_swapped, 40)
check("3 s of overlap the app cannot represent -> counted as miss", 0.04 < d["miss"] < 0.10 and d["confusion"] < 0.02, f"miss {d['miss']*100:.1f}%")
check("found() ignores labels under 5 s", found([(0, 10, "a"), (10, 12, "b"), (12, 30, "c")]) == 2)

print("\nhallucination filter — the JS test's 20 cases")
cases = [("Thank you.", 5.0, True), ("Thank you.", 0.8, False), ("Yes, agreed.", 1.7, False), ("Right.", 0.5, False),
         ("Mm-hmm.", 0.6, False), ("you", 3.0, True), ("Subtitles by the Amara.org community", 3.0, True),
         ("Please like and subscribe!", 2.0, True), ("[Music]", 3.0, True), ("(applause)", 2.0, True), ("", 2.0, True),
         ("...", 1.0, True), ("the the the the the the the the the the", 3.0, True),
         ("so we said so we said so we said so we said so we said", 4.0, True), (" ".join(["word"] * 40), 3.0, True),
         ("I think Aisha should own the medical reports item, she has the context from last quarter.", 5.5, False),
         ("The claims backlog is sitting at about three hundred and forty cases this week.", 6.3, False),
         ("Can we come back to that after the break?", 2.9, False), ("谢谢大家，我们下周再讨论这个问题。", 3.0, False),
         ("字幕由 Amara.org 社区提供", 3.0, True)]
wrong = [(t, s, e) for t, s, e in cases if looks_hallucinated(t, s) != e]
check(f"{len(cases) - len(wrong)} of {len(cases)} agree with the JS", not wrong, str(wrong[:3]))
print("\n" + (f"{bad} wrong" if bad else "all checks passed"))
raise SystemExit(1 if bad else 0)
