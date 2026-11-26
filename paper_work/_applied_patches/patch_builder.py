import re
p = r"c:\LLM\LLM_CIA\paper_work\build_paper.py"
b = open(p, encoding="utf8").read()


def rep(old, new, count=1):
    global b
    assert old in b, "missing: " + old[:70]
    b = b.replace(old, new, count)


# 1. author runs: line break at the start of each affiliation run
rep(r'''    r = p.add_run(name); r.font.size = Pt(9)
    for ln in lines:
        r.add_break()
        r2 = p.add_run(ln); r2.font.size = Pt(9); r2.italic = not ln.startswith("[")''',
    r'''    r = p.add_run(name); r.font.size = Pt(9)
    for ln in lines:
        r2 = p.add_run(); r2.add_break(); r2.add_text(ln); r2.font.size = Pt(9); r2.italic = not ln.startswith("[")''')

# 2. strip leftover template text boxes / pictures
rep(r'''P.anchor = anchor._p
''', r'''P.anchor = anchor._p
for el in list(doc.element.body.iter(qn("w:r"))):
    if el.find(qn("mc:AlternateContent")) is not None or el.find(qn("w:pict")) is not None:
        el.getparent().remove(el)
''')

# 3. test-mode numbering shift
rep(r'''    build_lib.FIGN.update({"cal": 7}); build_lib.TABN.update({"sel": 6, "pq": 7, "err": 8})''',
    r'''    build_lib.FIGN.update({"cal": 7, "pr": 8}); build_lib.TABN.update({"abl": 99, "sel": 7, "det": 8, "brk": 9, "pq": 10, "err": 11})''')

# 4. dynamic cross-references
for old, new in [
    ("Fig. 1 shows the retrieval", "{F:arch} shows the retrieval"),
    ("Table II summarises the data.", "{T:data} summarises the data."),
    ("at step {LG[-1]['step']} (Fig. 3)", "at step {LG[-1]['step']} ({F:curve})"),
    ("Table IV and Fig. 4 answer RQ1", "{T:main} and {F:main} answer RQ1"),
    ("Fig. 5 and Table V show that on HealthVer", "{F:conf} and {T:pc} show that on HealthVer"),
    ("Fig. 6 breaks the synthetic results", "{F:kinds} breaks the synthetic results"),
    ("Table VI and Fig. 7 address RQ2", "{T:abl} and {F:abl} address RQ2"),
    ("Fig. 8 and Table VII address RQ3", "{F:cal} and {T:sel} address RQ3"),
    ("Table VIII reports the held-out test split", "{T:pq} reports the held-out test split"),
    ("The examples in Table IX show", "The examples in {T:err} show"),
    ("Three observations follow from Table I.", "Three observations follow from {T:rel}."),
]:
    rep(old, new)

# 5. exact family counts from the generator module
rep(r'''ROOT = Path(__file__).parent.parent''', r'''ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import build_claim_dataset as BCD
NANT, NQ, NSW, NSYN, NMOD, NADD, NOADD, NPRE = (len(BCD.ANTONYMS) // 2, len(BCD.QUANT) // 2, len(BCD.OOD_SWAP) // 2, len(BCD.SAFE_SYN) // 2,
                                               len(BCD.MODALS), len(BCD.TRAIN_ADD), len(BCD.OOD_ADD), len(BCD.PREFIXES))''')
rep("a synonym swap from a list of 13 safe pairs with probability 0.35", "a synonym swap from a list of {NSYN} safe pairs with probability 0.35")
rep("antonym replacement (33 pairs), quantifier reversal (7 pairs) or numeric scaling", "antonym replacement ({NANT} pairs), quantifier reversal ({NQ} pairs) or numeric scaling")
rep("(12 pairs such as men↔women, left↔right, acute↔chronic)", "({NSW} pairs such as men↔women, left↔right, acute↔chronic)")
rep("and uses six unseen overclaim templates in place of the ten used in training.", "and uses {NOADD} unseen overclaim templates in place of the {NADD} used in training.")
open(p, "w", encoding="utf8").write(b)
print("patched")
