p = r"c:\LLM\LLM_CIA\paper_work\build_paper.py"
b = open(p, encoding="utf8").read()


def rep(old, new, all_=False):
    global b
    assert old in b, "missing: " + old[:80]
    b = b.replace(old, new) if all_ else b.replace(old, new, 1)


rep('HAVE_BASE = "base" in M', '''HAVE_BASE = "base" in M
MODELS_MAIN = [(t, n) for t, n in [("mixed", "Fine-tuned Qwen (ours)"), ("deberta_ft", "DeBERTa fine-tuned"), ("nli_deberta", "DeBERTa NLI zero-shot"),
                                   ("fewshot", "Qwen 6-shot"), ("base", "Qwen zero-shot")] if t in M]''')

# main table + detection table loops
old_loop = 'for tag, nm in (("mixed", "Fine-tuned (ours)"), ("nli_deberta", "DeBERTa-v3 NLI")) + ((("base", "Qwen3-1.7B base"),) if HAVE_BASE else ()):'
assert b.count(old_loop) == 2
b = b.replace(old_loop, "for tag, nm in MODELS_MAIN:")
rep('rows, [1.9, 1.2, 0.8, 1.0, 1.0, 2.4], font=6.3, align=["l", "l", "c", "c", "c", "c"],', 'rows, [2.15, 1.15, 0.8, 0.95, 0.95, 2.4], font=6.2, align=["l", "l", "c", "c", "c", "c"],')
rep('rows, [1.9, 1.2, 0.95, 0.95, 1.15, 1.25, 0.8], font=6.1, align=["l", "l", "c", "c", "c", "c", "c"], shade_rows=(0, 1, 2),',
    'rows, [2.0, 1.15, 0.85, 0.85, 1.1, 1.2, 0.7], font=5.9, align=["l", "l", "c", "c", "c", "c", "c"], shade_rows=(0, 1, 2),')

# comparison paragraph after the main results text
rep('P.body(f"The 0.90 target is not met on real claims:', '''def verdict(d):
    if d["lo"] > 0:
        return "higher than"
    if d["hi"] < 0:
        return "lower than"
    return "statistically indistinguishable from"


if "deberta_ft" in M:
    pd_ = {s: A["paired"][f"mixed_vs_deberta_ft/{s}"] for s in SETS}
    P.body(f"A natural question is whether the gain comes from the LLM backbone or simply from in-domain supervision. We therefore fine-tuned the same DeBERTa-v3-small cross-encoder on the identical 6,500 training examples (three epochs, checkpoint selected on the same validation data). "
           f"It reaches macro-F1 {ci('deberta_ft','medquad_test_iid')}, {ci('deberta_ft','medquad_test_ood')} and {ci('deberta_ft','healthver_test')} on MQ-iid, MQ-ood and HealthVer. "
           f"The adapted Qwen verifier is {verdict(pd_['medquad_test_iid'])} it on MQ-iid ({pdiff('deberta_ft','medquad_test_iid')}), {verdict(pd_['medquad_test_ood'])} it on MQ-ood ({pdiff('deberta_ft','medquad_test_ood')}) and {verdict(pd_['healthver_test'])} it on HealthVer ({pdiff('deberta_ft','healthver_test')}). "
           "This comparison separates the contribution of supervised in-domain data from that of the model family; the project nevertheless uses a decoder LLM because the same model also drafts and repairs answers, so a single backbone serves all stages.")
if "fewshot" in M:
    P.body(f"Prompting alone is not sufficient either: six in-context exemplars (one real and one synthetic per class, taken from the training sets) lift the unadapted Qwen3-1.7B to macro-F1 {ci('fewshot','medquad_test_iid')}, {ci('fewshot','medquad_test_ood')} and {ci('fewshot','healthver_test')}, "
           f"compared with {ci('base','medquad_test_iid')}, {ci('base','medquad_test_ood')} and {ci('base','healthver_test')} zero-shot; the adapted model\\u2019s paired advantage on HealthVer over the six-shot model is {pdiff('fewshot','healthver_test')}.")
P.body(f"The 0.90 target is not met on real claims:''')

# appendix tag list
rep('TAGS = [("mixed", "Fine-tuned"), ("real_only", "Real only"), ("synth_only", "Synth. only"), ("nli_deberta", "DeBERTa NLI"), ("base", "Qwen base")]',
    'TAGS = [("mixed", "Fine-tuned"), ("real_only", "Real only"), ("synth_only", "Synth. only"), ("deberta_ft", "DeBERTa FT"), ("nli_deberta", "DeBERTa NLI"), ("fewshot", "Qwen 6-shot"), ("base", "Qwen base")]')

# related-work table caption unaffected; update appendix sentence
rep("The unadapted backbone and the NLI baseline fail mostly on the neutral and entailment classes of real claims, whereas the adapters fail mostly on contradiction under unseen edit types; ",
    "The unadapted backbone and the zero-shot NLI baseline fail mostly on the neutral and entailment classes of real claims, whereas the adapters fail mostly on contradiction under unseen edit types; ")
open(p, "w", encoding="utf8").write(b)
print("patched4")
