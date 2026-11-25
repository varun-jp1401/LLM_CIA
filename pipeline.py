"""Local Qwen3 generation with a transparent lexical retrieval/verification baseline."""
from __future__ import annotations

import re
import math
from pathlib import Path

DATA = Path(__file__).parent / "data" / "MedQuAD"
MODEL = "Qwen/Qwen3-1.7B"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.split()) >= 4]


class MedicalPipeline:
    def __init__(self, adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map="auto", local_files_only=True)

        # Optionally load a LoRA adapter (fine-tuned model)
        if adapter_path and Path(adapter_path).exists():
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                self.model.eval()
                print(f"[pipeline] LoRA adapter loaded from {adapter_path}")
            except Exception as e:
                print(f"[pipeline] Warning: could not load adapter ({e}). Using base model.")

        from transformers import AutoModelForSequenceClassification
        self.nli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-small", local_files_only=True)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            "cross-encoder/nli-deberta-v3-small", local_files_only=True
        ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.docs = self._load_medquad()

    @staticmethod
    def _load_medquad():
        docs = []
        for p in DATA.rglob("*.xml"):
            try:
                import xml.etree.ElementTree as ET
                root = ET.parse(p).getroot()
                extract = lambda node: " ".join(t.strip() for t in node.itertext() if t.strip())
                # Preserve matching question/answer pairs. Some records contain many
                # QAPairs; flattening all questions and answers corrupts evidence.
                pairs = root.findall(".//QAPair")
                if not pairs:
                    pairs = [root]
                for pair in pairs:
                    q = " ".join(extract(x) for x in pair.findall(".//Question"))
                    a = " ".join(extract(x) for x in pair.findall(".//Answer"))
                    if q and len(a) >= 40:
                        docs.append({"text": f"Question: {q}\nAnswer: {a}", "source": str(p.relative_to(DATA))})
            except (ET.ParseError, OSError, UnboundLocalError):
                continue
        return docs

    def generate(self, prompt: str, max_new_tokens=240):
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inp = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        answer = self.tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        # Avoid model self-evaluation/repeated continuation leaking into the answer.
        answer = re.split(r"\n(?:The answer is|Overall,|In summary, the answer)", answer, maxsplit=1)[0]
        return answer.strip()

    @staticmethod
    def _terms(text):
        return set(re.findall(r"[a-z]{3,}", text.lower()))

    def retrieve(self, query, k=3):
        terms = self._terms(query)
        ranked = []
        n_docs = max(1, len(self.docs))
        dfs = {term: sum(term in self._terms(d["text"]) for d in self.docs) for term in terms}
        for d in self.docs:
            dt = self._terms(d["text"])
            score = sum(math.log(1 + (n_docs - dfs[t] + .5) / (dfs[t] + .5)) for t in terms if t in dt)
            score /= 1 - .75 + .75 * len(dt) / 120
            # Favor matching the source question itself and avoid subtype-only matches.
            source_question = d["text"].split("\nAnswer:", 1)[0].lower()
            score += 4.0 * len(terms & self._terms(source_question)) / max(1, len(terms))
            doc_q_terms = self._terms(source_question)
            # Match the requested question intent (symptoms, causes, treatment, etc.)
            # instead of allowing broad disease-name overlap to dominate.
            intent_groups = {
                "symptom": {"symptom", "symptoms", "sign", "signs"},
                "cause": {"cause", "causes", "caused"},
                "treatment": {"treatment", "treatments", "treat", "managed", "management"},
                "diagnosis": {"diagnose", "diagnosed", "diagnosis", "tests"},
                "prevention": {"prevent", "prevention"},
            }
            generic_tokens = {"what", "are", "is", "the", "common", "can", "how", "does", "do", "help", "reduce", "risk", "of", "for", "and", "or", "with"}
            generic_tokens |= set().union(*intent_groups.values())
            entity_terms = terms - generic_tokens
            score += 18.0 * len(entity_terms & doc_q_terms) / max(1, len(entity_terms))
            for intent, variants in intent_groups.items():
                if terms & variants:
                    if doc_q_terms & variants:
                        score += 9.0
                    else:
                        score *= .45
            # Subtype disease records are poor matches for a general disease query.
            subtype_terms = {"pulmonary", "portal", "drug", "induced", "renovascular", "malignant", "pregnancy", "secondary", "refractory", "sideroblastic", "b12", "folate"}
            unmatched_specific = (doc_q_terms & subtype_terms) - terms
            if unmatched_specific:
                score *= .45 ** len(unmatched_specific)
            if score:
                ranked.append((score, d))
        return [dict(d, score=round(s, 3)) for s, d in sorted(ranked, reverse=True, key=lambda x:x[0])[:k]]

    def verify(self, claim, evidence):
        """NLI entailment screening: premises are evidence snippets, hypotheses are claims."""
        if not evidence:
            return {"label": "neutral", "confidence": 0.0, "entailment_score": 0.0}
        labels = {int(k): v.lower() for k,v in self.nli_model.config.id2label.items()}
        claim_terms = self._terms(claim)
        candidates = []
        for item in evidence:
            answer = item["text"].split("\nAnswer:", 1)[-1]
            chunks = [x.strip() for x in re.split(r"(?<=[.!?])\s+", answer) if len(x.split()) >= 5]
            for chunk in chunks:
                terms = self._terms(chunk)
                lexical = len(claim_terms & terms) / max(1, len(claim_terms))
                candidates.append((lexical, chunk, item["source"]))
        # NLI on short, relevant answer sentences avoids truncating the evidence behind long question lists.
        candidates = sorted(candidates, reverse=True)[:12]
        # Batch all candidate premise-hypothesis pairs in a single forward pass
        premises = [chunk for _, chunk, _ in candidates]
        sources = [src for _, _, src in candidates]
        batch = self.nli_tokenizer(
            premises,
            [claim] * len(premises),
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        batch = {k: v.to(self.nli_model.device) for k, v in batch.items()}
        with self.torch.inference_mode():
            probs = self.torch.softmax(self.nli_model(**batch).logits, dim=-1)
        entail_scores = probs[:, entail_idx]
        best_idx = int(entail_scores.argmax())
        best_entail = float(entail_scores[best_idx])
        pred_label_idx = int(probs[best_idx].argmax())
        return {
            "label": labels.get(pred_label_idx, "neutral"),
            "confidence": float(probs[best_idx, pred_label_idx]),
            "entailment_score": best_entail,
            "source": sources[best_idx],
        }

    def run(self, question, retriever=None, is_qa_choice: bool = False):
        if is_qa_choice:
            prompt = f"Answer this biomedical yes/no/maybe question with one word (yes, no, or maybe) followed by a brief explanation.\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Answer this medical information question cautiously in 3-5 sentences. Say when uncertain.\nQuestion: {question}\nAnswer:"
        draft = self.generate(prompt)
        evidence = retriever.retrieve(question, k=5) if retriever else self.retrieve(question, k=5)
        claims = []
        for claim in _sentences(draft):
            terms = self._terms(claim)
            check = self.verify(claim, evidence)
            risk = "high" if re.search(r"\b(dose|dosage|prescription|drug interaction|overdose|stop taking|start taking|emergency|diagnos(?:is|e) of|ace inhibitors?|beta blockers?|diuretics?|acetaminophen|ibuprofen|naproxen|antibiotics?)\b", claim, re.I) else "medium"
            supported = check["entailment_score"] >= .70
            claims.append({"claim": claim, "verifier_label": check["label"], "confidence": round(check["entailment_score"],3), "risk": risk,
                           "status": "potentially_supported" if supported else "unverified",
                           "evidence_source": check.get("source")})
        flagged = [c for c in claims if c["status"] == "unverified"]
        abstained = not evidence
        # Do not make unsupported treatment/dose edits based on weak lexical matches.
        if flagged and not abstained and not any(c["risk"] == "high" for c in flagged):
            context = "\n\n".join(f"[{i+1}] {d['text'][:650]} (source: {d['source']})" for i,d in enumerate(evidence[:3]))
            if is_qa_choice:
                repair_prompt = f"Using only the evidence below, answer this question with one word (yes, no, or maybe) followed by a brief explanation.\nQuestion: {question}\nDraft: {draft}\nEvidence:\n{context}\nAnswer:"
            else:
                repair_prompt = f"Answer cautiously using only the supplied evidence. Do not add facts beyond it. Keep uncertainty clear and cite evidence labels like [1].\nQuestion: {question}\nDraft: {draft}\nEvidence:\n{context}\nRevised answer:"
            repaired = self.generate(repair_prompt, max_new_tokens=180)
            # Require NLI support for every repaired sentence before showing it.
            revised_claims = _sentences(repaired)
            checks = [self.verify(c, evidence) for c in revised_claims]
            if revised_claims and all(c["entailment_score"] >= .70 for c in checks):
                answer = repaired
            else:
                # For general, non-treatment claims, surface the model draft with an
                # explicit verification warning instead of falsely treating NLI-neutral as unsafe.
                answer = draft + "\n\nEvidence check: the available passages did not let the verifier confirm every statement. Treat this as unverified information."
        elif flagged and any(c["risk"] == "high" for c in flagged):
            abstained = True
            answer = "I can’t verify the potentially high-risk claim against the available evidence. Please consult a qualified healthcare professional or an authoritative medical source."
        else:
            answer = draft
        return {"question": question, "draft": draft, "answer": answer, "abstained": abstained,
                "claims": claims, "evidence": evidence, "model": MODEL,
                "note": "Independent NLI screening is an automated research heuristic; it can be wrong and is not clinical validation."}
