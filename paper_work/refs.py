"""Verified reference database -> IEEE-formatted strings. Metadata fetched from Crossref / arXiv (cached)."""
import json, re, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from pathlib import Path
CACHE = Path(__file__).parent / "refs_cache.json"

DOI = {
 "farquhar2024": "10.1038/s41586-024-07421-0", "han2024": "10.1038/s41746-024-01282-7",
 "asgari2025": "10.1038/s41746-025-01670-7", "hager2024": "10.1038/s41591-024-03097-1",
 "williams2024": "10.1038/s41467-024-52415-1", "sandmann2024": "10.1038/s41467-024-46411-8",
 "ji2023": "10.1145/3571730", "jin2019": "10.18653/v1/D19-1259", "abacha2019": "10.1186/s12859-019-3119-4",
 "sarrouti2021": "10.18653/v1/2021.findings-emnlp.297", "wadden2020": "10.18653/v1/2020.emnlp-main.609",
 "romanov2018": "10.18653/v1/D18-1187", "lee2020": "10.1093/bioinformatics/btz682", "gu2021": "10.1145/3458754",
 "luo2022": "10.1093/bib/bbac409", "singhal2023": "10.1038/s41586-023-06291-2", "singhal2025": "10.1038/s41591-024-03423-7",
 "thorne2018": "10.18653/v1/N18-1074", "guo2022": "10.1162/tacl_a_00454", "johnson2021": "10.1109/TBDATA.2019.2921572",
 "robertson2009": "10.1561/1500000019", "cormack2009": "10.1145/1571941.1572114",
 "thirunavukarasu2023": "10.1038/s41591-023-02448-8", "boie2026": "10.1007/s10916-026-02430-0",
 "liang2026": "10.2196/93761", "reddy2026": "10.1038/s41598-026-75909-y", "wang2025": "10.2196/70190",
 "franc2024": "10.2196/55648", "debusser2025": "10.1007/s11096-024-01839-2", "jin2021": "10.3390/app11146421",
 "kamoi2024": "10.1162/tacl_a_00713", "min2023": "10.18653/v1/2023.emnlp-main.741",
 "manakul2023": "10.18653/v1/2023.emnlp-main.557", "dhuliawala2024": "10.18653/v1/2024.findings-acl.212",
 "gao2023": "10.18653/v1/2023.acl-long.910", "laban2022": "10.1162/tacl_a_00453", "honovich2022": "10.18653/v1/2022.naacl-main.287",
 "zha2023": "10.18653/v1/2023.acl-long.634", "kamath2020": "10.18653/v1/2020.acl-main.503", "desai2020": "10.18653/v1/2020.emnlp-main.21",
 "kryscinski2020": "10.18653/v1/2020.emnlp-main.750", "maynez2020": "10.18653/v1/2020.acl-main.173", "pal2023": "10.18653/v1/2023.conll-1.21", "reimers2019": "10.18653/v1/D19-1410", "wolf2020": "10.18653/v1/2020.emnlp-demos.6",
}
ARXIV = {"hu2022": "2106.09685", "dettmers2023": "2305.14314", "yang2025": "2505.09388", "he2021": "2111.09543",
         "lewis2020": "2005.11401", "huang2023": "2310.01798", "guo2017": "1706.04599", "geifman2017": "1705.08500"}
ONLINE = {
 "pubmedbert_st": ("“S-PubMedBert-MS-MARCO (sentence-transformers model card),” Hugging Face. [Online]. Available: https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO", ""),
 "nli_ce": ("“nli-deberta-v3-small (cross-encoder model card),” Hugging Face. [Online]. Available: https://huggingface.co/cross-encoder/nli-deberta-v3-small", ""),
 "healthver_resplit": ("“HealthVer re-split (train/dev/test) dataset card,” Hugging Face. [Online]. Available: https://huggingface.co/datasets/jpd459/healthver_resplit", ""),
}
ABBR = {
 "Nature": "Nature", "Nature Medicine": "Nat. Med.", "npj Digital Medicine": "npj Digit. Med.", "Nature Communications": "Nat. Commun.",
 "ACM Computing Surveys": "ACM Comput. Surv.", "BMC Bioinformatics": "BMC Bioinformatics", "Bioinformatics": "Bioinformatics",
 "ACM Transactions on Computing for Healthcare": "ACM Trans. Comput. Healthcare", "Briefings in Bioinformatics": "Brief. Bioinform.",
 "Transactions of the Association for Computational Linguistics": "Trans. Assoc. Comput. Linguist.",
 "IEEE Transactions on Big Data": "IEEE Trans. Big Data", "Applied Sciences": "Appl. Sci.", "Journal of Medical Systems": "J. Med. Syst.",
 "JMIR AI": "JMIR AI", "Scientific Reports": "Sci. Rep.", "JMIR Medical Education": "JMIR Med. Educ.",
 "Journal of Medical Internet Research": "J. Med. Internet Res.", "International Journal of Clinical Pharmacy": "Int. J. Clin. Pharm.",
 "Foundations and Trends in Information Retrieval": "Found. Trends Inf. Retr.",
}
CONF = {
 "10.18653/v1/D19-": "in Proc. Conf. Empirical Methods Natural Lang. Process. 9th Int. Joint Conf. Natural Lang. Process. (EMNLP-IJCNLP)",
 "10.18653/v1/D18-": "in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)",
 "10.18653/v1/2020.emnlp-main": "in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)",
 "10.18653/v1/2020.emnlp-demos": "in Proc. Conf. Empirical Methods Natural Lang. Process.: Syst. Demonstrations",
 "10.18653/v1/2023.emnlp-main": "in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)",
 "10.18653/v1/2021.findings-emnlp": "in Findings Assoc. Comput. Linguist.: EMNLP",
 "10.18653/v1/2024.findings-acl": "in Findings Assoc. Comput. Linguist.: ACL",
 "10.18653/v1/2023.acl-long": "in Proc. 61st Annu. Meeting Assoc. Comput. Linguist. (ACL)",
 "10.18653/v1/2022.naacl-main": "in Proc. Conf. North Amer. Chapter Assoc. Comput. Linguist.: Human Lang. Technol. (NAACL-HLT)",
 "10.18653/v1/2020.acl-main": "in Proc. 58th Annu. Meeting Assoc. Comput. Linguist. (ACL)",
 "10.18653/v1/2023.conll": "in Proc. 27th Conf. Comput. Natural Lang. Learn. (CoNLL)",
 "10.18653/v1/N18": "in Proc. Conf. North Amer. Chapter Assoc. Comput. Linguist.: Human Lang. Technol. (NAACL-HLT)",
 "10.1145/1571941": "in Proc. 32nd Int. ACM SIGIR Conf. Res. Develop. Inf. Retr.",
}


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ref-builder/1.0 (research)"}), timeout=40).read()


def fetch():
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    for k, d in DOI.items():
        if k not in cache:
            cache[k] = json.loads(_get("https://api.crossref.org/works/" + urllib.parse.quote(d)))["message"]
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for k, i in ARXIV.items():
        if k not in cache:
            e = ET.fromstring(_get(f"http://export.arxiv.org/api/query?id_list={i}").decode()).find("a:entry", ns)
            cache[k] = {"arxiv": i, "title": re.sub(r"\s+", " ", e.find("a:title", ns).text),
                        "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
                        "year": int(e.find("a:published", ns).text[:4])}
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return cache


def initials(given):
    parts = re.split(r"\s+", given.replace("-", "- ").strip())
    out = []
    for p in parts:
        if not p:
            continue
        out.append(p[0].upper() + "." + ("-" if p.endswith("-") else ""))
    return " ".join(out).replace("- ", "-")


def fmt_names(names):
    f = [f"{initials(g)} {fam}".strip() for g, fam in names]
    if len(f) > 6:
        return f[0] + " et al."
    if len(f) <= 2:
        return " and ".join(f)
    return ", ".join(f[:-1]) + ", and " + f[-1]


def clean_title(t):
    t = re.sub(r"<[^>]+>", "", t)
    return re.sub(r"\s+", " ", t).strip()


def fmt(key, cache):
    if key in ONLINE:
        return ONLINE[key][0]
    m = cache[key]
    if "arxiv" in m:
        names = []
        for a in m["authors"]:
            p = a.split()
            names.append((" ".join(p[:-1]), p[-1]))
        return f"{fmt_names(names)}, “{m['title']},” arXiv:{m['arxiv']}, {m['year']}."
    names = [(a.get("given", ""), a.get("family", "")) for a in m["author"]]
    title = clean_title(m["title"][0])
    doi = DOI[key]
    yr = m["issued"]["date-parts"][0][0]
    if key == "lee2020":
        yr = 2020   # vol. 36, no. 4 print year
    if key == "debusser2025":
        yr = 2025   # print-volume year (Crossref lists the online-first year, 2024)
    ct = (m.get("container-title") or [""])[0]
    ct = re.sub(r"[^\x00-\x7f]", "", ct)
    ct = re.sub(r"\s+", " ", ct).strip()
    if m.get("type") == "journal-article":
        ab = ABBR.get(ct, ct)
        s = f"{fmt_names(names)}, “{title},” {ab}"
        if m.get("volume") and key != "robertson2009":
            s += f", vol. {m['volume']}"
        if m.get("issue") and key != "robertson2009":
            s += f", no. {m['issue']}"
        pg = m.get("page") or m.get("article-number")
        if pg and key != "robertson2009":
            s += (", pp. " if "-" in str(pg) else ", Art. no. ") + str(pg).replace("-", "–")
        s += f", {yr}"
    else:
        pre = next((v for p, v in CONF.items() if doi.startswith(p)), "in Proc.")
        s = f"{fmt_names(names)}, “{title},” {pre}, {yr}"
        if m.get("page"):
            s += ", pp. " + m["page"].replace("-", "–")
    return s + f", doi: {doi}."


if __name__ == "__main__":
    c = fetch()
    for k in list(DOI) + list(ARXIV) + list(ONLINE):
        print(k, "::", fmt(k, c)[:240])
