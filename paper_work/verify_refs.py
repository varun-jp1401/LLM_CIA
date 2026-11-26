import json, sys, urllib.request, urllib.parse, time
DOIS = """10.1038/s41586-024-07421-0
10.1038/s41746-024-01282-7
10.1038/s41746-025-01670-7
10.1038/s41591-024-03097-1
10.1038/s41467-024-52415-1
10.1038/s41467-024-46411-8
10.1145/3571730
10.18653/v1/D19-1259
10.1186/s12859-019-3119-4
10.18653/v1/2021.findings-emnlp.297
10.18653/v1/2020.emnlp-main.609
10.18653/v1/D18-1187
10.1093/bioinformatics/btz682
10.1145/3458754
10.1093/bib/bbac409
10.1038/s41586-023-06291-2
10.1038/s41591-024-03423-7
10.18653/v1/N18-1074
10.1162/tacl_a_00454
10.1109/TBDATA.2019.2921572
10.1561/1500000019
10.1145/1571941.1572114
10.1038/s41591-023-02448-8
10.1007/s10916-026-02430-0
10.2196/93761
10.1038/s41598-026-75909-y
10.2196/70190
10.2196/55648
10.1007/s11096-024-01839-2
10.3390/app11146421
10.1162/tacl_a_00713
10.18653/v1/2023.emnlp-main.741
10.18653/v1/2023.emnlp-main.557
10.18653/v1/2024.findings-acl.212
10.18653/v1/2023.acl-long.910
10.1109/TPAMI.2022.3148903
10.1016/j.jbi.2018.12.011
10.1093/jamia/ocaa026""".split()
for d in DOIS:
    try:
        r=urllib.request.urlopen(urllib.request.Request("https://api.crossref.org/works/"+urllib.parse.quote(d),headers={"User-Agent":"paper-ref-check/1.0"}),timeout=30)
        m=json.load(r)["message"]
        au=m.get("author",[{}]); a0=au[0].get("family","?")
        yr=(m.get("issued",{}).get("date-parts",[[None]])[0] or [None])[0]
        print(f"OK  {d}\n    {m['title'][0][:110]}\n    {a0} et al.({len(au)}) | {(m.get('container-title') or ['?'])[0][:60]} | {yr} | vol {m.get('volume')} iss {m.get('issue')} pp {m.get('page') or m.get('article-number')}")
    except Exception as e:
        print(f"ERR {d} {str(e)[:60]}")
    time.sleep(0.2)
