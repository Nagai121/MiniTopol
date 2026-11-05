"""Probe the published ATLAS total/metadata endpoints without large downloads."""
import json
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from pathlib import Path
ids = ["1k5n_A", "2au7_A", "3ilw_A", "5e3e_A"]
out = []
def visit(url, method="GET", headers=None, cap=8192):
    request = Request(url, method=method, headers={"User-Agent":"Atlas-Size-Research/1.0", **(headers or {})})
    try:
        with urlopen(request, timeout=25) as response:
            data = response.read(cap) if cap else b""
            return dict(status=response.status, final_url=response.url, headers={k:v for k,v in response.headers.items() if k.lower() in ["content-length","content-range","accept-ranges","content-type","transfer-encoding"]}, preview=data[:1500].decode("utf-8", errors="replace"))
    except Exception as e:
        return {"error": type(e).__name__+": "+str(e)[:300]}
for ident in ids:
    row={"id": ident}
    row["metadata"] = visit(f"https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/metadata/{ident}")
    row["total_head"] = visit(f"https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/total/{ident}", "HEAD", cap=0)
    row["zip_head"] = visit(f"https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{ident}/{ident}_total.zip", "HEAD", cap=0)
    row["zip_tail"] = visit(f"https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/{ident}/{ident}_total.zip", headers={"Range":"bytes=-128"},cap=128)
    out.append(row)
    print(json.dumps(row, ensure_ascii=False, indent=1), flush=True)
Path("analyses").mkdir(exist_ok=True)
Path("analyses/atlas_probe.json").write_text(json.dumps(out, indent=2,ensure_ascii=False)+"\n")
