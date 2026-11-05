"""Extract ATLAS XTC header atom counts and original TPR/XTC sizes by HTTP range."""
import argparse, csv, io, json, random, struct, tempfile, time, zipfile, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError
INDEX="https://raw.githubusercontent.com/bjing2016/alphaflow/master/splits/atlas.csv"
ROOT="https://www.dsimb.inserm.fr/ATLAS/database/ATLAS"
FRAMES=10001
def fetch(url,method="GET",span=None):
    h={"User-Agent":"MiniTopol-ATLAS-AF-size-survey/1.0"}
    if span:h["Range"]=f"bytes={span[0]}-{span[1]}"
    error=None
    for attempt in range(3):
        try:return urlopen(Request(url,method=method,headers=h),timeout=35)
        except (HTTPError,URLError,TimeoutError) as e:
            error=e
            if attempt<2:time.sleep(attempt+1)
    raise error
def slice_bytes(url,a,b):
    with fetch(url,span=(a,b)) as r:
        if r.status!=206:raise ValueError(f"HTTP range not honored: {r.status}")
        data=r.read(b-a+2)
    if len(data)!=b-a+1:raise ValueError("Wrong range length")
    return data
def inspect(pair):
    ident,length=pair
    url=f"{ROOT}/{ident}/{ident}_total.zip"
    try:
        with fetch(url,method="HEAD") as response:
            total=int(response.headers["Content-Length"])
        tail_len=min(total,131072)
        tail=slice_bytes(url,total-tail_len,total-1)
        with tempfile.TemporaryFile() as f:
            f.seek(total-tail_len);f.write(tail);f.flush();f.seek(0)
            with zipfile.ZipFile(f) as arc:infos=arc.infolist()
        xtcs=[z for z in infos if z.filename.lower().endswith(".xtc")]
        tprs=[z for z in infos if z.filename.lower().endswith(".tpr")]
        if not xtcs or not tprs:raise ValueError(f"missing file types: {[z.filename for z in infos]}")
        atoms=[]
        for z in xtcs:
            a=z.header_offset
            data=slice_bytes(url,a,min(a+32767,total-1))
            if data[:4]!=b"PK\x03\x04":raise ValueError("Invalid local ZIP header")
            n,e=struct.unpack_from("<HH",data,26)
            raw=data[30+n+e:]
            if z.compress_type==zipfile.ZIP_STORED: decoded=raw
            elif z.compress_type==zipfile.ZIP_DEFLATED:decoded=zlib.decompressobj(-15).decompress(raw,256)
            else:raise ValueError(f"ZIP compression unsupported: {z.compress_type}")
            if len(decoded)<8:raise ValueError("XTC first frame header incomplete")
            magic,natoms=struct.unpack_from(">ii",decoded)
            if magic!=1995 or natoms<1:raise ValueError(f"Bad XTC magic or natoms: {magic} {natoms}")
            atoms.append(natoms)
        return dict(id=ident,protein_residues=length,xtc_files=len(xtcs),tpr_files=len(tprs),
             all_atoms_min=min(atoms),all_atoms_max=max(atoms),frame_count_per_xtc_assumed=FRAMES,
             AF_assumed=sum(a*FRAMES for a in atoms),zip_bytes=total,
             xtc_bytes=sum(z.file_size for z in xtcs),
             tpr_bytes=sum(z.file_size for z in tprs),
             xtc_zip_member_bytes=sum(z.compress_size for z in xtcs),
             tpr_zip_member_bytes=sum(z.compress_size for z in tprs),status="ok")
    except Exception as exc:
        return dict(id=ident,protein_residues=length,status="error",
                    error=f"{type(exc).__name__}: {exc}"[:270])
def main():
    p=argparse.ArgumentParser();p.add_argument("--limit",type=int,default=5)
    p.add_argument("--workers",type=int,default=5)
    args=p.parse_args()
    with fetch(INDEX) as f:src=f.read().decode()
    ids=[(r["name"],len(r["seqres"])) for r in csv.DictReader(io.StringIO(src))]
    chosen=ids if args.limit>=len(ids) else random.Random(20261105).sample(ids,args.limit)
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(inspect,pair) for pair in chosen]),1):
            row=f.result();rows.append(row)
            if i<12 or i%25==0:print(i,"/",len(chosen),json.dumps(row),flush=True)
    rows.sort(key=lambda r:r["id"])
    Path("analyses").mkdir(exist_ok=True)
    Path("analyses/atlas_raw_af_survey.json").write_text(json.dumps({
       "source":INDEX,"n_requested":len(chosen),"frame_count_per_xtc_assumed":FRAMES,"rows":rows},indent=2)+"\n")
    ok=[x for x in rows if x["status"]=="ok"]
    fields=["id","protein_residues","xtc_files","tpr_files","all_atoms_min","all_atoms_max",
       "frame_count_per_xtc_assumed","AF_assumed","zip_bytes","xtc_bytes","tpr_bytes",
       "xtc_zip_member_bytes","tpr_zip_member_bytes"]
    with open("analyses/atlas_raw_af_survey.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(ok)
    print(f"Completed {len(ok)} valid of {len(chosen)}",flush=True)
if __name__=="__main__":main()
