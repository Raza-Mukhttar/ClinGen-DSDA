# File: medinotes_repro/scripts/human_eval_analysis.py
"""
Process human evaluation CSV with columns: id, user_type, query_type, accuracy, completeness, satisfaction, usefulness
Produces aggregated Table I results as JSON/CSV.
"""
import argparse, csv, json, statistics, os

def aggregate(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as fh:
        r=csv.DictReader(fh)
        for line in r:
            rows.append(line)
    # group by user_type & query_type
    groups={}
    for row in rows:
        k=(row["user_type"], row["query_type"])
        groups.setdefault(k, []).append(row)
    out={}
    for k,items in groups.items():
        acc = [float(i["accuracy"]) for i in items]
        comp = [float(i["completeness"]) for i in items]
        sat = [float(i["satisfaction"]) for i in items]
        use = [float(i["usefulness"]) for i in items]
        out[f"{k[0]}|{k[1]}"]={
            "accuracy": statistics.mean(acc),
            "completeness": statistics.mean(comp),
            "satisfaction": statistics.mean(sat),
            "usefulness": statistics.mean(use),
            "n": len(items)
        }
    # averages
    avg = {
        "accuracy": statistics.mean([v["accuracy"] for v in out.values()]),
        "completeness": statistics.mean([v["completeness"] for v in out.values()]),
        "satisfaction": statistics.mean([v["satisfaction"] for v in out.values()]),
        "usefulness": statistics.mean([v["usefulness"] for v in out.values()])
    }
    return {"groups": out, "average": avg}

if __name__=="__main__":
    import argparse, json
    p=argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    args=p.parse_args()
    res=aggregate(args.csv)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as fh:
        json.dump(res,fh,indent=2)
    print("Wrote human eval summary to", args.out)
