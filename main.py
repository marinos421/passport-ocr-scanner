import argparse
import csv
import json
import os
from pathlib import Path

from services.ocr_service import extract_passport, save_debug_overlay

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def process_one(path: str, pretty: bool, fix: bool, debug: bool):
    data = extract_passport(path, fix=fix)
    if debug:
        out_dbg = Path(path).with_suffix("")  # strip ext
        out_dbg = f"{out_dbg}_debug.jpg"
        try:
            save_debug_overlay(path, out_dbg)
            print(f"[debug] saved {out_dbg}")
        except Exception as e:
            print(f"[debug] failed: {e}")
    print(json.dumps(data, indent=2 if pretty else None, ensure_ascii=False))
    return data

def process_folder(folder: str, pretty: bool, fix: bool, debug: bool, csv_out: str | None):
    src = Path(folder)
    files = [p for p in src.rglob("*") if p.suffix.lower() in EXTS]
    if not files:
        raise SystemExit(f"No images found under {src.resolve()}")

    rows = []
    ok = 0
    for p in sorted(files):
        try:
            d = extract_passport(str(p), fix=fix)
            if debug:
                try:
                    save_debug_overlay(str(p), str(p.with_suffix("").as_posix() + "_debug.jpg"))
                except Exception:
                    pass
            ok += 1 if d.get("final_checksum_ok") else 0
            rows.append({"file": p.name, **d})
            print(("✅" if d.get("final_checksum_ok") else "❌"), p.name, d.get("country"), d.get("passport_number"))
            if pretty and not csv_out:
                print(json.dumps(d, indent=2, ensure_ascii=False))
        except Exception as e:
            print("❌", p.name, "ERROR:", e)

    print(f"\n=== SUMMARY ===  {ok}/{len(files)} final checksums OK")

    if csv_out:
        fields = [
            "file","doc_type","country","surname","given_names",
            "passport_number","passport_number_checksum_ok",
            "nationality","birth_date","birth_date_checksum_ok",
            "sex","expiry_date","expiry_date_checksum_ok",
            "personal_number","personal_number_checksum_ok","final_checksum_ok"
        ]
        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                out = {k: r.get(k) for k in fields}
                out["file"] = r.get("file")
                w.writerow(out)
        print(f"[csv] wrote {csv_out}")

def main():
    ap = argparse.ArgumentParser(description="TD3 passport extractor")
    ap.add_argument("--image", "-i", help="Path to a single image")
    ap.add_argument("--folder", "-f", help="Folder of images (recurses)")
    ap.add_argument("--csv", help="Write results to CSV when using --folder")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout")
    ap.add_argument("--debug", action="store_true", help="Save MRZ region overlay next to inputs")
    ap.add_argument("--fix", action="store_true", help="Apply light MRZ digit/letter corrections before parsing")
    args = ap.parse_args()

    if bool(args.image) == bool(args.folder):
        raise SystemExit("Choose exactly one: --image OR --folder")

    if args.image:
        if not os.path.exists(args.image):
            raise SystemExit(f"Image not found: {args.image}")
        process_one(args.image, pretty=args.pretty, fix=args.fix, debug=args.debug)
    else:
        if not os.path.isdir(args.folder):
            raise SystemExit(f"Folder not found: {args.folder}")
        process_folder(args.folder, pretty=args.pretty, fix=args.fix, debug=args.debug, csv_out=args.csv)

if __name__ == "__main__":
    main()
