# tests/run_one.py
import argparse
import json
import sys
from services.ocr_service import init_ocr, load_image_bgr, extract_mrz_two_lines
from parsers.td3 import parse_td3

def run_one(image_path: str, pretty: bool = True) -> int:
    ocr = init_ocr()
    img = load_image_bgr(image_path)
    L1, L2 = extract_mrz_two_lines(img, ocr)
    data = parse_td3(L1, L2)

    # Nice stdout
    def flag(ok): return "✅" if ok else "❌"
    print("\n=== RESULT ===")
    print(f"file:              {image_path}")
    print(f"doc_type:          {data['doc_type']}")
    print(f"issuer(country):   {data['country']}")
    print(f"nationality:       {data['nationality']}")
    print(f"surname:           {data['surname']}")
    print(f"given_names:       {data['given_names']}")
    print(f"passport_number:   {data['passport_number']}  {flag(data['passport_number_checksum_ok'])}")
    print(f"birth_date:        {data['birth_date']}       {flag(data['birth_date_checksum_ok'])}")
    print(f"sex:               {data['sex']}")
    print(f"expiry_date:       {data['expiry_date']}      {flag(data['expiry_date_checksum_ok'])}")
    print(f"personal_number:   {data['personal_number']}  {flag(data['personal_number_checksum_ok'])}")
    print(f"final_checksum_ok: {flag(data['final_checksum_ok'])}")
    print("\nJSON:")
    print(json.dumps(data, indent=2 if pretty else None, ensure_ascii=False))
    return 0 if data["final_checksum_ok"] else 1

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Test a single passport image (TD3).")
    ap.add_argument("--image", required=True, help="Path to image")
    ap.add_argument("--compact", action="store_true", help="Compact JSON")
    args = ap.parse_args()
    sys.exit(run_one(args.image, pretty=not args.compact))
