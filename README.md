# 🛂 Passport OCR Scanner

An AI-powered tool that detects and extracts **Machine Readable Zone (MRZ)** data from passport images — focused on **European TD3-type passports**.  
Built with [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and custom Python parsing logic.

---

## ✨ Features
- Detects MRZ zones automatically from full passport images  
- Parses and validates **TD3** MRZ lines (2 × 44-char)  
- Performs checksum validation (passport number, birth date, expiry date)  
- `--fix` flag repairs common OCR confusions (`O→0`, `I→1`, etc.)  
- `--debug` mode saves overlay images for visualization  
- Modular architecture (`services/`, `parsers/`, `main.py`) for future expansion

---

## 🧩 Project Structure
```
passport-ocr-scanner/
├── main.py                # CLI entry point
├── services/
│   └── ocr_service.py     # OCR, detection, and debug overlay
├── parsers/
│   └── td3.py             # TD3 MRZ parsing & validation
├── tests/
│   └── run_one.py         # Single-image testing script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/marinos421/passport-ocr-scanner.git
cd passport-ocr-scanner
python -m venv .venv
.\.venv\Scripts\activate    # on Windows
pip install -r requirements.txt
```

---

## 🚀 Usage

Run OCR on a folder of passport images:

```bash
python main.py --folder ".\samples" --csv ".\tmp_one\results.csv" --debug --fix
```

| Flag | Description |
|------|--------------|
| `--folder` | Folder containing input passport images |
| `--image`  | (Optional) Single image path |
| `--csv`    | Output CSV file path |
| `--debug`  | Saves MRZ overlay images |
| `--fix`    | Enables character auto-correction for OCR mistakes |

---

## ✅ Current Status
The project currently **successfully parses most TD3 passports (3 / 6 tested)**.  
Low-contrast or non-standard MRZ fonts may still fail — future updates will include improved preprocessing and line selection.

---

## 🧠 Planned Improvements
- TD1 & TD2 format support (ID cards, older passports)  
- Better preprocessing for blurry/low-light images  
- Perspective correction & auto-rotation  
- Fallback text extraction for printed fields  

---

## 💡 Tech Stack
- **Python 3.10**
- **PaddleOCR + PaddlePaddle**
- **OpenCV / NumPy**

---

## 🧑‍💻 Author
**Marinos Aristeidou**  
Computer Engineering & Informatics, University of Ioannina 🇬🇷  
[GitHub Profile](https://github.com/marinos421) • [LinkedIn](https://linkedin.com/in/marinosaristeidou)

---

## 📜 License
MIT License
