from pathlib import Path

import pdfplumber


pdf_path = Path(r"F:\Challenger program\主办方资料\无人船自主停靠泊仿真平台及任务-v1.1.pdf")
out_path = Path(r"F:\Challenger program\output\reference_pdf_text.txt")

out_path.parent.mkdir(parents=True, exist_ok=True)
chunks = []
with pdfplumber.open(pdf_path) as pdf:
    for index, page in enumerate(pdf.pages, 1):
        chunks.append(f"\n--- page {index} ---\n")
        chunks.append(page.extract_text() or "")

out_path.write_text("\n".join(chunks), encoding="utf-8")
print(out_path)
