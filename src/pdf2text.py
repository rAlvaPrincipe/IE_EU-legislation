import pdfplumber

input_pdf = "./data/GDPR.pdf"
output_txt = "output.txt"

with pdfplumber.open(input_pdf) as pdf, open(output_txt, "w", encoding="utf-8") as f:
    for page in pdf.pages:
        text = page.extract_text()
        if text: 
            f.write(text + "\n")

