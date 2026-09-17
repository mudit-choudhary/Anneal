import os
from pdf2image import convert_from_path
from tqdm import tqdm  # pip install tqdm
import time

# Configuration
SOURCE_DIR = "/media/mudit/DarkD'wine/ResearchPapersYOLO_FT/PDFs"
DEST_DIR = "/media/mudit/DarkD'wine/ResearchPapersYOLO_FT/images"
DPI = 300  # 300 is standard for OCR/Annotation. Use 600 for extremely small text.

def main():
    # 1. Ensure destination exists
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created directory: {DEST_DIR}")

    # 2. Get list of PDF files
    pdf_files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {SOURCE_DIR}")
        return

    print(f"Found {len(pdf_files)} PDFs. Starting conversion...")
    counter = 25
    cnt = 0
    # 3. Process each PDF
    for pdf_file in tqdm(pdf_files, desc="Converting PDFs"):
        pdf_path = os.path.join(SOURCE_DIR, pdf_file)
        pdf_name = os.path.splitext(pdf_file)[0]
        
        try:
            # Convert PDF to list of images
            # fmt='jpeg' produces smaller files than png, good for photos/scans
            pages = convert_from_path(pdf_path, dpi=DPI, fmt='jpeg')
            
            # Save each page
            for i, page in enumerate(pages):
                # Naming format: PaperName_page_01.jpg
                image_filename = f"{pdf_name}_page_{i+1:02d}.jpg"
                image_path = os.path.join(DEST_DIR, image_filename)
                
                page.save(image_path, 'JPEG')
            cnt += 1
            if cnt == counter:
                time.sleep(7)
                cnt = 0
        except Exception as e:
            print(f"\nError converting {pdf_file}: {e}")

    print(f"\nSuccess! All images saved to: {DEST_DIR}")

if __name__ == "__main__":
    main()