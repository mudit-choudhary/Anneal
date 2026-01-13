import fitz
import os
from config import PARSED_DIR, REGISTRY_URL
import requests


def save_text(text, file_name):

    file_path = PARSED_DIR + "/" + file_name + ".txt"

    with open(file_path, mode="w") as f:
        f.write(text)


def parse(pdf_path: str=""):
    if pdf_path=="":
        return {"text": "not found"}
    
    file_name_with_extension = pdf_path.split('/')[-1]
    file_name = file_name_with_extension.split('.')[0]

    doc = fitz.open(pdf_path)

    total_pages = len(doc)
    # batch_size = 1000
    # for start_page in range(0, total_pages, batch_size):
        # end_page = min(start_page + batch_size, total_pages)
    batch_text = ""
    for page_number in range(0, total_pages):
        page = doc[page_number]
        text = page.get_text()
        # blocks = page.get_text('blocks')
        text = text.strip()
        # images = page.get_images()
        # for i, img in enumerate(images):
        #     xref = img[0]
        #     pix = fitz.Pixmap(doc, xref)
        #     if pix.n < 5:  # this is GRAY or RGB
        #         pix.save(f'image_{i}_page_{page_number}.png')
        #     else:  # CMYK: convert to RGB first
        #         pix1 = fitz.Pixmap(fitz.csRGB, pix)
        #         pix1.save(f'image_{i}_page_{page_number}.png')
        #         pix1 = None
        #     pix = None
        # print(blocks)
        # continue
        if text[-1] == '.':
            text = text + '\n'
        batch_text = batch_text + f"\n\nPage: {page_number+1}\n" + text
    # break
    save_text(batch_text, file_name)


    data = {"filename": file_name,
            "status": "parsed"}
                
    response = requests.post(
        f"{REGISTRY_URL}/update_status",
        json=data,
        timeout=20
    )
    response_data = response.json()
    if not response_data['success']:
        raise
            
    print("\n", "Filename: ", file_name_with_extension, "File Length: ", len(doc),"\n", "="*50)


# if __name__ == '__main__':
#     pdf_path = '/home/mudit/Desktop/RAGSetup/data/raw_pdfs/Completion_by_Comprehension_Guiding_Code_Generation_with_Multi-Granularity_Understanding.pdf'
#     parse(pdf_path)