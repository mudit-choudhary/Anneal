import re
import json
import requests

from config import REGISTRY_URL


def process_pdf_txt(txt_file, output_txt, output_json):
    filename = txt_file.split('/')[-1]
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean page markers and odd-page title repetitions
        content = re.sub(r'Page:\s*\d+\n[^.\n]*?\n\d+\n', '', content, flags=re.MULTILINE)  # Page:3 title 3
        content = re.sub(r'(?m)^Page\s*\d+\s*$', '', content)  # Standalone Page X
        
        lines = content.split('\n')
        paragraphs = []
        current_para = []
        current_page = 1
        word_count = 0
        
        title_pattern = r'Completion by Comprehension.*Understanding'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            words = len(line.split())
            
            # Page change (not paragraph break)
            if re.match(r'Page\s*\d+', line):
                if current_para:
                    para_text = ' '.join(current_para).strip()
                    if para_text and len(para_text.split()) > 5:  # Min 5 words
                        paragraphs.append({
                            'text': para_text,
                            'page': current_page,
                            'word_count': len(para_text.split())
                        })
                    current_para = []
                continue
            
            # Skip title repetitions on odd pages
            if re.search(title_pattern, line):
                continue
                
            # Paragraph break conditions (prioritized)
            is_break = False
            
            # 1. Clear section headers
            if (re.match(r'^\d+[\.\s]', line) or  # 3.1, 5.2
                re.match(r'^[A-Z][a-z]+:', line) or  # RQ1:
                re.match(r'^Fig\.', line) or
                re.match(r'^Table\s', line)):
                is_break = True
            
            # 2. Short lines (<20 words) after sentence-ending line
            elif words < 20 and current_para and current_para[-1].rstrip().endswith('.'):
                is_break = True
                
            # 3. Very short lines that look like captions/headers
            elif words < 10:
                is_break = True
                
            if is_break and current_para:
                para_text = ' '.join(current_para).strip()
                if len(para_text.split()) > 3:
                    paragraphs.append({
                        'text': para_text,
                        'page': current_page,
                        'word_count': len(para_text.split())
                    })
                current_para = []
            
            current_para.append(line)
            word_count += words
        
        # Final paragraph
        if current_para:
            para_text = ' '.join(current_para).strip()
            if len(para_text.split()) > 3:
                paragraphs.append({
                    'text': para_text,
                    'page': current_page,
                    'word_count': len(para_text.split())
                })
        
        # Save formatted text
        with open(output_txt, 'w', encoding='utf-8') as f:
            for i, para in enumerate(paragraphs):
                f.write(para['text'])
                f.write('\n\n')
        
        # Save JSON with metadata
        json_data = {
            'filename': txt_file,
            'total_paragraphs': len(paragraphs),
            'total_words': sum(p['word_count'] for p in paragraphs),
            'paragraphs': paragraphs
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        file = filename.split('.')[0]
        data = {"filename": file,
                "status": "processed"}
                    
        response = requests.post(
            f"{REGISTRY_URL}/update_status",
            json=data,
            timeout=20
        )
        response_data = response.json()
        if not response_data['success']:
            raise
        print(f"Processed: {len(paragraphs)} paragraphs, {json_data['total_words']} words")
        return True
    
    except Exception as e:
        print(f"Error Processing File: {filename}", "\n", e)
        return False