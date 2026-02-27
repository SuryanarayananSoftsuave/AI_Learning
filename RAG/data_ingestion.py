import fitz
import pytesseract
from PIL import Image
import io
from pathlib import Path

pdf_path = Path(r"C:\Users\yogeshkannah\Music\AI\Sophia Documentation (1).pdf")

print(pdf_path.exists()) 

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
        break
    return text  


def detect_pdf_type(pdf_path: str) -> dict:
        doc = fitz.open(pdf_path)
        
        results = {
            "total_pages": len(doc),
            "pages_analysis": [],
            "pdf_type": None,
            "recommended_strategy": None
        }
        
        # Sample first 3 pages for efficiency
        sample_pages = min(3, len(doc))
        
        for page_num in range(sample_pages):
            page = doc[page_num]
            
            # Count extractable text characters
            text = page.get_text()
            text_char_count = len(text.strip())
            
            # Count images on this page
            image_list = page.get_images()
            image_count = len(image_list)
            
            # Measure page area covered by text blocks
            blocks = page.get_text("blocks")
            text_coverage = sum(
                (b[2]-b[0]) * (b[3]-b[1]) 
                for b in blocks
            ) / (page.rect.width * page.rect.height)
            
            results["pages_analysis"].append({
                "page": page_num + 1,
                "text_chars": text_char_count,
                "images": image_count,
                "text_coverage_ratio": round(text_coverage, 3)
            })
        
        # Classification logic
        avg_text_chars = sum(
            p["text_chars"] for p in results["pages_analysis"]
        ) / sample_pages
        
        avg_images = sum(
            p["images"] for p in results["pages_analysis"]
        ) / sample_pages
        
        if avg_text_chars > 100:
            results["pdf_type"] = "text_based"
            results["recommended_strategy"] = "direct_extraction"
            
        elif avg_text_chars < 50 and avg_images > 0:
            results["pdf_type"] = "scanned"
            results["recommended_strategy"] = "ocr_required"
            
        else:
            results["pdf_type"] = "mixed"
            results["recommended_strategy"] = "hybrid_extraction"
        
        doc.close()
        return results

extracted_text = detect_pdf_type(pdf_path)      
print(extracted_text)             