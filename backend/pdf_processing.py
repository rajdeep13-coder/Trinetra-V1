import io
import logging
import re
from typing import Dict, List, Optional, Tuple
import shutil

import fitz  
from PIL import Image
import numpy as np
import pandas as pd
from google import generativeai as genai
from dotenv import load_dotenv
import os


TESSERACT_AVAILABLE = shutil.which('tesseract') is not None
if TESSERACT_AVAILABLE:
    import pytesseract

load_dotenv()


genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PDF_PROMPT = """
Process this PDF to clean and enhance it. Follow these steps:

1. **Input**: User uploads PDF file
2. **Cleaning Tasks**:
   - Auto-rotate pages to correct orientation
   - Remove visual noise/artifacts
   - Fix text formatting and alignment
   - Remove empty pages
   - Preserve original content structure
3. **Output**:
   - Return cleaned PDF
   - Include brief report:
     "Cleaned document.pdf: Rotated pages, removed artifacts, fixed formatting"
4. **Error Handling**:
   - If unrecoverable corruption: "Error: Could not process document.pdf"
"""


QUALITY_CHECK_CATEGORIES = {
    "missing_content": "Check for pages with missing or incomplete content",
    "duplicate_content": "Check for duplicate pages or content blocks",
    "formatting_issues": "Check for inconsistent formatting, fonts, or layouts",
    "text_quality": "Check for OCR errors, broken text, or poor readability",
    "metadata_completeness": "Check for missing or incomplete document metadata",
    "image_quality": "Check for low-resolution or corrupted images",
    "structure_issues": "Check for broken links, bookmarks, or document structure"
}

def generate_pdf_prep_code(quality_issues: Dict, user_instructions: str = "") -> str:
    """Generate code for PDF preparation based on quality issues and user instructions."""
    try:
       
        prompt = f"Given these PDF quality issues:\n{quality_issues}\n"
        if user_instructions:
            prompt += f"\nAdditional user instructions:\n{user_instructions}\n"
        prompt += "\nGenerate Python code to clean and process the PDF using PyMuPDF (fitz) that:"
        prompt += "\n1. Addresses the identified quality issues"
        prompt += "\n2. Follows user instructions if provided"
        prompt += "\n3. Returns the processed PDF and a report of changes made"

      
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
      
        return response.text

    except Exception as e:
        logger.error(f"Error generating PDF prep code: {str(e)}")
        return ""

def execute_pdf_prep_code(code: str, pdf_file: bytes) -> Tuple[bytes, Dict]:
    """Execute generated PDF preparation code."""
    try:
       
        namespace = {
            'fitz': fitz,
            'np': np,
            'Image': Image,
            'pdf_file': pdf_file,
            'logger': logger
        }
        
      
        exec(code, namespace)
        
   
        result = namespace.get('result', pdf_file)
        report = namespace.get('report', {'errors': ['No report generated']})
        
        return result, report
        
    except Exception as e:
        logger.error(f"Error executing PDF prep code: {str(e)}")
        return pdf_file, {'errors': [str(e)]}

def process_pdf(pdf_file: bytes, selected_operations: Dict = None) -> Tuple[bytes, Dict]:
    """Process a PDF file and return the cleaned version with a report."""
    try:
       
        quality_issues = check_pdf_quality(pdf_file)
        
      
        if selected_operations:
            cleaning_code = generate_pdf_prep_code(quality_issues, str(selected_operations))
            if cleaning_code:
                return execute_pdf_prep_code(cleaning_code, pdf_file)
    except Exception as e:
        logger.error(f"Error in PDF preparation: {str(e)}")
        return pdf_file, {"errors": [str(e)]}

    try:
       
        pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
        total_pages = len(pdf_document)
        report = {
            "total_pages": total_pages,
            "repaired_pages": 0,
            "ocr_pages": 0,
            "removed_pages": 0,
            "avg_ocr_confidence": 0.0,
            "errors": []
        }

      
        if pdf_document.is_encrypted:
            report["errors"].append("Error: PDF is password-protected")
            return pdf_file, report

       
        cleaned_pdf = fitz.open()
        total_ocr_confidence = 0.0

        for page_num in range(total_pages):
            page = pdf_document[page_num]
            
          
            if page.get_text().strip() == "":
                pix = page.get_pixmap()
                if np.mean(pix.samples) > 250:  
                    report["removed_pages"] += 1
                    continue

           
            if len(page.get_text().strip()) < 100:  
                if TESSERACT_AVAILABLE:
                    try:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                     
                        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                        confidence = float(np.mean([conf for conf in ocr_data["conf"] if conf != -1]))
                        
                        if confidence > 0:
                            total_ocr_confidence += confidence
                            report["ocr_pages"] += 1
                            
                           
                            new_page = cleaned_pdf.new_page(width=page.rect.width, height=page.rect.height)
                            new_page.insert_text((50, 50), ocr_data["text"])
                        else:
                           
                            cleaned_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num + 1}: {str(e)}")
                        cleaned_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
                else:
                   
                    logger.info(f"Tesseract not available, using PyMuPDF text extraction for page {page_num + 1}")
                    cleaned_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
            else:
               
                cleaned_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
                report["repaired_pages"] += 1

        
        if report["ocr_pages"] > 0:
            report["avg_ocr_confidence"] = round(total_ocr_confidence / report["ocr_pages"], 2)

       
        output_buffer = io.BytesIO()
        cleaned_pdf.save(output_buffer)
        cleaned_pdf.close()
        pdf_document.close()

        return output_buffer.getvalue(), report

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        report["errors"].append(f"Error: {str(e)}")
        return pdf_file, report

def check_pdf_quality(pdf_file: bytes) -> Dict:
    """Check PDF quality and return issues found."""
    try:
        pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
        total_pages = len(pdf_document)
        
        issues = {
            "issue_detected": False,
            "results_df": [],
            "recommendation": "",
            "analytics": {
                "total_pages": total_pages,
                "text_content": {},
                "formatting": {},
                "images": {},
                "metadata": {}
            }
        }
        
        quality_issues = []
        text_analytics = {"total_words": 0, "avg_words_per_page": 0}
        format_analytics = {"fonts": set(), "alignments": set()}
        image_analytics = {"total_images": 0, "avg_dpi": 0}
        
      
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            text_content = page.get_text().strip()
            
           
            words = text_content.split()
            text_analytics["total_words"] += len(words)
            
         
            if not text_content:
                quality_issues.append({
                    "Issue Type": "Missing Content",
                    "Page Number": page_num + 1,
                    "Details": "Page contains no text",
                    "Severity": "High",
                    "Can Fix": True
                })
            
          
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        for s in l["spans"]:
                            format_analytics["fonts"].add(s["font"])
                            format_analytics["alignments"].add(round(b["bbox"][0]))
            
            images = page.get_images()
            image_analytics["total_images"] += len(images)
            for img in images:
                if img[2] < 150:
                    quality_issues.append({
                        "Issue Type": "Low Image Quality",
                        "Page Number": page_num + 1,
                        "Details": f"Image resolution: {img[2]} DPI",
                        "Severity": "High",
                        "Can Fix": True
                    })
                image_analytics["avg_dpi"] += img[2]
        
     
        text_analytics["avg_words_per_page"] = text_analytics["total_words"] / total_pages if total_pages > 0 else 0
        image_analytics["avg_dpi"] = image_analytics["avg_dpi"] / image_analytics["total_images"] if image_analytics["total_images"] > 0 else 0
        
       
        issues["analytics"]["text_content"] = text_analytics
        issues["analytics"]["formatting"] = {
            "unique_fonts": len(format_analytics["fonts"]),
            "alignment_variations": len(format_analytics["alignments"])
        }
        issues["analytics"]["images"] = image_analytics
        
       
        metadata = pdf_document.metadata
        required_metadata = ['title', 'author', 'subject', 'keywords']
        missing_metadata = [field for field in required_metadata if not metadata.get(field)]
        if missing_metadata:
            quality_issues.append({
                "Issue Type": "Incomplete Metadata",
                "Page Number": "N/A",
                "Details": f"Missing metadata fields: {', '.join(missing_metadata)}",
                "Severity": "Low",
                "Can Fix": True
            })
        
      
        if quality_issues:
            issues["issue_detected"] = True
            issues["results_df"] = pd.DataFrame(quality_issues)
            issues["recommendation"] = "PDF quality issues detected. Select issues to fix and click 'Process PDF' to improve quality."
        
        pdf_document.close()
        return issues
        
    except Exception as e:
        logger.error(f"Error checking PDF quality: {str(e)}")
        return {
            "issue_detected": True,
            "results_df": pd.DataFrame([{
                "Issue Type": "Processing Error",
                "Page Number": "N/A",
                "Details": str(e),
                "Severity": "High",
                "Can Fix": False
            }]),
            "recommendation": "Error occurred while checking PDF quality.",
            "analytics": {}
        }
