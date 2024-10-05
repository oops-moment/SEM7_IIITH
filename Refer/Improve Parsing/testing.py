import fitz  # PyMuPDF


def compare_fitz_ocr_methods(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Iterate through each page of the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Get blocks from get_text("dict")
        fitz_blocks = page.get_text("dict")["blocks"]
        
        # Get extracted data from OCR
        ocr_page = page.get_textpage_ocr()
        ocr_data = ocr_page.extractDICT()
        
        print(f"--- Page {page_num + 1} ---")
        print("Fitz get_text('dict') Blocks:")
        for block in fitz_blocks:
            print(block)
        
        print("\nOCR extractDICT Output:")
        print(ocr_data)
        
        print("\n\nComparison Finished for this Page.\n")
        break
        
# Example usage
pdf_path = 'theory.pdf'
compare_fitz_ocr_methods(pdf_path)