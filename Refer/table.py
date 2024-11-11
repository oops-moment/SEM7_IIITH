import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path

def extract_tables_from_pdf(pdf_path, dpi=300):
    # Convert PDF to images (one image per page)
    pages = convert_from_path(pdf_path, dpi=dpi)
    tables_data = []

    for page_number, page in enumerate(pages):
        # Convert page to grayscale for processing
        gray = cv2.cvtColor(np.array(page), cv2.COLOR_BGR2GRAY)
        # Apply threshold to get a binary image
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        # Detect horizontal and vertical lines using morphological operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        
        # Detect lines
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        table_structure = cv2.add(horizontal_lines, vertical_lines)

        # Find contours for table cells
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page_data = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 30 and h > 20:  # Filter out small boxes/noise
                cell_img = gray[y:y+h, x:x+w]
                
                # OCR extraction from each cell
                cell_text = pytesseract.image_to_string(cell_img, config="--psm 6").strip()
                page_data.append({"cell_text": cell_text, "coordinates": (x, y, w, h)})

        tables_data.append({"page": page_number + 1, "table_data": page_data})

    return tables_data

# Example usage
pdf_path = "2407.01219v1.pdf"
tables = extract_tables_from_pdf(pdf_path)
for table in tables:
    print(f"Page {table['page']} Table Data:", table["table_data"])