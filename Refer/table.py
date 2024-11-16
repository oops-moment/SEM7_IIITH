import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path

def extract_tables_from_pdf(pdf_path, dpi=300):
    # Convert PDF to images (one image per page)
    pages = convert_from_path(pdf_path, dpi=dpi)
    tables_data = []

    for page_number, page in enumerate(pages):
        # Convert page to a numpy array and then to grayscale for processing
        image = np.array(page)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding for better table structure detection
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)

        # Detect horizontal and vertical lines using morphological operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        
        # Morphological operations to detect lines
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        # Combine the horizontal and vertical lines to create the table structure
        table_structure = cv2.add(horizontal_lines, vertical_lines)

        # Find contours for table cells
        contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        page_data = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Adjust size filtering as needed to detect meaningful cells
            if w > 50 and h > 20:
                cell_img = gray[y:y+h, x:x+w]
                
                # OCR extraction from each cell
                cell_text = pytesseract.image_to_string(cell_img, config="--psm 6").strip()
                
                # Save only non-empty cells
                if cell_text:
                    page_data.append({"cell_text": cell_text, "coordinates": (x, y, w, h)})

        # Append data for the current page if any table data is found
        if page_data:
            tables_data.append({"page": page_number + 1, "table_data": page_data})

    return tables_data

def organize_table_data(tables_data):
    organized_data = []
    for table in tables_data:
        table_data = table['table_data']
        
        # Sort cells by their vertical position (y-coordinate) and then by horizontal position (x-coordinate)
        table_data.sort(key=lambda x: (x['coordinates'][1], x['coordinates'][0]))
        
        # Organize by rows (grouping cells based on their y-coordinate proximity)
        rows = []
        current_row = []
        last_y = table_data[0]['coordinates'][1] if table_data else 0
        
        for cell in table_data:
            x, y, _, _ = cell['coordinates']
            # If the y-coordinate is close to the previous one, it's the same row
            if abs(y - last_y) < 20:  # Threshold for row grouping
                current_row.append(cell['cell_text'])
            else:
                rows.append(current_row)
                current_row = [cell['cell_text']]
            last_y = y
        if current_row:
            rows.append(current_row)  # Append the last row
        
        organized_data.append({
            'page': table['page'],
            'rows': rows
        })
    
    return organized_data
# Example usage
pdf_path = "VERIZON_2022_10K.pdf"
tables = extract_tables_from_pdf(pdf_path)
table_Data_organized = organize_table_data(tables)

# Write output to a text file
output_path = "output.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    for table in tables:
        f.write(f"Page {table['page']} Table Data:\n")
        for cell in table["table_data"]:
            f.write(f"  {cell['cell_text']} (Coordinates: {cell['coordinates']})\n")
        f.write("\n")

output_path = "output_organized.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    for table in table_Data_organized:
        f.write(f"Page {table['page']} Table Data:\n")
        for row in table["rows"]:
            f.write(f"  {row}\n")
        f.write("\n")

print(f"Table data has been written to {output_path}")