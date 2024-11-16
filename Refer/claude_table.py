import pdfplumber
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import re
import os

class PDFTableExtractor:
    def __init__(self):
        """Initialize the PDF Table Extractor"""
        self.current_page = 0
        self.previous_table_structure = None

    def extract_tables(self, pdf_path: str) -> Dict[int, List[pd.DataFrame]]:
        """
        Extract tables from all pages of a PDF file
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict[int, List[pd.DataFrame]]: Dictionary with page numbers as keys and list of tables as values
        """
        all_tables = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                self.current_page = page_num + 1
                tables = self._process_page(page)
                if tables:
                    all_tables[page_num + 1] = tables
                    
        return all_tables
    
    
    def _process_page(self, page) -> List[pd.DataFrame]:
        """
        Process a single page and extract tables.
        
        Args:
            page: pdfplumber page object
            
        Returns:
            List[pd.DataFrame]: List of tables found on the page
        """
        tables = []
        settings_list = [
            {'vertical_strategy': 'text', 'horizontal_strategy': 'text'},
            {'vertical_strategy': 'lines', 'horizontal_strategy': 'lines'},
        ]

        for settings in settings_list:
            try:
                extracted = page.extract_tables(table_settings=settings)
                if extracted is None:
                    print(f"Warning: No tables found on page {self.current_page} with settings {settings}")
                    continue  # Skip to the next setting

                for table in extracted:
                    if table:  # Check if the extracted table is not empty
                        processed_table = self._process_table(table)
                        if not processed_table.empty:
                            tables.append(processed_table)

            except Exception as e:
                print(f"Error processing page {self.current_page} with settings {settings}: {e}")
                continue

        return tables if tables else None  # Return None if no tables were processed

    def _process_table(self, table: List[List[str]]) -> pd.DataFrame:
        """
        Process and clean extracted table data
        
        Args:
            table (List[List[str]]): Raw table data
            
        Returns:
            pd.DataFrame: Processed table
        """
        # Convert to DataFrame
        df = pd.DataFrame(table)
        
        # Clean the data
        df = df.replace('', np.nan)
        df = df.replace(r'^\s*$', np.nan, regex=True)
        
        # Handle missing headers
        if df.iloc[0].isna().all() or (df.iloc[0] == df.iloc[0].iloc[0]).all():
            # Generate column names if headers are missing
            df.columns = [f'Column_{i}' for i in range(len(df.columns))]
        else:
            # Use first row as headers if present
            df.columns = df.iloc[0]
            df = df.iloc[1:]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Try to infer numeric columns
        for col in df.columns:
            try:
                # Remove common currency and thousand separators
                cleaned_values = df[col].str.replace('$', '').str.replace(',', '')
                if pd.to_numeric(cleaned_values, errors='coerce').notna().all():
                    df[col] = pd.to_numeric(cleaned_values)
            except:
                continue
        
        return df

    def _detect_table_structure(self, text_elements: List[Dict]) -> Tuple[List[float], List[float]]:
        """
        Detect table structure from text elements
        
        Args:
            text_elements (List[Dict]): List of text elements with positions
            
        Returns:
            Tuple[List[float], List[float]]: Detected column and row positions
        """
        x_positions = sorted(set([elem['x0'] for elem in text_elements]))
        y_positions = sorted(set([elem['top'] for elem in text_elements]))
        
        # Filter out noise in positions
        x_positions = self._cluster_positions(x_positions)
        y_positions = self._cluster_positions(y_positions)
        
        return x_positions, y_positions

    def _cluster_positions(self, positions: List[float], threshold: float = 5.0) -> List[float]:
        """
        Cluster nearby positions to handle slight misalignments
        
        Args:
            positions (List[float]): List of positions
            threshold (float): Distance threshold for clustering
            
        Returns:
            List[float]: Clustered positions
        """
        if not positions:
            return []
            
        clusters = [[positions[0]]]
        
        for pos in positions[1:]:
            if pos - clusters[-1][-1] < threshold:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
                
        return [sum(cluster) / len(cluster) for cluster in clusters]

    def extract_structured_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract structured data from the table
        
        Args:
            df (pd.DataFrame): Input table
            
        Returns:
            Dict[str, Any]: Structured data
        """
        structured_data = {}
        
        # Try to identify key-value pairs
        for idx, row in df.iterrows():
            key = row.iloc[0]
            if isinstance(key, str) and not pd.isna(key):
                values = row.iloc[1:].dropna().tolist()
                if values:
                    structured_data[key.strip()] = values
                    
        return structured_data

def main():
    """
    Example usage of the PDF Table Extractor
    """
    # Initialize the extractor
    extractor = PDFTableExtractor()
    
    # Extract tables from PDF
    pdf_path = "VERIZON_2022_10K.pdf"
    tables_by_page = extractor.extract_tables(pdf_path)
    
    # Create a directory for output files
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the extracted tables
    for page_num, tables in tables_by_page.items():
        print(f"\nPage {page_num}:")
        for i, table in enumerate(tables):
            # Extract structured data
            structured_data = extractor.extract_structured_data(table)
            print("\nStructured Data:")
            print(structured_data)
            
            # Write structured data to a text file
            output_file = os.path.join(output_dir, f"page_{page_num}_table_{i + 1}.txt")
            with open(output_file, "w") as f:
                f.write(f"Page {page_num}, Table {i + 1}\n")
                for key, values in structured_data.items():
                    f.write(f"{key}: {', '.join(map(str, values))}\n")

if __name__ == "__main__":
    main()