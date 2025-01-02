import fitz

def get_page_range(pdf_path, out_path, start_page, end_page):
    """
    Extracts text from a range of pages from a PDF file using PyMuPDF.
    """
    document = fitz.open(pdf_path)

    # Create a new PDF for the output
    output_document = fitz.open()
    text = ""
    # Extract pages from the specified range
    for page_num in range(start_page, end_page + 1):
        page = document.load_page(page_num)
        output_document.insert_pdf(document, from_page=page_num, to_page=page_num)

    # Save the extracted pages to the output file
    output_document.save(out_path)

    print(f"Pages {start_page + 1} to {end_page + 1} have been extracted and saved to '{out_path}'.")


def parse_pdf_sections(document):
    """
    Parses the sections of a PDF document using PyMuPDF. The main sections and sub sections are identified based on the font size of the text.
    This is not for general use and is specific to the format of the PDF document used in this example.
    DONOT use this for production code. This is just for demonstration purposes.

    All paragraphs of a section are kept in same chunk to preserve the context. Code snippents and images are not included in the output.
    """
    parsed_dict = {}
    main_section_key = ""
    section_key = ""
    for page in range(document.page_count):
        #print(f"Page {page + 1}")
        text_dict = document[page].get_text("dict")

        for block in text_dict.get("blocks", []):
            #print(f"Block {block['number']}")

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    #print(span)
                    text = span.get("text", "").strip()

                    if len(text) > 0:
                        if not text.isnumeric() and text.strip().lower() not in ["Chapter 1: Introduction to Apache Spark: A Unified Analytics Engine".lower(), "CHAPTER 1".lower()]:
                            #print(text)
                            font_size = span.get("size", 0)
                            if font_size > 17 and font_size < 19:
                                #print(text) 
                                main_section_key = text
                                parsed_dict[main_section_key] = {}
                                section_key = ""
                                sub_section_key = ""
                            
                            if font_size > 15 and font_size < 16:
                                #print("\t", text)
                                section_key = text
                                sub_section_key = ""
                                if main_section_key:
                                    parsed_dict[main_section_key][section_key] = {}

                            if font_size > 11 and font_size < 12:
                                #print("\t\t", text)
                                sub_section_key = text
                                if main_section_key and section_key:
                                    parsed_dict[main_section_key][section_key][sub_section_key] = []

                            if font_size > 10 and font_size < 11:
                                #print("\t\t\t", text)
                                if main_section_key and section_key and sub_section_key:
                                    parsed_dict[main_section_key][section_key][sub_section_key].append(text)
                                
                                if main_section_key and section_key and not sub_section_key:
                                    sub_section_key = "content"
                                    parsed_dict[main_section_key][section_key][sub_section_key] = [text]

    for main_section_key in parsed_dict:
        for section_key in parsed_dict[main_section_key]:
            for sub_section_key in parsed_dict[main_section_key][section_key]:
                content = parsed_dict[main_section_key][section_key][sub_section_key]
                parsed_dict[main_section_key][section_key][sub_section_key] = " ".join(content)
                
    return parsed_dict


def fixed_size_chunking(text, metadata, chunk_size, overlap, char=False):
    """
    Splits the input text into chunks of a fixed size with optional overlap.

    Parameters:
    text (str): The input text to be chunked.
    chunk_size (int): The size of each chunk.
    overlap (int): The number of overlapping elements between consecutive chunks.
    char (bool): If True, chunk by characters. If False, chunk by words. Default is False.

    Returns:
    list: A list of text chunks.
    """

    if char:
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    else:
        text = text.split()
        return [ metadata + text[i:i+chunk_size] for i in range(0, len(text) - len(metadata), chunk_size - overlap)]