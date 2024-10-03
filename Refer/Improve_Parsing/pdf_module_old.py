# TODO breakdown file into smaller chunks
from typing import List, Counter
import fitz
from tqdm import tqdm
import json
import os
import sys
import re
import glob
import argparse
from collections import defaultdict
import re
import copy
from time import time
# import concurrent.futures
from celery import group as celery_group
from celery.result import allow_join_result
from billiard import Pool, cpu_count
from itertools import cycle, islice
# from multiprocessing import Pool, cpu_count

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from header_footer_detection import remove_header_footer
    from document_hierarchy import create_hierarchy
    from pdf_table_utils import *
    from table_to_text_task import table2text_task
    # from app.core.celery_app import celery_app
    sys.path.append('../../../app')
else:
    from .header_footer_detection import remove_header_footer
    from .document_hierarchy import create_hierarchy
    from .pdf_table_utils import *
    from .table_to_text_task import table2text_task

from app.core.celery_app import celery_app
from app.schemas.table_to_text import TableToTextRequest

first_line_end_thesh = 0.8

block_miss = defaultdict(list)
block_double = defaultdict(list)


class CustomEncoder(json.JSONEncoder):

    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)


def remove_postscript_and_style(font_name, suffix="MT", styles=["Bold", "Italic", "BoldItalic", "Regular"]):
    cleaned_font_name = font_name.replace(f"-{suffix}", "").replace(suffix, "")

    for style in styles:
        cleaned_font_name = cleaned_font_name.replace(f"-{style}", "").replace(style, "")

    cleaned_font_name = cleaned_font_name.strip()

    return cleaned_font_name


def compare_span(span1, span2):
    # return all([span1[key] == span2[key] for key in ['font', 'flags', 'color']]) and abs(span1["size"] - span2["size"]) < 0.3
    return remove_postscript_and_style(span1["font"]) == remove_postscript_and_style(span2["font"]) and abs(span1["size"] - span2["size"]) < 0.3


def font_hash(span):
    # return f'{span["font"]}|{round(span["size"], 1)}|{span["flags"]}|{span["color"]}'
    return f'{remove_postscript_and_style(span["font"])}|{round(span["size"], 1)}'


def merge(block1, block2):
    lines = block1["lines"].copy()
    lines.extend(block2["lines"])
    block2["lines"] = lines
    return block2


def coherent_check(line1, line2, strict=False):
    line_1_ends_with_punctuation = line1['spans'][-1]['text'].strip()[-1] in ['.', '?', '!']
    line_2_starts_with_capital = line2['spans'][0]['text'][0].lower() != line2['spans'][0]['text'][0]
    if strict:
        return not line_1_ends_with_punctuation and not line_2_starts_with_capital
    else:
        return not line_1_ends_with_punctuation or not line_2_starts_with_capital


def is_same_line(line1, line2):
    line1, line2 = (line2, line1) if line2['bbox'][1] < line1['bbox'][1] else (line1, line2)
    if line2['bbox'][1] < line1['bbox'][3]:
        return (line1['bbox'][3] - line2['bbox'][1]) / (line2['bbox'][3] - line1['bbox'][1]) > 0.8


def check_merge(block1, block2, line_gaps):
    if block1['table_info'] or block2['table_info']:
        return block1['table_info'] and block2['table_info'] and block1['table_info'] == block2['table_info']

    # ensure the last line of first block doesn't end before the threshold
    total_width = block1["bbox"][2] - block1["bbox"][0]
    line1 = block1["lines"][-1]
    line2 = block2["lines"][0]
    if (line1["bbox"][2] - block1["bbox"][0]) / total_width < first_line_end_thesh:
        if not is_same_line(line1, line2):
            return False

    common_span = False
    common_spans = []

    # compare spans of same font in the text
    for span1 in line1["spans"]:
        span1_hash = font_hash(span1)
        for span2 in line2["spans"]:
            span2_hash = font_hash(span2)
            if compare_span(span1, span2) and not block2["list_item_start"]:
                gap = line2["bbox"][1] - line1["bbox"][3]
                gap = round(gap, 1)
                # joining blocks from different page
                if line1["word_bbox"][0][4] != line2["word_bbox"][0][4]:
                    return coherent_check(line1, line2, strict=True)
                # joining black from different column
                elif line1["bbox"][3] > line2["bbox"][1]:
                    return True if is_same_line(line1, line2) else coherent_check(line1, line2, strict=True)
                # # join block with fonts not merged before, ex in equidistance docs
                # elif span1_hash not in line_gaps:
                #     return True
                # join if block gap is same as line gap
                elif span1_hash in line_gaps and any([abs(line_gap - gap) < 0.2 for line_gap in line_gaps[span1_hash]]):
                    return True
                common_span = True
                common_spans.append((f"{span1_hash}~{gap}"))

    common_spans = list(set(common_spans))

    # all lines are same distant, so we keep track of missed gaps in two queues
    if common_span and font_hash(line1["spans"][-1]) == font_hash(line2["spans"][0]):
        for span_hash_gap in common_spans:
            span_hash, gap = span_hash_gap.split('~')
            if float(span_hash.split('|')[1]) * 1.5 > float(gap):
                if len(block1['lines']) == 1 and len(block2['lines']) == 1:
                    for gaps_offset in range(-3, 4):
                        new_span_hash_gap = f"{span_hash}~{round(float(gap)+(gaps_offset/10),1)}"
                        if block1['idx'] in block_miss[new_span_hash_gap]:
                            block_double[new_span_hash_gap].append(block1['idx'])
                        else:
                            block_miss[new_span_hash_gap].append(block1['idx'])
                        if block2['idx'] in block_miss[new_span_hash_gap]:
                            block_double[new_span_hash_gap].append(block2['idx'])
                        else:
                            block_miss[new_span_hash_gap].append(block2['idx'])

    if not common_span:
        return is_same_line(line1, line2)

    return False


def should_divide(line1, line2, block, line_gaps, common_hashes):
    total_width = block["bbox"][2] - block["bbox"][0]
    gap = round(line2["bbox"][1] - line1["bbox"][3], 1)

    # divide if no common spans
    if not any([font_hash(span) in common_hashes for span in line2["spans"]]):
        return True

    # divide if gap is more than 2 times the smallest gap
    for span_hash in common_hashes:
        if span_hash in line_gaps and len(line_gaps[span_hash].keys()) > 1:
            all_gaps = sorted(line_gaps[span_hash].keys())
            if gap >= 2 * all_gaps[0]:
                return True

    # detect bullet points
    if detect_list_item(line2):
        return True

    if coherent_check(line1, line2):
        return False
    if (line1["bbox"][2] - block["bbox"][0]) / total_width > first_line_end_thesh:
        return False
    if (line2["bbox"][2] - block["bbox"][0]) / total_width < first_line_end_thesh:
        return False
    return True


def check_divide(block):
    blocks = []
    lines = []

    line_gaps = {}

    for i, line in enumerate(block["lines"][:-1]):
        gap = round(block["lines"][i + 1]["bbox"][1] - line["bbox"][3], 1)

        if gap > 0:
            line1_hash = list(set([font_hash(span) for span in line["spans"]]))
            line2_hash = list(set([font_hash(span) for span in block["lines"][i + 1]["spans"]]))
            common_hashes = [span_hash for span_hash in line1_hash if span_hash in line2_hash]

            for span_hash in common_hashes:
                line_gaps[span_hash] = line_gaps[span_hash] if span_hash in line_gaps else defaultdict(int)
                line_gaps[span_hash][gap] += 1

    common_hashes = None
    for i, line in enumerate(block["lines"]):
        common_hashes = list(set([font_hash(span) for span in line["spans"] if not common_hashes or font_hash(span) in common_hashes]))
        lines.append(line)
        if i < len(block["lines"]) - 1 and should_divide(line, block["lines"][i + 1], block, line_gaps, common_hashes):
            blocks.append(block.copy())
            blocks[-1]["lines"] = lines
            blocks[-1]["common_hashes"] = common_hashes
            lines = []
            common_hashes = None

    if lines:
        blocks.append(block.copy())
        blocks[-1]["lines"] = lines
        blocks[-1]["common_hashes"] = common_hashes

    for i, block in enumerate(blocks):
        if i > 0:
            blocks[i]['list_item_start'] = False
    return blocks


def detect_list_item(line):
    initial_token = line["tokens"][0]
    return re.match(
        r"^\s*\(?([a-z]|[A-Z]|\d{1,3}|(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})|(xc|xl|l?x{0,3})(ix|iv|b?i{0,3}))(\.|\)){1,2}\s*$",
        initial_token,
    )


def remove_list_items(block):
    if not block["lines"] or not block["lines"][0]["tokens"]:
        return block

    if detect_list_item(block["lines"][0]):
        block["list_item_start"] = True
        block["lines"][0]["tokens"] = block["lines"][0]["tokens"][1:]
        block["lines"][0]["word_bbox"] = block["lines"][0]["word_bbox"][1:]
    return block


def remove_empty_lines(block):
    for line_no, line in enumerate(block['lines']):
        block['lines'][line_no]["spans"] = [span for span in line["spans"] if span["text"].strip()]
    block["lines"] = [line for line in block["lines"] if line["tokens"] and line["spans"]]
    return block


def clean_blocks(blocks):
    """
    TODO: Remove Header and Footers from page_data
    TODO: detect tables and reconfigure block data accordingly
    """
    blocks = [remove_list_items(block) for block in blocks]
    blocks = [remove_empty_lines(block) for block in blocks]
    blocks = [block for block in blocks if block["lines"]]

    return blocks


def modify_block_for_debug(block_list):
    debug_block_list = []
    for block in block_list:
        new_block = copy.deepcopy(block)
        new_block["text"] = []
        for line in new_block['lines']:
            new_block["text"].append(" ".join(span['text'] for span in line['spans']))
            line.pop('dir', None)
            line.pop('wmode', None)
            line.pop('bbox', None)
            line.pop('tokens', None)
            line.pop('word_bbox', None)
            line.pop('spans', None)
            line.pop('edited_text', None)
        new_block.pop('lines', None)
        debug_block_list.append(new_block)
    return debug_block_list


def modify_tree_block_for_debug(children):
    tree = []
    for head in children:
        formatted_head = {"level": head.level, 'element_id': head.element_id}
        if head.heading:
            formatted_head['heading'] = {"style": head.heading.style, "text": " ".join(head.heading.tokens), "table_info": head.heading.table_info},
        if head.children:
            formatted_head['children'] = modify_tree_block_for_debug(head.children)
        tree.append(formatted_head)

    return tree


def get_token_list(block):
    tokens = []
    word_bbox = []
    for line in block["lines"]:
        tokens.extend(line["tokens"])
        word_bbox.extend(line["word_bbox"])
    return {"tokens": tokens, "word_bbox": word_bbox}


def block_table_data(page_data, tables, page_no, debug):
    filtered_tables = filter_unique_tables(tables)
    # print(f"Page {page_no} has {len(filtered_tables)} tables")
    # for table in filtered_tables:
    #     print(table.bbox)
    page_tables_merged = []

    if len(filtered_tables):
        for tbl in filtered_tables:
            table_text = tbl.extract()
            table_merged = merge_table(tbl, table_text, page_data)
            page_tables_merged.append(table_merged)
        for table_index, table in enumerate(filtered_tables):
            table_bbox_large = fitz.Rect([table.bbox[0] - 10, table.bbox[1] - 10, table.bbox[2] + 10, table.bbox[3] + 10])
            for i, block in enumerate(page_data['blocks']):
                if fitz.Rect(block['bbox']) in table_bbox_large:
                    page_data['blocks'][i]['table_info'] = f"{page_no}_{table_index}"
                    continue
    return page_data, page_tables_merged


def get_page_data(vector):
    idx = vector[0]
    cpu_cnt = vector[1]
    filepath = vector[2]
    debug = vector[3]
    doc = fitz.open(filepath)
    num_pages = doc.page_count
    metadata=doc.metadata

    log_message=f"Number of pages in the file: {num_pages} and metadata is {metadata}\n"
    with open("metrics_th_old.txt", "a") as log_file:
            log_file.write(log_message)

    seg_size = int(num_pages / cpu_cnt) + (1 if num_pages % cpu_cnt else 0)
    seg_from = idx * seg_size
    seg_to = min(seg_from + seg_size, num_pages)

    results = []

    for page_no in tqdm(range(seg_from, seg_to), desc=f"Extracting {os.path.basename(filepath)} seg {idx}", disable=seg_to <= seg_from):
        # for page_no in range(seg_from, seg_to):
        page = doc[page_no]
        # ocr_page = page.get_textpage_ocr(flags=fitz.TEXT_INHIBIT_SPACES & ~fitz.TEXT_PRESERVE_IMAGES)
        # ocr_page = page.get_textpage_ocr(flags=~fitz.TEXT_PRESERVE_IMAGES)
        start_time = time()
        ocr_page = page.get_textpage_ocr()
        print(f"\nPyMuPDF (fitz) version: {fitz.__version__}\n")
        log_message=f"Time taken to execute get_textpage_ocr: {time()-start_time}\n"
        with open("metrics_th_old.txt", "a") as log_file:
            log_file.write(log_message)
        # ocr_page = page
        # page = page.get_textpage_ocr()
        start_time = time()
        page_data = ocr_page.extractDICT()
        log_message=f"Time taken to execute extractDICT: {time()-start_time}\n"
        with open("metrics_th_old.txt", "a") as log_file:
            log_file.write(log_message)
        # page_data = page.get_text("dict", flags=fitz.TEXT_INHIBIT_SPACES & ~fitz.TEXT_PRESERVE_IMAGES)
        # page_data = page.get_text("dict", flags=~fitz.TEXT_PRESERVE_IMAGES)
        # page_data = page.get_text("dict", flags=0)

        # ocr_page_text = ocr_page_data.extractDICT()
        # with open(f"/tenant/logs/ocr_page_{page_no}.txt", "w+") as f:
        #     json.dump(list(ocr_page_data.__dict__.keys()), f, indent=4)
        #     # json.dump(ocr_page_text.keys(), f, indent=4)
        #     json.dump(ocr_page_text, f, indent=4)
        #     # json.dump(ocr_page_data.__dict__, f, indent=4)

        # word_data_list = page.get_text("words", flags=~fitz.TEXT_PRESERVE_IMAGES)
        # word_data_list = page.get_text("words", flags=0)

        start_time = time()
        word_data_list = ocr_page.extractWORDS()
        log_message=f"Time taken to execute extractWORDS: {time()-start_time}\n"
        with open("metrics_th_old.txt", "a") as log_file:
            log_file.write(log_message)
        # word_data_list = page.get_text("words", flags=~fitz.TEXT_PRESERVE_IMAGES)
        # for word_data in word_data_list:
        #     block_no = word_data[5]
        #     line_no = word_data[6]

        #     if (block_no >= len(page_data["blocks"]) or line_no >= len(page_data["blocks"][block_no]["lines"])):
        #         word_data_list = page.get_text("words", flags=fitz.TEXT_INHIBIT_SPACES & ~fitz.TEXT_PRESERVE_IMAGES)
        #         break

        start_time = time()
        tables = page.find_tables()
        log_message=f"Time taken to execute find_tables: {time()-start_time}\n"
        with open("metrics_th_old.txt", "a") as log_file:
            log_file.write(log_message)
        # tables = []

        page_data["blocks"] = [block for block in page_data["blocks"] if block["type"] == 0]
        [block.update({'list_item_start': False, 'page_no': page_no, 'table_info': ''}) for block in page_data["blocks"]]

        # initialize empty list
        for block_no, block in enumerate(page_data["blocks"]):
            for line_no, line in enumerate(block["lines"]):
                # for s in line['spans']:
                #     # print(s)
                #     text = s["text"]
                #     if INVALID_UNICODE in text:  # invalid characters encountered!
                #         print("invalid at:", block_no, line_no, text) # invoke OCR
                page_data["blocks"][block_no]["lines"][line_no]["tokens"] = []
                page_data["blocks"][block_no]["lines"][line_no]["word_bbox"] = []

        # insert table data if block part of a table
        start_time = time()
        page_data, page_tables = block_table_data(page_data, tables, page_no, debug)
        log_message=f"Time taken to execute block_table_data fn in get_page_data: {time()-start_time}\n"
        with open("metrics_th_old.txt", "a") as log_file:
            log_file.write(log_message)

        for word_data in word_data_list:
            block_no = word_data[5]
            line_no = word_data[6]
            if (block_no >= len(page_data["blocks"]) or line_no >= len(page_data["blocks"][block_no]["lines"])):
                continue
            bbox = list(word_data[:4])
            bbox[0] = bbox[0] / page_data["width"]
            bbox[1] = bbox[1] / page_data["height"]
            bbox[2] = bbox[2] / page_data["width"]
            bbox[3] = bbox[3] / page_data["height"]
            page_data["blocks"][block_no]["lines"][line_no]["tokens"].append(word_data[4])
            page_data["blocks"][block_no]["lines"][line_no]["word_bbox"].append(tuple(bbox + [page_no]))

        start_time=time()
        page_data["blocks"] = clean_blocks(page_data["blocks"])
        divided_block_list = []
        for block in page_data["blocks"]:
            divided_block_list.extend(check_divide(block))
        page_data["blocks"] = clean_blocks(divided_block_list)
        log_message=f"Time taken to execute clean_blocks: {time()-start_time}\n"
        with open("metrics_th_old.txt", "a") as log_file:
            log_file.write(log_message)
        

        # group blocks with same font and sort the groups by up to down and then left to right
        if page_data["blocks"]:
            all_groups = []

            default_group = {"blocks": [], "bbox": [1e8, 1e8, 0, 0], "table_info": ""}
            group = copy.deepcopy(default_group)
            common_hashes = page_data["blocks"][0]["common_hashes"]

            for block in page_data["blocks"]:
                common_hashes = [span_hash for span_hash in common_hashes if span_hash in block["common_hashes"]]
                if block["table_info"] != group["table_info"] or (not block["table_info"] and (not common_hashes or block["bbox"][1] > group["bbox"][1] + 30)):
                    all_groups.append(group)
                    group = copy.deepcopy(default_group)
                    common_hashes = block["common_hashes"]
                group["blocks"].append(block)
                group["bbox"][0] = min(group["bbox"][0], block["bbox"][0])
                group["bbox"][1] = min(group["bbox"][1], block["bbox"][1])
                group["bbox"][2] = max(group["bbox"][2], block["bbox"][2])
                group["bbox"][3] = max(group["bbox"][3], block["bbox"][3])
                group["table_info"] = block["table_info"]

            if group["blocks"]:
                all_groups.append(group)

            # sort groups by up to down and then left to right
            all_groups = sorted(all_groups, key=lambda x: (x["bbox"][1], x["bbox"][0]))

            # get the sorted blocks
            page_data["blocks"] = [block for group in all_groups for block in group["blocks"]]

        results.append((page_data["blocks"], page_tables, page_data['width'], page_data['height']))
    doc.close()
    return results


def pdf_parser(filepath, table_request=None, debug=False, debug_path="/tenant/logs/test_output"):
    metrics_file="metrics_th_old.txt"
    start_time_main = time()
    log_message = f"Analysing the file {filepath}\n"
    with open(metrics_file, "a") as log_file:
            log_file.write(log_message)

    if table_request is None:
        table_request = TableToTextRequest().dict()

    # print("Parsing file:", filepath)
    # print("debug_path:", debug_path)
    file_name = os.path.basename(filepath)
    if debug:
        debug_path = os.path.join(debug_path, file_name)
        os.makedirs(debug_path, exist_ok=True)

    if debug:
        print(os.path.abspath(filepath))

    cpu_cnt = min(cpu_count(), 8)
    cpu_cnt=1
    log_message = f"Number of CPUs: {cpu_cnt}\n"
    with open(metrics_file, "a") as log_file:
            log_file.write(log_message)

    parse_vectors = [(i, cpu_cnt, filepath, debug) for i in range(cpu_cnt)]

    start_time = time()

    log_message=f"Executing the function get_page_data\n"
    with open(metrics_file, "a") as log_file:
            log_file.write(log_message)
    pool = Pool()
    results = pool.map(get_page_data, parse_vectors, 1)
    pool.close()
    
    log_message=f"Time taken to execute get_page_data: {time()-start_time}\n"
    with open(metrics_file, "a") as log_file:
            log_file.write(log_message)

    page_wise_block_list = []
    page_wise_tables_list = []
    page_wise_width = []
    page_wise_height = []
    fonts = []
    line_gaps = {}
    num_pages = 0
    bold_spans_per_size = {}
    for result in results:
        num_pages += len(result)
        # extract line gap
        for page_blocks, page_tables, page_width, page_height in result:
            for block in page_blocks:
                if block['lines'] and block['lines'][0]['spans']:
                    fonts.append(block['lines'][0]['spans'][0]['size'])
                if len(block["lines"]) > 1 and not block['table_info'] and not (detect_list_item(block["lines"][0]) or detect_list_item(block["lines"][1])):
                    for span1 in block["lines"][0]["spans"]:
                        # fonts.append(span1["size"])
                        if span1['size'] not in bold_spans_per_size:
                            bold_spans_per_size[span1['size']] = {"bold": 0, "non_bold": 0}
                        if span1['flags'] & 2**4:
                            bold_spans_per_size[span1['size']]["bold"] += 1
                        else:
                            bold_spans_per_size[span1['size']]["non_bold"] += 1
                        span1_hash = font_hash(span1)
                        for span2 in block["lines"][1]["spans"]:
                            span2_hash = font_hash(span2)
                            if span1_hash == span2_hash and span2["bbox"][1] > span1["bbox"][3]:
                                gap = span2["bbox"][1] - span1["bbox"][3]
                                gap = round(gap, 1)
                                if gap < float(span1_hash.split('|')[1]) and (span1_hash not in line_gaps or gap not in line_gaps[span1_hash]):
                                    line_gaps[span1_hash] = line_gaps[span1_hash] if span1_hash in line_gaps else []
                                    # print("Line gap:", gap, span1_hash, span1, span2)
                                    line_gaps[span1_hash].append(gap)

            page_wise_block_list.append(page_blocks)
            page_wise_tables_list.append(page_tables)
            page_wise_width.append(page_width)
            page_wise_height.append(page_height)

    # join tables accross pages
    start_time = time()
    page_wise_tables_list, table_mapping = join_tables_accross_pages(page_wise_tables_list)
    log_message=f"Time taken to execute join_tables_accross_pages: {time()-start_time}\n"
    with open(metrics_file, "a") as log_file:
            log_file.write(log_message)

    table_jsons = {}
    for page_no, tables in enumerate(page_wise_tables_list):
        for table_no, table in enumerate(tables):
            table_key = f"{page_no}_{table_no}"
            if table_mapping[table_key] == table_key:
                table_merged = process_table(table)
                row_bboxs, col_line_no, column_bboxs, row_line_no, all_cells = table_rows_and_columns(table_merged)
                table_merged, row_bboxs, row_line_no, row_ignored_prefix = remove_empty_table_lines(table_merged, row_bboxs, row_line_no, all_cells)
                table_merged, column_bboxs, col_line_no, col_ignored_prefix = remove_empty_table_lines(table_merged, column_bboxs, col_line_no, all_cells, start_idx=1, end_idx=3)
                # table_merged = detect_headers(table_merged, row_bboxs, row_line_no, all_cells)
                col_headers = detect_headers(table_merged, column_bboxs, col_line_no, col_ignored_prefix, all_cells, row_line_no, x_start=0, x_end=2, start_idx=1, end_idx=3)
                row_headers = detect_headers(table_merged, row_bboxs, row_line_no, row_ignored_prefix, all_cells, col_line_no, first_row_header=False, x_start=1, x_end=3, start_idx=0, end_idx=2)
                table_json = table_to_json(table_merged, row_bboxs, col_line_no, column_bboxs, row_line_no, row_ignored_prefix, col_ignored_prefix, filepath, col_headers, row_headers, page_no,
                                           page_wise_width[page_no], page_wise_height[page_no])
                # print(table_key, table_json['row_count'])
                if table_json['row_count'] < 2:
                    table_mapping[table_key] = ""
                    continue
                table_jsons[table_key] = table_json

    block_list = []
    if len(page_wise_block_list) > 1:
        page_wise_block_list = remove_header_footer(page_wise_block_list, debug)
    for page_wise_blocks in page_wise_block_list:
        block_list.extend(page_wise_blocks)

    if debug:
        with open(f'{debug_path}/tables.json', 'w') as f:
            json.dump(page_wise_tables_list, f, indent=4)
        with open(f'{debug_path}/block_list_before_merge.json', 'w') as f:
            json.dump(modify_block_for_debug(block_list), f, indent=4)

    block_list = [{**block, 'idx': idx} for idx, block in enumerate(block_list)]

    if len(block_list):

        redemptions = 2
        redemption_thresh = 3

        for trial in range(redemptions):
            global block_miss
            global block_double

            block_miss = defaultdict(list)
            block_double = defaultdict(list)

            # if debug:
            #     print(line_gaps)

            merged_block_list = [block_list[0]]

            for block in block_list[1:]:
                can_merge = check_merge(merged_block_list[-1], block, line_gaps)
                # if debug:
                #     print(" ".join(get_token_list(merged_block_list[-1])["tokens"]), " ".join(get_token_list(block)["tokens"]), can_merge)
                if can_merge:
                    merged_block_list[-1] = merge(merged_block_list[-1], block)
                else:
                    merged_block_list.append(block)
                # if debug:
                #     print('-' * 100)

            block_list = merged_block_list

            # # if debug:
            # print("+" * 20)
            # print("Block double:", json.dumps(block_double, indent=4))
            # print("block_miss:", json.dumps(block_miss, indent=4))
            # print("line_gaps:", json.dumps(line_gaps, indent=4))

            for span_hash_gap in block_double.keys():
                if len(block_double[span_hash_gap]) > redemption_thresh:
                    span_hash, gap = span_hash_gap.split('~')
                    if span_hash in line_gaps and line_gaps[span_hash]:
                        avg_gap = sum(line_gaps[span_hash]) / len(line_gaps[span_hash])
                        if abs(float(gap) - avg_gap) > 1.5:
                            continue
                    line_gaps[span_hash] = line_gaps[span_hash] if span_hash in line_gaps else []
                    line_gaps[span_hash].append(float(gap))

    block_list = clean_blocks(block_list)
    # dump the output of block_list to json
    if debug:
        with open(f'{debug_path}/block_list.json', 'w') as f:
            json.dump(modify_block_for_debug(block_list), f, indent=4)

    # Calculate most frequent font from fonts
    font_counter = Counter(fonts)
    if not font_counter:
        print('Warning, no fonts detected! Skipping')
        return []

    common_font = font_counter.most_common(1)[0][0]
    # print(bold_spans_per_size, font_counter)
    try:
        regular_text_is_bold = bold_spans_per_size[common_font]["bold"] / (bold_spans_per_size[common_font]["bold"] + bold_spans_per_size[common_font]["non_bold"]) > 0.5
    except Exception as e:
        print(f"Error {e},\nbold spans per size: {bold_spans_per_size}, common_font: {common_font}")
        regular_text_is_bold = False
    block_list = create_hierarchy(block_list, {'_body_size': common_font}, regular_text_is_bold)

    # Save to JSON file with custom encoder
    if debug:
        with open(f'{debug_path}/output.json', 'w') as f:
            json.dump(modify_tree_block_for_debug(block_list), f, cls=CustomEncoder, indent=4)

    # insert section data into table objects
    visited_table = defaultdict(bool)

    def parse_tree(head, parent, section_text):
        if head.heading and head.heading.table_info and table_mapping[head.heading.table_info] == head.heading.table_info:
            if not visited_table[head.heading.table_info]:
                visited_table[head.heading.table_info] = True
                table_key = head.heading.table_info
                if parent:
                    if parent.heading:
                        table_jsons[table_key]["table_section_title"] = " ".join(parent.heading.tokens)
                    table_jsons[table_key]["table_section_text"] = section_text
        if head.children:
            section_text = ""
            for child in head.children:
                if section_text:
                    section_text += "\n"
                section_text += parse_tree(child, head, section_text)
        return " ".join(head.heading.tokens) if head.heading and not head.heading.table_info else ""
    
    start_time = time()
    for head in block_list:
        parse_tree(head, None, "")

    log_message=f"Time taken to execute parse_tree: {time()-start_time}\n"
    with open(metrics_file, "a") as log_file:
            log_file.write(log_message)
    if debug:
        with open(f'{debug_path}/table_jsons.json', 'w') as f:
            json.dump(table_jsons, f, indent=4)

    # generate table2text over table_jsons using multiprocessing
    filtered_tables = table_jsons.keys()

    def formatTable2text(table):
        table_input = copy.deepcopy(table_request)
        table_input["table"] = table
        table_input["backend"] = "Groq/LLAMA70B"
        return table_input
        # return {"backend": "vLLM/Mistralv2AWQ", "table": table, "method": "naive", "max_new_tokens": 300, "temperature": 0.01, "top_k": 50, "top_p": 0.95, "prune_dangling_sents": True}

    # celery_tasks = celery_group([
    #     celery_app.signature(
    #         "pipelines.celery_tasks.tasks.table2text_task",
    #         args=(formatTable2text(table_jsons[table_key]),),
    #         priority=1

    #     ) for table_key in filtered_tables
    # ]).delay()

    # with allow_join_result():
    #     # should return list of table2text outputs
    #     table_outputs = celery_tasks.get()

    start_time = time()
    table_outputs = [table2text_task(formatTable2text(table_jsons[table_key])) for table_key in filtered_tables]
    log_message=f"Time taken to execute table2text_task: {time()-start_time}\n"
    with open(metrics_file, "a") as log_file:
            log_file.write(log_message)
    # print(table_outputs)
    if debug:
        for num, table in enumerate(table_jsons):
            with open(f"{debug_path}/table_{num}.json", "w") as f:
                json.dump(table, f)

    for table_key, table_outputs in zip(filtered_tables, table_outputs):
        table_jsons[table_key]["table2text_output"] = table_outputs['output']

    visited_table = defaultdict(bool)

    # replace nodes with teble2text outputs
    def replaceTableObjects(head):
        if head.heading and head.heading.table_info:
            table_key = head.heading.table_info
            assert not head.children
            if table_mapping[table_key] == table_key and not visited_table[table_key]:
                visited_table[head.heading.table_info] = True
                row_objects = []
                for row in table_jsons[table_key]["table2text_output"]:
                    row_object = copy.deepcopy(head)
                    row_object.heading.tokens = (row["linearized_table"] if table_request['only_linearize'] else row["row_text_result"]).strip().split()
                    row_object.heading.word_bbox = list(islice(cycle(row["cell_bboxs"]), len(row_object.heading.tokens)))
                    row_object.heading.original_text = row["row_text"]
                    row_object.heading.generation_text = row["linearized_table_for_generate"]
                    row_objects.append(row_object)
                    # print(row_object.__dict__)
                return row_objects
            else:
                return []
        if head.children:
            new_children = []
            for child in head.children:
                new_children.extend(replaceTableObjects(child))
            head.children = new_children
        return [head]

    final_block_list = []
    for head in block_list:
        final_block_list.extend(replaceTableObjects(head))

    log_message=f"Time taken to execute overall file parsing: {time()-start_time_main}\n"
    with open(metrics_file, "a") as log_file:
            log_file.write(log_message)
    return final_block_list

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--doc_dir", help="Directory containing files", required=True)
#     args = parser.parse_args()

#     docs = [args.doc_dir] if args.doc_dir.endswith(".pdf") else glob.glob(os.path.join(args.doc_dir, f"**/*.pdf"), recursive=True)

#     for doc in docs:
#         pdf_parser(doc, debug=True, debug_path="/tenant/logs/test_output")

if __name__ == "__main__":
    pdf_file = "theory.pdf" 

    if os.path.exists(pdf_file) and pdf_file.endswith(".pdf"):
        pdf_parser(pdf_file, debug=True, debug_path="/tenant/logs/test_output")
    else:
        print(f"File {pdf_file} not found or is not a PDF.")
