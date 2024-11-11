import os
import re
import time
import pickle as pkl
import json
import random
import time
import scipy
import traceback
# from sentence_transformers import SentenceTransformer
from sqlalchemy.sql.operators import exists
from raven import Client
from pickle import dump as pickle_dump
from itertools import chain
from tqdm import tqdm
from uuid import UUID, uuid4
from celery.utils.log import get_task_logger
from celery import Task
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import TensorDataset
import csv
import re
from app import schemas

from app.core.celery_app import celery_app
from app.core.config import settings
from app.db.session import SessionLocal
from app import crud
from app.parser_utils.pdf_module import pdf_parser
from app.parser_utils.ocr_module import ocr_parser
from app.parser_utils.csv_parser import csv_parser
from app.common_tasks import delete_data, delete_group_data
from celery.result import allow_join_result
from transformers import (AutoTokenizer)
# from app.parser_utils.sbert import embed_batch, embed_query, cos
from app.crud import s3 as s3_client

from app.crud.web_api_handler import WebAPI

handler = WebAPI(settings.WEB_API_URL, settings.WEB_API_USER, settings.WEB_API_PASS)

client_sentry = Client(settings.SENTRY_DSN)

logger = get_task_logger(__name__)
logger.setLevel(settings.logging_level)

# import transformers
# transformers.logging.set_verbosity_error()

# import nltk libraries here

import re
import json
import nltk
from nltk import pos_tag, word_tokenize

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


class ExtractDataTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model and frequency dict.
    """

    abstract = True

    def __init__(self):
        super().__init__()
        self.nlp = None

    def __call__(self, *args, **kwargs):

        logger.info(f"args: {args}")
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if self.nlp is None:
            logger.info("Loading Models...")
            import spacy

            self.nlp = spacy.load(settings.spacy_model, exclude=["ner", "lemmatizer"])
            logger.info("Spacy Model loaded")

            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.pretrained_name_or_path,
                do_lower_case=settings.do_lower_case,
                cache_dir=settings.cache_dir,
            )

        return self.run(*args, **kwargs)


@celery_app.task(bind=True, base=ExtractDataTask)
def proper_truecase(self, question):
    doc = self.nlp(question)
    return " ".join([(token.text.upper() if token.pos_ == "PROPN" else token.text) for token in doc])


@celery_app.task(bind=True, base=ExtractDataTask)
def extract_data(
    self,
    doc_id,
    raise_error_on_fail=True,
    skip_if_latest_parsed=True,
    join_segment_paras=True,
    keywords_top_k=15,
    segment_keywords=False,
    append_summary=False,
    append_doc_title=True,
    table_request=None,
):
    db = SessionLocal()
    document = crud.document.get(db=db, id=doc_id)
    if skip_if_latest_parsed and document.status == "Done" and document.doc_processed_version == settings.DOC_PIPELINE_VERSION:
        logger.info(f"Doc {doc_id} already processed, skipping...")
        db.close()
        return doc_id
    save_name = document.save_name
    save_name = save_name.replace("/", "_")
    document.status = "Parsing"
    document.doc_processed_version = settings.DOC_PIPELINE_VERSION

    # if settings.UPDATE_DOC_STATUS:
    #     res = handler.update_doc_status(doc_id, document.status)
    db.commit()
    db.close()

    parse_result = parse_file(doc_id,
                              self.nlp,
                              join_segment_paras=join_segment_paras,
                              keywords_top_k=keywords_top_k,
                              segment_keywords=segment_keywords,
                              append_summary=append_summary,
                              append_doc_title=append_doc_title,
                              table_request=table_request)

    if parse_result == -1:
        if raise_error_on_fail:
            raise Exception("Failed Parsing")
        return doc_id
    else:
        db = SessionLocal()
        document = crud.document.get(db=db, id=doc_id)
        document.status = "Parsing"
        # if settings.UPDATE_DOC_STATUS:
        #     res = handler.update_doc_status(doc_id, document.status)
        db.commit()
        db.close()

    store_data, qg_data = parse_result

    data = {"data": []}
    index = 0
    limit = settings.extract_batch_size
    while index < len(store_data):
        data["data"].append({
            "title": save_name + ";;" + str(index),
            "doc_idx": doc_id,
            "paragraphs": [con for con in store_data[index:min(len(store_data), index + limit)]],
        })
        index += limit

    word_count = get_document_metrics(doc_id, data, self.tokenizer)
    if settings.UPDATE_DOC_STATUS:
        # webapi check if doc_id with document.word_count should be processed
        print("webapi check if doc_id with document.word_count should be processed")
        should_process = True
        print(doc_id, word_count)
        should_process = handler.check_doc_status(doc_id, word_count)
        if not should_process:
            db = SessionLocal()
            document = crud.document.get(db=db, id=doc_id)
            document.status = "Failed:Token Limit Exceeded"
            db.commit()
            db.close()
            if raise_error_on_fail:
                raise Exception("Failed:Token Limit Exceeded")
            return doc_id

    qg_contexts_dir = os.path.join(settings.qg_contexts_dir, save_name)
    with open(qg_contexts_dir, "w+") as f:
        json.dump(qg_data, f, indent=1)

    phrase_save_dir = os.path.join(settings.phrase_data_dir, save_name)
    with open(phrase_save_dir, "w+") as f:
        json.dump(data, f, indent=1)
    logger.info(phrase_save_dir)

    if settings.MULTI_HOSTS:
        s3 = s3_client.get_s3_instance()

        logger.info(f"Uploading {phrase_save_dir}")
        s3_client.upload_file(phrase_save_dir, phrase_save_dir[1:], s3=s3)

        logger.info(f"Uploading {qg_contexts_dir}")
        s3_client.upload_file(qg_contexts_dir, qg_contexts_dir[1:], s3=s3)

    return doc_id


def get_document_metrics(doc_id, data, tokenizer):
    db = SessionLocal()
    document = crud.document.get(db=db, id=doc_id)
    document.para_count = sum([len(segment['paragraphs']) for segment in data['data']])
    document.char_count = sum([sum([len(con['context']) for con in segment['paragraphs']]) for segment in data['data']])
    document.word_count = sum([sum([len(tokenizer.encode(con['context'])) for con in segment['paragraphs']]) for segment in data['data']])
    word_count = document.word_count
    # for segment in data['data']:
    #     for con in segment['paragraphs']:
    #         print(' '.join(tokenizer.tokenize(con['context'])))

    # document.word_count = sum([len(con['context'].split(' ')) for con in store_data])

    logger.info(f"Doc: {document.save_name}")
    logger.info(f"No of paragraph: {document.para_count}")
    logger.info(f"Word count: {document.word_count}")
    logger.info(f"Char count: {document.char_count}")
    db.commit()
    db.close()
    return word_count


@celery_app.task(bind=True, base=ExtractDataTask)
def update_metrics(self, doc_id):
    db = SessionLocal()
    document = crud.document.get(db=db, id=doc_id)

    phrase_save_dir = os.path.join(settings.phrase_data_dir, document.save_name)

    if not os.path.exists(phrase_save_dir):
        return doc_id

    if settings.MULTI_HOSTS:
        s3_client.download_file(phrase_save_dir, phrase_save_dir[1:])
        logger.info("Downloaded phrase data")

    with open(phrase_save_dir, "r") as f:
        data = json.load(f)

    get_document_metrics(doc_id, data, self.tokenizer)
    return doc_id


def parse_file(doc_id, nlp, join_segment_paras=True, update_status=True, keywords_top_k=15, segment_keywords=False, append_summary=False, append_doc_title=True, table_request=None):
    db = SessionLocal()
    document = crud.document.get(db=db, id=doc_id)
    file_path = document.file_path
    doc_name = document.file_name
    save_name = document.save_name
    save_name = save_name.replace("/", "_")
    doc_title = '_'.join(save_name.split('_')[2:])
    s3_path = document.s3_path
    db.close()

    def mark_failed(failed_msg):
        if update_status:
            db = SessionLocal()
            document = crud.document.get(db=db, id=doc_id)
            document.status = failed_msg
            # if settings.UPDATE_DOC_STATUS:
            #     res = handler.update_doc_status(doc_id, document.status)
            db.commit()
            db.close()

    metadata = None

    if settings.MULTI_HOSTS and s3_path:
        s3_client.download_file(file_path, s3_path)

    logger.info(file_path)
    logger.info(f"Extracting data from {doc_name}")

    segments_text = {}

    def block_to_segment_list(block_list):
        segment_list = []

        def format_title(title, is_last=False):
            title = str(title).strip()
            if len(title) > 80 and not is_last:
                return title[:40] + " ... " + title[-40:]
            return title

        def parse_tree(head, parent_parse_title=[], title_id='0', previous_font_size=-1):
            parse_title = list(parent_parse_title)
            if head.heading and head.heading.tokens:
                segment_list.append({
                    "tokens": head.heading.tokens,
                    "word_bbox": head.heading.word_bbox,
                    "parse_title": "| ".join([format_title(title_text, is_last=(i == len(parse_title) - 1)) for i, title_text in enumerate(parse_title)]),
                    "title_id": title_id,
                    "original_text": [head.heading.original_text],
                    "generation_text": [head.heading.generation_text],
                })
                if title_id not in segments_text:
                    segments_text[title_id] = ""
                segments_text[title_id] += " " + " ".join(head.heading.tokens)
            if head.children:
                if head.heading:
                    title_text = " ".join(head.heading.tokens)
                    if head.heading.style.size == previous_font_size:
                        parse_title[-1] += " " + title_text
                    else:
                        parse_title.append(title_text)
                for child in head.children:
                    parse_tree(child, parse_title, title_id + "_" + str(head.element_id), head.heading.style.size if head.heading else previous_font_size)

        for child_id, block in enumerate(block_list):
            parse_tree(block, title_id=str(child_id))

        return segment_list

    segment_keywords_dict = {}

    try:
        if file_path.lower().endswith('.pdf'):
            doc_type = "pdf"
            # is_text_pdf = check_pdf(file_path)
            is_text_pdf = True
            if is_text_pdf:  # PDF is text extractable
                block_list = pdf_parser(file_path, table_request)
                segment_list = block_to_segment_list(block_list)
                if segment_keywords:
                    for title_id, segment_text in segments_text.items():
                        encoding_task = celery_app.signature(
                            "app.transformer_common_tasks.keyword_extraction",
                            args=([segment_text], keywords_top_k),
                        ).set(queue='transformer-query-queue').delay()

                        with allow_join_result():
                            segment_keywords_dict[title_id] = encoding_task.get()[0]
            # if not is_text_pdf or len(segment_list) == 0:  # Needs OCR
            #     segment_list = ocr_parser(file_path)
        elif file_path.endswith('.csv'):
            doc_type = "csv"
            segment_list, metadata = csv_parser(file_path)
        else:
            mark_failed("Failed:Unrecognized File Type")
            return -1
    except Exception as e:
        # log here
        print(traceback.format_exc())
        logger.info(e)
        mark_failed("Failed:Corrupted Document")
        return -1

    # Log the pdf parser outputs for easy debugging

    logger.info(f"Extracted items count: {len(segment_list)}")

    # Zero segments case
    if len(segment_list) == 0:
        logger.info(f"Zero text segments found for {doc_name}")
        # Save doc status as extraction failed
        mark_failed("Failed:No data Extracted")
        return -1

    def preprocess_content(content):
        content = content.replace("\n", " ")
        content = content.replace("\t", " ")
        content = content.replace("\x00", "")
        content = re.sub(r" +", " ", content)
        content = content.strip()
        content = content.encode('utf-8', errors='ignore').decode()
        return content

    store_data = []
    qg_data = []

    # merge segments from the same section

    if doc_type == "pdf":
        for sengment_id, segment in enumerate(segment_list):
            segment_list[sengment_id]["tokens"] = [preprocess_content(token) for token in segment["tokens"]]
            segment_list[sengment_id]["word_bbox"] = [word_bbox for token_id, word_bbox in enumerate(segment_list[sengment_id]["word_bbox"]) if len(segment_list[sengment_id]["tokens"][token_id])]
            segment_list[sengment_id]["tokens"] = [token for token in segment_list[sengment_id]["tokens"] if len(token)]
            segment_list[sengment_id]["content"] = " ".join(segment_list[sengment_id]["tokens"])
        segment_list = [segment for segment in segment_list if len(segment["content"])]
    else:
        segment_list = [preprocess_content(segment) for segment in segment_list]
        segment_list = [segment for segment in segment_list if len(segment)]

    new_segment_list = []
    paragraph_bounds = []
    metadata_items = []
    if doc_type == "pdf":
        # merge segments that belong to same page
        # join based on section data and keep it optional to test
        for segment in segment_list:
            # if len(new_segment_list) == 0 or segment["word_bbox"][0][4] != new_segment_list[-1]["word_bbox"][0][4]:
            if not join_segment_paras or len(new_segment_list) == 0 or segment["title_id"] != new_segment_list[-1]["title_id"]:
                new_segment_list.append(segment)
                paragraph_bounds.append([[0, len(segment["content"])]])
            else:
                add_punct = ""
                if new_segment_list[-1]["content"][-1].isalnum():
                    add_punct = "."
                paragraph_bounds[-1].append([len(new_segment_list[-1]["content"]) + 1 + len(add_punct), len(new_segment_list[-1]["content"]) + 1 + len(add_punct) + len(segment["content"])])
                new_segment_list[-1]["content"] += add_punct + "\n" + segment["content"]
                new_segment_list[-1]["tokens"] += segment["tokens"]
                new_segment_list[-1]["word_bbox"] += segment["word_bbox"]
                new_segment_list[-1]["original_text"] += segment["original_text"]
                new_segment_list[-1]["generation_text"] += segment["generation_text"]
    else:
        # merge every k segments
        section_size = 1
        for segment_id, segment in enumerate(segment_list):
            if segment_id % section_size == 0:
                new_segment_list.append(segment)
                paragraph_bounds.append([[0, len(segment)]])
                if metadata is not None and len(metadata):
                    metadata_items.append([metadata[segment_id]])
            else:
                paragraph_bounds[-1].append([len(new_segment_list[-1]) + 1, len(new_segment_list[-1]) + 1 + len(segment)])
                new_segment_list[-1] += "\n" + segment
                if metadata is not None and len(metadata):
                    metadata_items[-1].append(metadata[segment_id])

    segment_list = new_segment_list

    def extract_metadata(content):
        if append_summary:
            # summarize the content
            encoding_task = celery_app.signature(
                "app.transformer_processing.summarize",
                args=(content,),
            ).delay()
            # top 5 nearest paragraphs
            with allow_join_result():
                summarization = encoding_task.get()
            return summarization
        else:
            return content

    if doc_type == "pdf":
        contents = [segment["content"] for segment in segment_list]
        extra_contexts = [(f"{doc_title}: " if append_doc_title else '') + segment["parse_title"] + (f" {segment_keywords_dict[segment['title_id']]}" if segment_keywords else '') +
                          (f" {extract_metadata(segments_text[segment['title_id']])}" if append_summary else '') + ": " for segment in segment_list]
    else:
        contents = segment_list
        extra_contexts = [extract_metadata(content) for content in contents]

    parsed_list = list(nlp.pipe(contents))

    for segment_id, segment in enumerate(segment_list):
        content = contents[segment_id]
        parsed_content = parsed_list[segment_id]
        para_bounds = paragraph_bounds[segment_id]
        extra_context = extra_contexts[segment_id]
        metadata_item = []
        if len(metadata_items):
            metadata_item = metadata_items[segment_id]

        sent_batch = list(parsed_content.sents)

        if len(sent_batch) == 0:
            logger.info(f"\n -- WARNING: no sents found for text below!\n{content}\n -- ")

        sentences = []
        start_index = 0

        for sent_id, sent in enumerate(sent_batch):

            # sent_features = [stemmer.stem(t.text) for t in sent if t.pos_ in ["NOUN", "PROPN"]]
            # sent_features = list(set(sent_features))

            end_index = start_index + len(sent.text)
            sentences.append({
                "text": sent.text,
                # "key_features": sent_features,
                "boundaries": [start_index, end_index],
                "bbox": [1e8, 1e8, 0, 0, -1],
                "para_id": len(store_data),
                "sent_id": sent_id,
                "id": str(uuid4()),
                "doc_idx": doc_id,
                "phrases": [],
                "title": save_name + ";;",
            })

            start_index = end_index
            if start_index < len(content) and content[start_index].isspace():
                start_index += 1

        if doc_type == "pdf":
            sent_id = 0
            word_start = 0
            for token, bbox in zip(segment["tokens"], segment["word_bbox"]):
                if sentences[sent_id]["boundaries"][1] <= word_start:
                    sent_id += 1
                    if sent_id >= len(sentences):
                        mark_failed("Failed: Parsing error sent mismatch")
                        logger.info(f"Error in parsing {doc_name}\n{token}\n{sentences[-1]}")
                        return -1
                # join only blocks from single page.
                # TODO multiple bboxes
                if sentences[sent_id]["bbox"][4] == -1 or sentences[sent_id]["bbox"][4] == bbox[4]:
                    sentences[sent_id]["bbox"][0] = min(bbox[0], sentences[sent_id]["bbox"][0])
                    sentences[sent_id]["bbox"][1] = min(bbox[1], sentences[sent_id]["bbox"][1])
                    sentences[sent_id]["bbox"][2] = max(bbox[2], sentences[sent_id]["bbox"][2])
                    sentences[sent_id]["bbox"][3] = max(bbox[3], sentences[sent_id]["bbox"][3])
                    sentences[sent_id]["bbox"][4] = bbox[4]

                word_start += len(token) + 1

        store_data.append({
            "context": content,
            "metadata": metadata_item,
            "sentences": sentences,
            "original_para_id": segment_id,
            "document": doc_name,
            "para_bounds": para_bounds,
            "parse_titles": [extra_context if extra_context.strip() else doc_title],
        })
        if 'original_text' in segment:
            store_data[-1]["original_text"] = segment["original_text"]
            store_data[-1]["generation_text"] = segment["generation_text"]

        for send_id in range(0, len(sentences), settings.qg_sent_batch):
            r_idx = min(send_id + settings.qg_sent_batch, len(sentences))
            qg_data.append({
                "sentences": [sent["text"] for sent in sentences[send_id:r_idx]],
            })

    # Zero segments case
    if len(store_data) == 0:
        logger.info(f"Zero text segments extracted for {doc_name}")
        # Save doc status as extraction failed
        mark_failed("Failed:No data Extracted post processing")
        return -1

    return store_data, qg_data


"""
Question generation class + pipeline + modules
"""

# class QuestionGenerationTask(Task):
#     """
#     Abstraction of Celery's Task class to support loading ML model.
#     """

#     abstract = True

#     def __init__(self):
#         super().__init__()
#         self.nlp = None
#         self.spacy = None

#     def __call__(self, *args, **kwargs):

#         logger.info(f"args: {args}")
#         """
#         Load model on first call (i.e. first task processed)
#         Avoids the need to load model on each task request
#         """
#         if self.nlp is None:
#             try:
#                 from app.parser_utils.qg_pipelines import pipeline
#                 self.nlp = pipeline("question-generation", model="valhalla/t5-base-qg-hl")
#             except Exception as e:
#                 logger.info('QG Model Load failed')
#                 return

#         if self.spacy is None:
#             logger.info("Loading Models...")
#             import spacy

#             self.spacy = spacy.load(settings.spacy_model, exclude=["ner", "lemmatizer"])
#             logger.info("Spacy Model loaded")
#         return self.run(*args, **kwargs)

## Helper Functions for Context Validation

def is_numeric_heavy(text):
    tokens = text.split()
    numeric_tokens = [token for token in tokens if token.isdigit()]
    return len(numeric_tokens) / len(tokens) > 0.8

def is_too_short(text):
    return len(text.split()) < 10

def lacks_verbs(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    verbs = [word for word, pos in tagged if pos.startswith('VB')]
    return len(verbs) == 0

def is_repetitive(text):
    tokens = text.split()
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens) < 0.5

def is_valid_context(context):
    return not (is_numeric_heavy(context) or is_too_short(context) or lacks_verbs(context) or is_repetitive(context))

## Helper Functions for Question Validation
def is_question_generic(question):
    generic_phrases = ["What is mentioned", "from this context", "according to the context", "in the given", "in the context", "in the provided"]
    for phrase in generic_phrases:
        if phrase in question.lower():
            return True
    return False

def question_mismatch_with_context(question, context):
    question_keywords = set(re.findall(r'\b\w+\b', question.lower()))
    context_keywords = set(re.findall(r'\b\w+\b', context.lower()))
    common_words = question_keywords & context_keywords
    return len(common_words) < 2

def is_question_incomplete(question):
    return question.endswith('...')

def is_question_only_numbers(question):
    return re.fullmatch(r'[\d\s]+', question)

def is_valid_question(question, context):
    return not (is_question_generic(question) or question_mismatch_with_context(question, context) or is_question_incomplete(question) or is_question_only_numbers(question))

# Validate the contexts to ensure they are coherent, relevant, and suitable for question generation.
def validate_contexts(data, qg_request):
    # Improved validation prompt
    system_prompt = """
    You are a context validation assistant. Your task is to analyze the given context and determine if it is **valid** or **invalid** based on its coherence, relevance, and potential to support meaningful questions.
    
    - Respond with **"valid"** if the context is factual, clear, relevant, and contains structured information suitable for question generation.
    - Respond with **"invalid"** if the context is nonsensical, irrelevant, or lacks coherence (e.g., random numbers, unrelated sequences, or incomplete sentences).
    
    Examples:
    - Context: "The solar system consists of the Sun and objects that orbit it, including planets, moons, and asteroids." → **valid**
    - Context: "234jkdf 9832!! ?@ 123 solar system orbit 3344 77888." → **invalid**

    Output format: valid/invalid
    """
    
    # Updated user prompt template for clarity
    user_prompt_template = '''Based on the provided context, determine whether it is **valid** or **invalid**:
    - Context: {context}
    - Output: "valid" with a brief reason for valid contexts or "invalid" with a reason for invalid contexts.
    '''
    
    valid_contexts = []
    context_tasks=[]
    context_dict={}
    for section in data:
        for paragraph in section['paragraphs']:
            # Process paragraphs with sufficient length for validation
            if len(paragraph['context'].split()) > 30:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_template.format(context=paragraph['context'])}
                ]
                # Set up the request for the LIMA model
                request = schemas.VLLMRequest(
                    messages=messages,
                    max_tokens=qg_request['max_new_tokens'],
                    temperature=qg_request['temperature'],
                    top_k=qg_request['top_k'],
                    top_p=qg_request['top_p']
                )
                
                # Trigger asynchronous task and fetch result
                task = celery_app.signature(
                    "pipelines.celery_tasks.tasks.qg_task",
                    args=(qg_request, request.dict())
                ).delay()

                context_tasks.append((paragraph['original_para_id'],task))
                context_dict[paragraph['original_para_id']]=paragraph['context']

    for context_id,task in context_tasks:
        with allow_join_result():
            result = task.get()
        
        output_content=result['output']['choices'][0]['message']['content'].strip().lower()
                
        if 'valid' in output_content:
                valid_contexts.append(context_dict[context_id])      
                
    return valid_contexts

# Ensure the questions are clear, concise, and avoid ambiguity.

def generate_questions_from_valid_contexts(valid_contexts, qg_request, question_dir, save_name):
    question_id = 0
    generated = {}
    encoding_tasks = []
    count_invalid = 0

    # Enhanced system prompt with a structured example
    system_prompt = """
    You are an experienced Teacher preparing quiz questions. Your task is to create **three questions** and their answers based directly on the context provided. 
    Ensure that each question is:
    - Clear and focused on the context.
    - Relevant and unique (avoid repetition).
    - Formulated to encourage curiosity without prefacing phrases like "From this context" or "According to the text".

    Here is the required output format:
    - Q: [Question 1]
    - A: [Answer 1]
    - Q: [Question 2]
    - A: [Answer 2]
    - Q: [Question 3]
    - A: [Answer 3]

    Respond with just the questions and answers in the above format, no additional text.
    """

    user_prompt_template = """
    Based on the context below, create a quiz with **three questions** and their **answers** following this structure:
    Q: [Question 1]
    A: [Answer 1]
    Q: [Question 2]
    A: [Answer 2]
    Q: [Question 3]
    A: [Answer 3]

    Context:
    {sample_context}
    """
    
    for context_id, context in enumerate(valid_contexts):
        # Prepare the messages with improved system and user prompts
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_template.format(sample_context=context)}
        ]
        
        # Set up the LIMA model request
        request = schemas.VLLMRequest(
            messages=messages,
            max_tokens=qg_request['max_new_tokens'],
            temperature=qg_request['temperature'],
            top_k=qg_request['top_k'],
            top_p=qg_request['top_p']
        )

        # Schedule the encoding task
        encoding_task = celery_app.signature(
            "pipelines.celery_tasks.tasks.qg_task",
            args=(qg_request, request.dict())
        ).delay()
        
        # Initialize entry for this context
        generated[context_id] = {
            "context": context,
            "context_len": len(context),
            "generated": []
        }
        encoding_tasks.append((context_id, encoding_task))
    
    # Process each encoding task
    for context_id, encoding_task in encoding_tasks:
        try:
            with allow_join_result():
                generated_response = encoding_task.get()

            # Retrieve generated content
            question_ans_set = generated_response['output']['choices'][0]['message']['content']
            question_ans_set = re.split('\n+', question_ans_set.strip())

            # Parse questions and answers
            for idx in range(0, len(question_ans_set), 2):
                question = question_ans_set[idx].replace("Q: ", "").strip()
                answer = question_ans_set[idx + 1].replace("A: ", "").strip()

                # # Validation checks for question and context
                # if is_valid_context(context) and is_valid_question(question, context):
                #     generated[context_id]["generated"].append({
                #         "id": f"{question_id}_{save_name}",
                #         "question": question,
                #         "answers": [answer],
                #         "answer_len": len(answer)
                #     })
                #     question_id += 1
                # else:
                #     count_invalid += 1
                if is_valid_question(question, generated[context_id]["context"]):
                        generated[context_id]["generated"].append({
                            "id": f"{question_id}_{save_name}",
                            "question": question,
                            "answers": [answer],
                            "answer_len": len(answer)
                        })

        except Exception as e:
            print(f"Error processing context {context_id}: {e}")
    
    # Save generated questions to file
    with open(question_dir, "w") as f:
        json.dump(list(generated.values()), f, indent=1)

    print("Processing completed.", "Invalid Questions:", count_invalid)


@celery_app.task(ignore_result=False, bind=True)
def generate_questions(self,doc_id, qg_request, recreate=True):
    # Setup and preliminary checks
    db = SessionLocal()
    document = crud.document.get(db=db, id=doc_id)
    save_name = document.save_name.replace("/", "_")
    db.close()

    phrase_save_dir = os.path.join(settings.phrase_data_dir, save_name)

    s3 = None
    if settings.MULTI_HOSTS:
        s3 = s3_client.get_s3_instance()
        s3_client.download_file(phrase_save_dir, phrase_save_dir[1:], s3=s3)
    
    question_dir = os.path.join(settings.question_data_dir, save_name)
    if not recreate and os.path.exists(question_dir) and document.questions_generated:
        logger.info(f"{save_name} already has generated questions")
        db = SessionLocal()  # Update gen questions to false until done
        document = crud.document.get(db=db, id=doc_id)
        document.questions_generated = True
        db.commit()
        db.close()
        return
    
    data = None
    if os.path.exists(phrase_save_dir):
        with open(phrase_save_dir, "r") as f:
            data = json.load(f)['data']

    # Step 1: Validate Contexts
    valid_contexts = validate_contexts(data,qg_request)

    if not valid_contexts:
        logger.info("No valid contexts found.")
        return  

    # Step 2: Generate Questions from Validated Contexts
    generate_questions_from_valid_contexts(valid_contexts, qg_request,question_dir,save_name)
    
    if settings.MULTI_HOSTS:
        logger.info(f"Uploading {question_dir}")
        s3_client.upload_file(question_dir, question_dir[1:], s3=s3)
    # Mark document as having generated questions
    db = SessionLocal()
    document = crud.document.get(db=db, id=doc_id)
    document.questions_generated = True
    db.commit()
    db.close()
    logger.info(f"Questions generated for document: {save_name}")

@celery_app.task(ignore_result=False, bind=True)
def mrc_dataset(self, save_names, dataset_name):
    logger.info(f"save names: {len(save_names)}")
    logger.info(dataset_name)

    total_questions = 0

    mrc_dataset = {"data": []}
    for save_name in save_names:
        save_name = save_name.replace("/", "_")

        if not os.path.exists(os.path.join(settings.phrase_data_dir, save_name)):
            logger.info(f"{save_name} not parsed")
            continue
        # Generate or load previously generated questions

        question_dir = os.path.join(settings.question_data_dir, save_name)
        if not os.path.exists(question_dir):
            logger.info(f"{save_name} questions not generated")
            continue
        with open(question_dir, "r") as f:
            generated = json.load(f)

        # Append the questions
        def modify_answers(block, context):
            new_qas = {}
            new_qas["question"] = block["question"]
            new_qas["id"] = block["id"]
            new_qas["answer_len"] = block["answer_len"]
            new_qas["answers"] = [{
                "text": ans,
                "answer_start": context.index(ans) if ans in context else -1,
                "answer_len": len(ans),
            } for ans in block["answers"] if block["question"].strip() != ""]
            # } for ans in block["answers"] if "�" not in context and ans in context and block["question"].strip() != ""]
            return new_qas

        def qas(block):
            new_block = {}
            new_block["context"] = block["context"]
            new_block["title"] = save_name
            new_block["context_len"] = block["context_len"]
            new_block["qas"] = [modify_answers(generated, block["context"]) for generated in block["generated"]]
            new_block["qas"] = [qas for qas in new_block["qas"] if len(qas["answers"])]
            return new_block

        doc_questions = {
            "title": save_name,
            "paragraphs": [qas(block) for block in generated if block["context"].isascii()],
        }
        for para in doc_questions["paragraphs"]:
            total_questions += len(para["qas"])
        mrc_dataset["data"].append(doc_questions)

    logger.info(f"Total Questions in {dataset_name}: {total_questions}")

    dataset_dir = os.path.join(settings.training_data_dir, dataset_name + "_MRC")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    with open(os.path.join(dataset_dir, "dev.json"), "w+") as f:
        json.dump(mrc_dataset, f, indent=1)
    db = SessionLocal()
    group = crud.group.get(db=db, id=dataset_name.split("_")[1])
    if group:
        group.questions_generated = True
        db.commit()
    db.close()

    return dataset_name
