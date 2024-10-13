import argparse
import json
import os
import random
import logging
import pickle
import torch
import h5py
import numpy as np
from tqdm import tqdm
from time import time
from celery import chain, group
import hashlib
from densephrases.utils.embed_utils import int8_to_float, float_to_int8
from app.core.celery_app import celery_app
from transformers import AutoTokenizer, AutoModel, DistilBertModel
from celery.result import allow_join_result
from app.core.config import settings
import scipy.sparse as sp

parsed_count = 0


def get_paragraphs(each, delimiter=" [PAR] "):
    return each["context"].split(delimiter)


logger = logging.getLogger(__name__)


def feature_to_text(feature):
    text = ""
    for token in feature.tokens:
        if token in ['[CLS]', '[SEP]']:
            continue
        if token.startswith("##"):
            text += token[2:]
        else:
            text += " " + token
    return text.strip()

#here

# def start_store_paragraph_embeddings(features, doc_save_name, default_dump_save_dir, recreate=False):

#     encoding_tasks = []

#     output_dir = os.path.join(default_dump_save_dir, doc_save_name)
#     dump_file = os.path.join(output_dir, f"dump/phrase/{doc_save_name}.hdf5")  # this is dependednt
#     parapraph_dump_file = os.path.join(output_dir, f"dump/custom/{doc_save_name}_v7.hdf5")
#     if os.path.exists(parapraph_dump_file) and not recreate:
#         phrase_dump = h5py.File(dump_file, "r")
#         paragraph_dump = h5py.File(parapraph_dump_file, "r+")
#         groups = paragraph_dump.keys()
#         if len(groups) and len(groups) == len(phrase_dump.keys()):
#             if all(all(required in list(paragraph_dump[group].keys()) for required in ['para_embeddings', 'repeat_count']) for group in groups):
#                 print(f"Paragraph embeddings already present for {doc_save_name}")
#                 paragraph_dump.close()
#                 phrase_dump.close()
#                 return []
#     else:
#         os.makedirs(os.path.dirname(parapraph_dump_file), exist_ok=True)

#         phrase_dump = h5py.File(dump_file, "r")
#         paragraph_dump = h5py.File(parapraph_dump_file, "w")

#     doc2features = {}
#     for feature in features:
#         if feature.doc_idx not in doc2features:
#             doc2features[feature.doc_idx] = []
#         doc2features[feature.doc_idx].append(feature)

#     for doc_idx, doc_group in phrase_dump.items():
#         if doc_idx in paragraph_dump.keys():
#             dg = paragraph_dump[doc_idx]
#         else:
#             dg = paragraph_dump.create_group(doc_idx)

#         dg.attrs["offset"] = doc_group.attrs["offset"]
#         dg.attrs["scale"] = doc_group.attrs["scale"]

#         assert doc_idx in doc2features
#         para_list = [feature_to_text(feature) for feature in doc2features[doc_idx]]
#         feature2par_idx = [feature.par_idx for feature in doc2features[doc_idx]]
#         # para_list = get_paragraphs({"context": doc_group.attrs["context"]})
#         encoding_task = None

#         if "para_embeddings" not in dg.keys() or recreate:

#             encoding_task = celery_app.signature(
#                 "app.transformer_common_tasks.encode",
#                 args=(para_list,),
#             ).delay()

#         encoding_tasks.append(encoding_task)

#         if "repeat_count" not in dg.keys():
#             global parsed_count
#             para_count = len(get_paragraphs({"context": doc_group.attrs["context"]}))

#             context = doc_group.attrs["context"]
#             word2char_start = doc_group["word2char_start"]
#             f2o_start = doc_group["f2o_start"]

#             def get_context(word_id):
#                 global parsed_count
#                 start_pos = word2char_start[f2o_start[word_id]].item()
#                 context_count = context.count(" [PAR] ", parsed_count, start_pos)
#                 parsed_count = start_pos
#                 return context_count

#             parsed_count = 0
#             tagged_context = -1
#             context_id = 0
#             repeat_count = []

#             for word_id in range(len(doc_group['start'])):
#                 context_count = get_context(word_id)
#                 context_id += context_count
#                 while context_id > tagged_context:
#                     repeat_count.append(0)
#                     tagged_context += 1
#                 repeat_count[-1] += 1

#             while len(repeat_count) < para_count:
#                 repeat_count.append(0)

#             print(repeat_count)

#             dg["repeat_count"] = np.array(repeat_count)
#             dg["feature2par_idx"] = np.array(feature2par_idx)

#     print(paragraph_dump.keys())
#     paragraph_dump.close()
#     phrase_dump.close()

#     print("done")
#     return encoding_tasks


def start_store_paragraph_embeddings_initial(features, doc_save_name, default_dump_save_dir, recreate=False):
    """
    This part handles the expansive embeddings, which can be processed 
    before phrase dumping. It processes based on the feature list.
    """
    encoding_tasks = []
    output_dir = os.path.join(default_dump_save_dir, doc_save_name)
    # dump_file = os.path.join(output_dir, f"dump/phrase/{doc_save_name}.hdf5")  
    parapraph_dump_file = os.path.join(output_dir, f"dump/custom/{doc_save_name}_v7.hdf5")
    
    # here what would be the new condition for already present paragraph embeddings
    # if os.path.exists(parapraph_dump_file) and os.path.exists(dump_file) and not recreate:
    #     phrase_dump = h5py.File(dump_file, "r")
    #     paragraph_dump = h5py.File(parapraph_dump_file, "r+")
    #     groups = paragraph_dump.keys()
    #     if len(groups) and len(groups) == len(phrase_dump.keys()):
    #         if all(all(required in list(paragraph_dump[group].keys()) for required in ['para_embeddings', 'repeat_count']) for group in groups):
    #             print(f"Paragraph embeddings already present for {doc_save_name}")
    #             paragraph_dump.close()
    #             phrase_dump.close()
    #             return []
    # else:
    
    os.makedirs(os.path.dirname(parapraph_dump_file), exist_ok=True)
    paragraph_dump = h5py.File(parapraph_dump_file, "w")
    
    doc2features = {}
    for feature in features:
        if feature.doc_idx not in doc2features:
            doc2features[feature.doc_idx] = []
        doc2features[feature.doc_idx].append(feature)

    for doc_idx, doc_features in doc2features.items():
        if doc_idx in paragraph_dump.keys():
            dg = paragraph_dump[doc_idx]
        else:
            dg = paragraph_dump.create_group(doc_idx)

        assert doc_idx in doc2features
        para_list = [feature_to_text(feature) for feature in doc2features[doc_idx]]
        feature2par_idx = [feature.par_idx for feature in doc2features[doc_idx]]
        encoding_task = None

        if "para_embeddings" not in dg.keys() or recreate:
            encoding_task = celery_app.signature(
                "app.transformer_common_tasks.encode",
                args=(para_list,),
            ).delay()
            encoding_tasks.append(encoding_task)
        
        dg["feature2par_idx"] = np.array(feature2par_idx)

    paragraph_dump.close()
    print("Expansive embeddings stored (independent of phrase dump).")
    return encoding_tasks



def start_store_paragraph_embeddings_with_phrases(features, doc_save_name, default_dump_save_dir, recreate=False):
    """
    This part is dependent on the phrase dump and handles the 
    calculation of the repetition count using the phrase results.
    """
    output_dir = os.path.join(default_dump_save_dir, doc_save_name)
    dump_file = os.path.join(output_dir, f"dump/phrase/{doc_save_name}.hdf5")
    parapraph_dump_file = os.path.join(output_dir, f"dump/custom/{doc_save_name}_v7.hdf5")

    if not os.path.exists(dump_file):
        print(f"Phrase dump file not found: {dump_file}")
        return []
    
    phrase_dump = h5py.File(dump_file, "r")
    paragraph_dump = h5py.File(parapraph_dump_file, "r+")
    
    doc2features = {}
    for feature in features:
        if feature.doc_idx not in doc2features:
            doc2features[feature.doc_idx] = []
        doc2features[feature.doc_idx].append(feature)

    for doc_idx, doc_group in phrase_dump.items():
        if doc_idx in paragraph_dump.keys():
            dg = paragraph_dump[doc_idx]
        else:
            dg = paragraph_dump.create_group(doc_idx)
        
        dg.attrs["offset"] = doc_group.attrs["offset"]
        dg.attrs["scale"] = doc_group.attrs["scale"]

        if "repeat_count" not in dg.keys():
            global parsed_count
            para_count = len(get_paragraphs({"context": doc_group.attrs["context"]}))

            context = doc_group.attrs["context"]
            word2char_start = doc_group["word2char_start"]
            f2o_start = doc_group["f2o_start"]

            def get_context(word_id):
                global parsed_count
                start_pos = word2char_start[f2o_start[word_id]].item()
                context_count = context.count(" [PAR] ", parsed_count, start_pos)
                parsed_count = start_pos
                return context_count

            parsed_count = 0
            tagged_context = -1
            context_id = 0
            repeat_count = []

            for word_id in range(len(doc_group['start'])):
                context_count = get_context(word_id)
                context_id += context_count
                while context_id > tagged_context:
                    repeat_count.append(0)
                    tagged_context += 1
                repeat_count[-1] += 1

            while len(repeat_count) < para_count:
                repeat_count.append(0)

            dg["repeat_count"] = np.array(repeat_count)

    paragraph_dump.close()
    phrase_dump.close()
    
    print("Repetition count calculated (dependent on phrase dump).")
    return []

    
def end_store_paragraph_embeddings(doc_save_name, encoding_tasks, default_dump_save_dir, recreate=False):

    output_dir = os.path.join(default_dump_save_dir, doc_save_name)
    dump_file = os.path.join(output_dir, f"dump/phrase/{doc_save_name}.hdf5")
    parapraph_dump_file = os.path.join(output_dir, f"dump/custom/{doc_save_name}_v7.hdf5")

    phrase_dump = h5py.File(dump_file, "r")
    paragraph_dump = h5py.File(parapraph_dump_file, "r+")

    for doc_group_id, (doc_idx, doc_group) in enumerate(phrase_dump.items()):

        if len(encoding_tasks):

            dg = paragraph_dump[doc_idx]

            encoding_task = encoding_tasks[doc_group_id]

            if encoding_task is not None:

                with allow_join_result():
                    para_embedding = encoding_task.get()

                para_embedding = np.array(para_embedding)

                dg["para_embeddings"] = float_to_int8(para_embedding, dg.attrs["offset"], dg.attrs["scale"])

    paragraph_dump.close()
    phrase_dump.close()

    print("done")


def start_store_sparse_embeddings(features, doc_save_name, default_dump_save_dir, recreate=False):

    output_dir = os.path.join(default_dump_save_dir, doc_save_name)
    parapraph_dump_file = os.path.join(output_dir, f"dump/custom/{doc_save_name}_sparse.hdf5")
    if os.path.exists(parapraph_dump_file) and not recreate:
        paragraph_dump = h5py.File(parapraph_dump_file, "r+")
        if all(required in list(paragraph_dump['sparse'].keys()) for required in ['data', 'row', 'col']):
            print(f"Sparse embeddings already present for {doc_save_name}")
            paragraph_dump.close()
            return None
    else:
        os.makedirs(os.path.dirname(parapraph_dump_file), exist_ok=True)

        paragraph_dump = h5py.File(parapraph_dump_file, "w")

    para_list = []
    records = []

    for feature_idx, feature in enumerate(features):
        start_position = min(feature.token_to_orig_map.values())
        end_position = max(feature.token_to_orig_map.values())
        if end_position - start_position <= 7:
            continue
        para_list.append(feature_to_text(feature))
        records.append({"start_para_id": feature.par_idx, "end_para_id": feature.par_idx, "start_pos": start_position, "end_pos": end_position, "title": [feature.title]})

    # with open('/tenant/logs/features.json', 'w') as f:
    #     json.dump(features, f, indent=4)

    if not settings.UPDATE_DOC_STATUS:
        with open('/tenant/logs/sparse.json', 'w') as f:
            json.dump([(para, record) for para, record in zip(para_list, records)], f, indent=4)

    records = [record for i, record in enumerate(records) if len(para_list[i].split()) > 15]
    para_list = [para for i, para in enumerate(para_list) if len(para.split()) > 15]

    if "sparse" in paragraph_dump.keys():
        dg = paragraph_dump["sparse"]
    else:
        dg = paragraph_dump.create_group("sparse")

    with open(os.path.join(output_dir, f"dump/custom/{doc_save_name}_sparse_records.pkl"), "wb") as f:
        pickle.dump(records, f)

    encoding_tasks = []
    batch_lens = []
    for para_index in range(0, len(para_list), settings.SPARSE_BATCH):
        batch_input = para_list[para_index:min(para_index + settings.SPARSE_BATCH, len(para_list))]
        encoding_task = celery_app.signature(
            "app.transformer_common_tasks.sparse_matrix",
            args=(batch_input,),
        ).delay()
        encoding_tasks.append(encoding_task)
        batch_lens.append(len(batch_input))

    paragraph_dump.close()
    print("done")
    return encoding_tasks, batch_lens


def end_store_sparse_embeddings(doc_save_name, encoding_tasks, batch_lens, default_dump_save_dir, recreate=False):

    output_dir = os.path.join(default_dump_save_dir, doc_save_name)
    parapraph_dump_file = os.path.join(output_dir, f"dump/custom/{doc_save_name}_sparse.hdf5")

    paragraph_dump = h5py.File(parapraph_dump_file, "r+")

    dg = paragraph_dump['sparse']

    if encoding_tasks is not None:
        sparse_embeddings = []
        for encoding_task, batch_len in tqdm(zip(encoding_tasks, batch_lens), desc="Joining sparse results"):
            with allow_join_result():
                sparse_ret = encoding_task.get()
                sparse_embeddings.append(sp.csr_matrix((sparse_ret['data'], (sparse_ret['row'], sparse_ret['col'])), shape=(batch_len, 30522)))

        sparse_embeddings = sp.vstack(sparse_embeddings).tocoo()

        dg["data"] = sparse_embeddings.data
        dg["row"] = sparse_embeddings.row
        dg["col"] = sparse_embeddings.col

    paragraph_dump.close()
    print("done")


if __name__ == "__main__":
    doc_save_names = ['doc_6acc7910-bc90-4561-bd2d-4e76761188d2_Hormone']
    # doc_save_names = ['doc_a51dfa16-c649-4c90-90e7-a1cb75cdcd71_kyc-norms', 'doc_22a89d30-127c-4f64-a357-f4aaeb1ebdae_Master Circular - Detection and Impounding of Counterfeit Notes']
    embeddings_tasks = start_store_paragraph_embeddings_initial(doc_save_names, '/tenant/document_dumps/', recreate=False)

    start_store_paragraph_embeddings_with_phrases(doc_save_names, '/tenant/document_dumps/', recreate=False)
    end_store_paragraph_embeddings(doc_save_names, embeddings_tasks, '/tenant/document_dumps/', recreate=False)
