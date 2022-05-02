"""Useful functions for reading files in IOB2 format 
and obtaining flat and nested entities.
"""
import re
from tqdm import tqdm
import warnings
from collections import defaultdict

def read_iob2_prediction_file(filepath):
    """Read files in IOB2 format and obtain a list of tags associated with each sentence.
    Args:
        filepath (string): filepath of IOB2 file. (first column token, second column real tag, third column predicted tag)
    Returns:
        list: list of dicts, which contains predicted and original entities.
    Example:
        Given the following IOB2 example
            Barack  B-PER B-PER
            Obama   I-PER I-PER
            is  O O
            a   O O
            politician  O O 
        >>> filepath = 'prediction.iob2'
        >>> read_iob2_file(filepath)
            [{"real": [(PER, 0, 1)], "pred": (PER, 0, 1)]}]
    """

    iob2_file = open(filepath, 'r').read()
    iob2_file = re.sub(r'\n\s*\n', '\n\n', iob2_file)
    chunks = []
    #for i, sent in enumerate(tqdm(iob2_file.split('\n\n'), desc="Getting entities from sentences in IOB2 file")):
    for i, sent in enumerate(iob2_file.split('\n\n')):
        real_tags = []
        pred_tags = []
        for line in sent.splitlines():
            real_tag = line.split()[1]
            pred_tag = line.split()[2]
            real_tags.append(real_tag)
            pred_tags.append(pred_tag)
        
        chunks.append({"real": get_entities(real_tags), "pred": get_entities(pred_tags)})
    return chunks

def merge_predictions(entities):
    """Merge predictions on different entity types into one data structure for retrieving nested entities.
    Args:
        entities (list of list of dics): List of dicts with original entities and predictions per entity type
    Returns:
        list: list of dicts, which contains predicted and original entities.
    Example:
        >>> entities = [[{"real": [(PER, 0, 1)], "pred": [(PER, 0, 1)]}], [{"real": [], "pred": [(ORG, 0, 1)]}]]
        >>> merge_predictions(entities)
            [{"real": [(PER, 0, 1)], "pred": [(PER, 0, 1), (ORG, 0, 1)]}]
    """
    chunks = []
    for i, entity_type in enumerate(entities):
        for j, entities_per_sentence in enumerate(entity_type):
            if i==0:
                chunks.append(entities_per_sentence)
                continue
      
            chunks[j]["real"].extend(entities_per_sentence["real"])
            chunks[j]["pred"].extend(entities_per_sentence["pred"])
    return chunks

def get_nestings(entities):
    """Gets nestings found per sentence.
    Args:
        list (list of dicts): list with the list of real and pred entities per sentence.
    Returns:
        list: list of nestings per sentences (each nesting is a list of entities).
    Example:
        >>> 
        >>> seq = 
        >>> 
        
    """
    
    nestings = [] 
    total = []

    for e1 in entities:
        is_outer = True 
        possible_nested_entity = [e1]
        
        for e2 in entities:
            if e1!=e2:
                s_e1 = e1[1]
                e_e1 = e1[2]
                s_e2 = e2[1]
                e_e2 = e2[2]
                
                if ((s_e1>s_e2 and e_e1<e_e2) or (s_e1==s_e2 and e_e1<e_e2) or (s_e1>s_e2 and e_e1==e_e2)):
                    is_outer = False 
                
                if (s_e2>=s_e1 and e_e2<=e_e1):
                    if e1 not in total:
                        total.append(e1)
                    if e2 not in total:
                        total.append(e2)
                    possible_nested_entity.append(e2)
            
        if len(possible_nested_entity)==1:
            is_outer = False
        
        if is_outer:
            possible_nested_entity.sort(key=lambda x: (x[2]-x[1], x[0]), reverse=True)
            if possible_nested_entity not in nestings:
                nestings.append(possible_nested_entity)
    return nestings




def get_entities(seq, suffix=False):
    """Gets entities from sequence.
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """

    def _validate_chunk(chunk, suffix):
        if chunk in ['O', 'B', 'I', 'E', 'S']:
            return

        if suffix:
            if not chunk.endswith(('-B', '-I', '-E', '-S')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

        else:
            if not chunk.startswith(('B-', 'I-', 'E-', 'S-')):
                warnings.warn('{} seems not to be NE tag.'.format(chunk))

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit('-', maxsplit=1)[0] or '_'
        else:
            tag = chunk[0]
            type_ = chunk[1:].split('-', maxsplit=1)[-1] or '_'

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks

def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end

def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start

if __name__=='__main__':
    anat_chunks = read_iob2_prediction_file('mlc-flair/clinical_trials_ANAT/test.tsv')
    chem_chunks = read_iob2_prediction_file('mlc-flair/clinical_trials_CHEM/test.tsv')
    entities = merge_predictions([anat_chunks, chem_chunks])
    print(entities[0]["real"])
    print(entities[0]["pred"])
    real_nestings, pred_nestings = get_nestings(entities)
    
    for r, p in zip(real_nestings, pred_nestings):
        if len(r)!=0:
            print(r)
            print(p)
