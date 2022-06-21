"""Metrics to assess performance on nested NER task given prediction
These metrics are still part of a work in progress, so they have not yet 
been officially published as a library.
"""

from nestednereval.utils import get_nestings
import numpy as np
from collections import defaultdict

def calculate_f1_score(tp, fp, fn):
  """Calculate F1 score using confusion matrix values.
    Args:
        tp (int): true positives 
        fp (int): false positives
        fn (int): false negatives
    Returns:
        precision: micro average precision
        recall: micro average recall
        f1: micro F1 score
    """
  precision = tp/(tp+fp) if (tp+fp)!=0 else 0
  recall = tp/(tp+fn) if (tp+fn)!=0 else 0
  f1 = (2*precision*recall)/(precision+recall) if (precision+recall)!=0 else 0
  return precision, recall, f1

def standard_metric(entities):
  """Calculate standard nested NER metric, which corresponds to the micro F1 score.
    Args:
        entities (list(dict)): List of dicts containing predicted and original entities.
    Returns:
        precision (int): micro average precision
        recall (int): micro average recall
        f1 (int): micro F1 score
        support (int): number of samples in partition
    """
  tp = 0
  fn = 0
  fp = 0
  support = 0

  for sent in entities:
    p = sent["pred"]
    g = sent["real"]

    for entity in p: 
        if entity in g: 
            tp+=1
        if entity not in g:
            fp+=1

    for entity in g:
        support+=1
        if entity not in p:
            fn+=1
  
  precision, recall, f1 = calculate_f1_score(tp, fp, fn)
  return precision, recall, f1, support

def length_metric(entities):
  support = 0
  
  from collections import defaultdict
  entities_length = defaultdict(lambda: defaultdict(int))
  entities_length_accuracy = defaultdict(int)
  entities_length_support = defaultdict(int)
  for sent in entities:
    p = sent["pred"]
    g = sent["real"]

    for entity in p: 
        if entity in g: 
            entities_length[entity[2]-entity[1]+1]["tp"]+=1
        if entity not in g:
            entities_length[entity[2]-entity[1]+1]["fp"]+=1

    for entity in g:
        entities_length_support[entity[2]-entity[1]+1]+=1
        support+=1
        if entity not in p:
            entities_length[entity[2]-entity[1]+1]["fn"]+=1
        else:
          entities_length_accuracy[entity[2]-entity[1]+1]+=1
  
  final_dict = defaultdict(int)
  for length, values in entities_length.items():
    precision, recall, f1 = calculate_f1_score(values["tp"], values["fp"], values["fn"])
    final_dict[length]=f1
  
  final_dict_acc = defaultdict(int)
  for k, v in entities_length_accuracy.items():
    final_dict_acc[k] = v/entities_length_support[k]

  return final_dict, final_dict_acc

def nesting_metric(entities):
  """Calculate micro F1 score over complete nestings (detecting inner and outer entities simultaneously).
    Args:
        entities (list(dict)): List of dicts containing predicted and original entities.
    Returns:
        precision (int): micro average precision
        recall (int): micro average recall
        f1 (int): micro F1 score
        support (int): number of samples in partition
    """
  
  tp = 0
  fn = 0
  fp = 0
  support = 0

  for sent in entities:
    pred_nestings = get_nestings(sent["pred"])
    test_nestings = get_nestings(sent["real"])
    
    for nesting in test_nestings:
      support+=1
      if nesting in pred_nestings:
        tp+=1
      else:
        fn+=1

    for nesting in pred_nestings:
      if nesting not in test_nestings:
        fp+=1
  
  precision, recall, f1 = calculate_f1_score(tp, fp, fn)
  return precision, recall, f1, support



def get_nestings_per_level(nestings):
  nestings_per_level = defaultdict(list)

  for nesting in nestings:
    entities_level = defaultdict(list)
    lvl = 0
    while(len(nesting)!=0):
      for entity1 in nesting:
        is_nested = False

        for entity2 in nesting:
          if entity1!=entity2:
            if (entity1[1]>entity2[1] and entity1[2]<=entity2[2]) or (entity1[1]>=entity2[1] and entity1[2]<entity2[2]):
          
              is_nested = True

        if not is_nested:
          entities_level[lvl].append(entity1)
      
      for e in entities_level[lvl]:
        nesting.remove(e)
      lvl+=1


    for k, v in entities_level.items():
      if k==0:
        for e in v:
          nestings_per_level[0].append(e)
      else:
        for e in v:
          nestings_per_level[1].append(e)

    
  return nestings_per_level
  


def nesting_level_metric_relaxed(entities):
  """Calculate micro F1 score over each level of nesting.
    Args:
        entities (list(dict)): List of dicts containing predicted and original entities.
    Returns: ToDO
    """

  max_depth = 2
  ar = [{"tp": 0, "fp": 0, "fn": 0} for i in range(max_depth)]
  support = defaultdict(int)

  for sent in entities:
    pred_nestings = get_nestings(sent["pred"])
    test_nestings = get_nestings(sent["real"])
  

    pred_levels = get_nestings_per_level(pred_nestings)
    test_levels = get_nestings_per_level(test_nestings)

    for k, v in pred_levels.items():
      
      for e in v:
        if e not in sent["real"]:
          ar[k]["fp"]+=1
        else:
          
          ar[k]["tp"]+=1
      
    
    for k, v in test_levels.items():
      for e in v:
        support[k]+=1
        if e not in sent["pred"]:
          ar[k]["fn"]+=1


  final_dict = defaultdict(int)
  for i, lvl in enumerate(ar):
    _, _, f1 = calculate_f1_score(lvl["tp"], lvl["fp"], lvl["fn"])
    final_dict[i]=(f1, support[i])
  return final_dict

def nesting_level_metric_strict(entities):
  """Calculate micro F1 score over each level of nesting.
    Args:
        entities (list(dict)): List of dicts containing predicted and original entities.
    Returns: ToDO
    """

  max_depth = 2
  ar = [{"tp": 0, "fp": 0, "fn": 0} for i in range(max_depth)]
  support = defaultdict(int)

  for sent in entities:
    pred_nestings = get_nestings(sent["pred"])
    test_nestings = get_nestings(sent["real"])
  

    pred_levels = get_nestings_per_level(pred_nestings)
    test_levels = get_nestings_per_level(test_nestings)

    for k, v in pred_levels.items():
 

      for e in v:
        if e not in test_levels[k]:
          ar[k]["fp"]+=1
        else:
          
          ar[k]["tp"]+=1
      
    

      
    
    for k, v in test_levels.items():
      for e in v:
        support[k]+=1
        if e not in pred_levels[k]:
          ar[k]["fn"]+=1


  final_dict = defaultdict(int)
  for i, lvl in enumerate(ar):
    precision, recall, f1 = calculate_f1_score(lvl["tp"], lvl["fp"], lvl["fn"])
    final_dict[i]=(f1, support[i])
  return final_dict


def flat_metric(entities):
  """Calculate micro F1 score over flat entities (not involved in any nesting).
    Args:
        entities (list(dict)): List of dicts containing predicted and original entities.
    Returns:
        precision (int): micro average precision
        recall (int): micro average recall
        f1 (int): micro F1 score
        support (int): number of samples in partition
    """
  tp = 0
  fn = 0
  fp = 0
  support = 0

  for sent in entities:
    pred_nestings = get_nestings(sent["pred"])
    pred_flat_entities = []
    for entity in sent["pred"]:
      is_nested = False
      for nesting in pred_nestings:
        if entity in nesting:
          is_nested = True
      if not is_nested:
        pred_flat_entities.append(entity)
    
    test_nestings = get_nestings(sent["real"])
    test_flat_entities = []
    for entity in sent["real"]:
      is_nested = False
      for nesting in test_nestings:
        if entity in nesting:
          is_nested = True
      if not is_nested:
        test_flat_entities.append(entity)

    for entity in test_flat_entities:
      support+=1
      if entity in sent["pred"]:
        tp+=1
      else:
        fn+=1

    for entity in pred_flat_entities:
      if entity not in sent["real"]:
        fp+=1

  precision, recall, f1 = calculate_f1_score(tp, fp, fn)
  return precision, recall, f1, support


def outer_metric(entities):
  """Calculate micro F1 score over outermost entities involved in nestings (longer entities).
    Args:
        entities (list(dict)): List of dicts containing predicted and original entities.
    Returns:
        precision (int): micro average precision
        recall (int): micro average recall
        f1 (int): micro F1 score
        support (int): number of samples in partition
    """
  tp = 0
  fn = 0
  fp = 0
  support = 0
  for sent in entities:
    pred_nestings = get_nestings(sent["pred"])
    test_nestings = get_nestings(sent["real"])
    
    for nesting in test_nestings:
      support+=1
      if nesting[0] in sent["pred"]:
        tp+=1
      else:
        fn+=1

    for nesting in pred_nestings:
      if nesting[0] not in sent["real"]:
        fp+=1
  
  precision, recall, f1 = calculate_f1_score(tp, fp, fn)
  return precision, recall, f1, support


def inner_metric(entities):
  """Calculate micro F1 score over inner entities (nested in other entities).
    Args:
        entities (list(dict)): List of dicts containing predicted and original entities.
    Returns:
        precision (int): micro average precision
        recall (int): micro average recall
        f1 (int): micro F1 score
        support (int): number of samples in partition
    """
  support = 0
  tp = 0
  fn = 0
  fp = 0

  for sent in entities:
    pred_nestings = get_nestings(sent["pred"])
    test_nestings = get_nestings(sent["real"])

    for nesting in test_nestings:
      for entity in nesting[1:]:
        support+=1
        if entity in sent["pred"]:
          tp+=1
        else:
          fn+=1

    for nesting in pred_nestings:
      for entity in nesting[1:]:
        if entity not in sent["real"]:
          fp+=1

  precision, recall, f1 = calculate_f1_score(tp, fp, fn)
  return precision, recall, f1, support



def nested_metric(entities):
  """Calculate micro F1 score over nested entities (inner or outer entities).
    Args:
        entities (list(dict)): List of dicts containing predicted and original entities.
    Returns:
        precision (int): micro average precision
        recall (int): micro average recall
        f1 (int): micro F1 score
        support (int): number of samples in partition
    """
  tp = 0
  fn = 0
  fp = 0
  support = 0

  for sent in entities:
    pred_nestings = get_nestings(sent["pred"])
    test_nestings = get_nestings(sent["real"])
    
    for nesting in test_nestings:
      for entity in nesting:
        support+=1
        if entity in sent["pred"]:
          tp+=1
        else:
          fn+=1

    for nesting in pred_nestings:
      for entity in nesting:
        if entity not in sent["real"]:
          fp+=1
    
  precision, recall, f1 = calculate_f1_score(tp, fp, fn)
  return precision, recall, f1, support

def nested_ner_metrics(entities):
    """Print all the metrics described above
    Args:
        entities (list(dict)): List of dicts containing predicted and original entities.
    Returns:
        None
    """
    standard_precision, standard_recall, standard_f1, support = standard_metric(entities)
    print(f'Standard metric\tPrecision: {np.round(standard_precision*100,2)}\tRecall: {np.round(standard_recall*100,2)}\tF1-Score: {np.round(standard_f1*100,2)}\tsupport: {support}')
    
    flat_precision, flat_recall, flat_f1, support = flat_metric(entities)
    print(f'Flat metric\tPrecision: {np.round(flat_precision*100,2)}\tRecall: {np.round(flat_recall*100,2)}\tF1-Score: {np.round(flat_f1*100,2)}\tsupport: {support}')
    
    inner_precision, inner_recall, inner_f1, support = inner_metric(entities)
    print(f'Inner metric\tPrecision: {np.round(inner_precision*100,2)}\tRecall: {np.round(inner_recall*100,2)}\tF1-Score: {np.round(inner_f1*100,2)}\tsupport: {support}')
    
    outer_precision, outer_recall, outer_f1, support = outer_metric(entities)
    print(f'Outer metric\tPrecision: {np.round(outer_precision*100,2)}\tRecall: {np.round(outer_recall*100,2)}\tF1-Score: {np.round(outer_f1*100,2)}\tsupport: {support}')
    
    nested_precision, nested_recall, nested_f1, support = nested_metric(entities)
    print(f'Nested metric\tPrecision: {np.round(nested_precision*100,2)}\tRecall: {np.round(nested_recall*100,2)}\tF1-Score: {np.round(nested_f1*100,2)}\tsupport: {support}')
    
    nesting_precision, nesting_recall, nesting_f1, support = nesting_metric(entities)
    print(f'Nesting metric\tPrecision: {np.round(nesting_precision*100,2)}\tRecall: {np.round(nesting_recall*100,2)}\tF1-Score: {np.round(nesting_f1*100,2)}\tsupport: {support}')
    

    