"""Metrics to assess performance on nested NER task given prediction
These metrics are still part of a work in progress, so they have not yet 
been officially published as a library.
"""

from nested_entities_utils import get_nestings
import numpy as np

def calculate_f1_score(tp, fp, fn):
  p = tp/(tp+fp) if (tp+fp)!=0 else 0
  r = tp/(tp+fn) if (tp+fn)!=0 else 0
  f1 = (2*p*r)/(p+r) if (p+r)!=0 else 0
  return p, r, f1

def standard_metric(entities):
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

def nesting_metric(entities):
  
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

def flat_metric(entities):
  
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
    
    