from nested_ner_metrics import standard_metric, flat_metric, nested_metric, nesting_metric

if __name__=='__main__':
    entities = [{"real": [("Body Part", 2, 2), ("Disease", 0, 2)], "pred": [("Body Part", 2, 2), ("Disease", 0, 2)]},
    {"real": [("Body Part", 2, 2), ("Disease", 0, 2)], "pred": [("Disease", 0, 2)]},
    {"real": [("Medication", 1,1)], "pred": [("Medication", 1,1)]}]
    print(standard_metric(entities))

    print(flat_metric(entities))

    print(nested_metric(entities))

    print(nesting_metric(entities))
    
            
            

          
          
                
                
               
           
           



    
  
