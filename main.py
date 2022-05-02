from nestednereval.metrics import standard_metric
from nestednereval.metrics import flat_metric
from nestednereval.metrics import nested_metric
from nestednereval.metrics import nesting_metric


if __name__=='__main__':
    
    entities = [{"real": [("Body Part", 2, 2), ("Disease", 0, 2)], "pred": [("Body Part", 2, 2), ("Disease", 0, 2)]},
    {"real": [("Body Part", 2, 2), ("Disease", 0, 2)], "pred": [("Disease", 0, 2)]},
    {"real": [("Medication", 1,1)], "pred": [("Medication", 1,1)]}]

    print(standard_metric(entities))

    print(flat_metric(entities))

    print(nested_metric(entities))

    print(nesting_metric(entities))
    
            
            

          
          
                
                
               
           
           



    
  
