import json
import numpy as np
import pandas as pd
from tableschema import Table

class DataCleaner(object):
    """DataCleaner has a set of functions to automatically clean 
    tabular data. It does things like
    - infer data type (numeric, categorical, date, boolean)
    - infer missing values
    - automatically describe data in simple (lay terms)
    - automatically run some inference / correlation algos
    """
    def __init__(self, csv_file):
        self.fname_ = csv_file
        self.table_ = Table(self.fname_)
        
    def InferType(self):
        # First pass: use tableschema inference to get syntactic shema
        # syntactic schema: integer, float, string, date
        syn_schema = self.table_.infer()

        # convert to json
        syn_schema_json = json.dumps(syn_schema, indent = 4)
        print("json:" + str(syn_schema_json))
        

        # 
        # semantic schema for a column:
        #   numeric: integer, float, integer range, float range, date, date range
        #   categorical: boolean, enum
        #   string: person name, geo location, etc
        #
        # other attributes for a column: is_identifier, is_unique, missing_id, human_description
        #
        # Signals from name (i.e., things like "id" "number" "num" "count" "code" "name")
        # - numeric-ness: "num" etc
        # - categorical-ness
        # - id-ness: "id" "code"
        # - string-ness: "name"
        # 
        # Signals from distribution (for syntactic types integer, float)
        # - basic stats: min, max, median, mode, etc
        # - gaussian fit?
        # - smoothness / monotonicity: bumpiness denotes categorical (code)
        # 
        # Signals from distribution (for syntactic types integer, string, date)
        # - num_distinct_val
        # - num_distinct_val_for_90%
        # - boolean-ness: "Yes" "No"; 0/1; up/down, etc
        # - nameness
        # - camelcase: denotes categorical
        # 
        return syn_schema



def test():
   pass 

        
def main():
    # Run test
    test()
    cleaner = DataCleaner("/home/maythakur/mt/uci_diabetes_data/dataset_diabetes/diabetic_data.csv")
    print(str(cleaner.InferType()))


        
if __name__ == "__main__":
    # execute only if run as a script
    main()
