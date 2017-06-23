"""
    Name:           json_to_csv.py
    Created:        23/6/2017
    Description:    Convert json annotations to csv tables.
"""
#==============================================
#                   Modules
#==============================================
import sys
import json
import pandas as pd
#==============================================
#                   Files
#==============================================


#==============================================
#                   Functions
#==============================================
def json_to_pandas(json_name, verbose=1):
    """
    Convert json file to pandas.
    """

    with open(json_name, 'r') as iOF:
        json_dict = json.loads(iOF.read())

    return pd.DataFrame(json_dict["annotations"])

#==============================================
#                   Main
#==============================================
if __name__ == '__main__':
    json_name = str(sys.argv[1])
    df = json_to_pandas(json_name)
    csv_name = '.'.join(json_name.split('.')[:-1])+'.csv'
    df.to_csv(csv_name, index=False, header=True)
