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

    return json_dict

#==============================================
#                   Main
#==============================================
if __name__ == '__main__':
    json_name = str(sys.argv[1])
    df = json_to_pandas(json_name)
    print df
