import json
import os
import re

data_dir = ''

result_dict = {}

for filename in os.listdir(data_dir):

    id = os.path.splitext(filename)[0]

    # Build the JSON file path
    json_file_path = os.path.join(data_dir, id + '.json')

    # If the JSON file exists, read construction_cdl
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r', encoding='UTF-8') as json_file:
            data = json.load(json_file)
            construction_cdl = data.get('construction_cdl')
            collinear_list = [re.split(r'[()]', value)[1] for value in construction_cdl if "Collinear" in value]
    else:
        print(f'{json_file_path}不存在')

    if collinear_list:
        result_dict[id] = collinear_list
    else:
        result_dict[id] = None

if __name__ == '__main__':

    # Save the result dictionary as a JSON file
    output_file = "collinear.json"
    with open(output_file, 'w', encoding='UTF-8') as f:
        json.dump(result_dict, f, indent=4, ensure_ascii=False)