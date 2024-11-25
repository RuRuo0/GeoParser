import json
import random
import re
from collections import deque

with open('collinear.json', 'r', encoding='UTF-8') as file:
    collinear = json.load(file)

def process_construction_cdl(cons_list):
    """
    Random transformation of the construction_cdl of a single sample
    After transformation, the meaning of the sample representation remains unchanged in the formalgeo formal system
    """
    shape_list = []
    collinear_list = []
    cocircular_list = []
    for i in range(len(cons_list)):
        matches = re.search(r'^(.*?)\((.*?)\)', cons_list[i])

        type = matches.group(1)
        values = matches.group(2).split(',')
        if type == 'Shape':
            rotations = random.randint(1, len(values))
            rotated_elements = rotate_list(values, rotations=rotations)
            new_item = f"{type}({','.join(rotated_elements)})"
            shape_list.append(new_item)
        elif type == 'Collinear':
            str = values[0]
            rotated_str = str if random.choice([True, False]) else str[::-1]
            new_item = f"{type}({rotated_str})"
            collinear_list.append(new_item)
        elif type == 'Cocircular':
            if len(values) == 1:
                new_item = f"{type}({values[0]})"
            else:
                rotations = random.randint(1, len(values[1]))
                values[1] = rotate_string(values[1], rotations=rotations)
                new_item = f"{type}({','.join(values)})"
            cocircular_list.append(new_item)

    # Scrambles the order of each type of substatement in a composition statement
    random.shuffle(shape_list)
    random.shuffle(collinear_list)
    random.shuffle(cocircular_list)
    return ','.join(shape_list + collinear_list + cocircular_list)

def rotate_list(elements, rotations=1):
    queue = deque(elements)
    queue.rotate(rotations)
    return list(queue)

def rotate_string(str, rotations=1):
    return str[rotations:] + str[:rotations]

def process_img_cdl(proId, imgcdl_list):
    """
    Random transformation of the image_cdl of a single sample
    After transformation, the meaning of the sample representation remains unchanged in the formalgeo formal system
    """
    if not imgcdl_list:
        return ''
    new_imgcdl_list = []
    for i in range(len(imgcdl_list)):
        matches = re.search(r'^([^(]+)\((.*)\)$', imgcdl_list[i])

        type = matches.group(1)
        values = matches.group(2)
        # Equations with addition, subtraction, multiplication, and division are not easy to parse,
        # so mark them as of type 'Equal-a'
        if type == 'Equal' and re.search(r'Add|Sub|Mul|Div', values):
            type == 'Equal-a'
        else:
            values = values.split(',')

        if type == 'Equal':
            lineNum = sum('LengthOfLine' in e for e in values)
            angleNum = sum('MeasureOfAngle' in e for e in values)
            if lineNum == 1:
                line = re.split(r'[()]', values[0])[1]
                line = line if random.choice([True, False]) else line[::-1]
                new_item = f"{type}(LengthOfLine({line}),{values[1]})"
            elif angleNum == 1:
                angle = list(re.split(r'[()]', values[0])[1])
                p = getPCollinearRandom(proId,angle[:2][::-1])[1]
                l = getPCollinearRandom(proId, angle[-2:])
                new_item = f"{type}(MeasureOfAngle({p+l}),{values[1]})"
            elif lineNum == 2:
                l1 = re.split(r'[()]', values[0])[1]
                l2 = re.split(r'[()]', values[1])[1]
                l1 = l1 if random.choice([True, False]) else l1[::-1]
                l2 = l2 if random.choice([True, False]) else l2[::-1]
                if random.choice([True, False]):
                    new_item = f"{type}(LengthOfLine({l1}),LengthOfLine({l2}))"
                else:
                    new_item = f"{type}(LengthOfLine({l2}),LengthOfLine({l1}))"
            elif angleNum == 2:
                angle1 = list(re.split(r'[()]', values[0])[1])
                angle2 = list(re.split(r'[()]', values[1])[1])
                p1 = getPCollinearRandom(proId, angle1[:2][::-1])[1]
                l1 = getPCollinearRandom(proId, angle1[-2:])
                p2 = getPCollinearRandom(proId, angle2[:2][::-1])[1]
                l2 = getPCollinearRandom(proId, angle2[-2:])
                if random.choice([True, False]):
                    new_item = f"{type}(MeasureOfAngle({p1+l1}),MeasureOfAngle({p2+l2}))"
                else:
                    new_item = f"{type}(MeasureOfAngle({p2+l2}),MeasureOfAngle({p1+l1}))"
            else:
                new_item = f"{type}({','.join(values)})"
        elif type == 'Equal-a':
            new_item = f"Equal({values})"
        elif type == 'PerpendicularBetweenLine':
            l1 = getPCollinearRandom(proId,list(values[0][::-1]))
            l2 = getPCollinearRandom(proId, list(values[1][::-1]))
            new_item = f"{type}({l1[::-1]},{l2[::-1]})"
        elif type == 'ParallelBetweenLine':
            l1 = getLCollinearRandom(proId, list(values[0]))
            l2 = getLCollinearRandom(proId, list(values[1]))
            new_item = f"{type}({l1},{l2})" if random.choice([True, False]) else f"{type}({l2[::-1]},{l1[::-1]})"

        new_imgcdl_list.append(new_item)

    # Shuffles the order of image conditions
    random.shuffle(new_imgcdl_list)
    return ','.join(new_imgcdl_list)

def getPCollinearRandom(proId, line):
    """"
    The collinear rays with A as the left endpoint in the direction of ray AB are randomly obtained
    Enter the question id, and a two-point line representing a list such as ['A','B']
    Returns as a string, e.g. "AC"
    """
    clist = collinear[proId]
    if clist is None:
        return ''.join(line)
    for item in clist:
        if line[0] in item and line[1] in item:
            item = item if item.index(line[0]) < item.index(line[1]) else item[::-1]
            line[1] = random.choice(item[item.index(line[0])+1:])
            break
    return ''.join(line)

def getLCollinearRandom(proId, line):
    """"
    Lines in the same direction in the direction of the straight line AB are randomly obtained, including AB
    Enter the question id, and a two-point line representing a list such as ['A','B']
    Returns as a string such as "CD"
    """
    clist = collinear[proId]
    if clist is None:
        return ''.join(line)
    for item in clist:
        if line[0] in item and line[1] in item:
            item = item if item.index(line[0]) < item.index(line[1]) else item[::-1]
            i1, i2 = random.sample(range(len(item)), 2)
            line = [item[i1], item[i2]] if i1 < i2 else [item[i2], item[i1]]
            break
    return ''.join(line)

def augment_data(original_data, n):
    """
    N times enhanced for raw data for construction_cdl and image_cdl
    The number of new datasets is n+1 times the size of the original
    """
    augmented_data = {}

    for key, value in original_data.items():
        augmented_data[key] = {0: value}
        for i in range(1, n + 1):
            construction_items = re.split(r',(?![^()]*\))', value['construction_cdl'][0].replace(' ', ''))
            imgcdl_items = re.findall(r'\w+\((?:[^()]|\([^)]*\))+\)', value['image_cdl'][0].replace(' ', ''))
            try:
                new_conscdl = process_construction_cdl(construction_items)
                new_imgcdl = process_img_cdl(key, imgcdl_items)
            except Exception as e:
                print(f"key: {key}")
                print(f"value: {construction_items,imgcdl_items}")
                # Create a new entry in the augmented data
            augmented_data[key][i] = value.copy()
            augmented_data[key][i]['construction_cdl'] = [new_conscdl]
            augmented_data[key][i]['image_cdl'] = [new_imgcdl]
    return augmented_data

if __name__ == '__main__':
    with open('dataset/train.json', 'r', encoding='UTF-8') as file:
        data = json.load(file)

    augmented_data = augment_data(data,2)

    # Offline data enhancement
    with open('dataset/train_aug.json', 'w', encoding='UTF-8') as file:
        json.dump(augmented_data, file, indent=4, ensure_ascii=False)