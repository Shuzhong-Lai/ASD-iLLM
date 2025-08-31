import json
import os

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print('save json at {}'.format(file_path))

def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_jsonl(file):
    data = []  
    with open(file, 'r', encoding='utf-8') as file:  
        for line in file:  
            try:  
                data.append(json.loads(line))  
            except json.JSONDecodeError as e:  
                print(f"Error decoding JSON: {e}")  
    return data  

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print('save jsonl at {}'.format(file_path))
    
def trans2standard_json(input_txt_path):
    def trans2role(spk):
        if spk == "医生":
            return "assistant"
        else:
            return "user"
    data = []
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            if line == "\n" or line == "":
                continue
            if '：' in line:
                line = line.replace('：', ':')
            spk, content = line.strip().split(':')
            data.append({
                "role": trans2role(spk).strip(),
                "content": content.strip()
            })
    standard_json = {}
    standard_json['messages'] = data
    return standard_json

# 可能存在的中英冒号问题修正
def check_txt(file_path):
    revise_file_content = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n" or line == "":
                continue
            if '：' in line:
                line = line.replace('：', ':')
            revise_file_content.append(line)
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in revise_file_content:
            f.write(line)

def read_txt(file_path):
    def trans2name(spk):
        if spk == "0":
            return "医生"
        else:
            return "儿童"
    output = ""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n" or line == "":
                continue
            if '：' in line:
                line = line.replace('：', ':')
            # print(line)
            spk, content = line.strip().split(':')
            output += trans2name(spk) + ": " + content + "\n"
    return output

# 将相同角色的连续消息合并成一个消息
def merge_json_sentence(data):
    new_messages = []  
    current_role = None  
    current_content = ""
    for message in data['messages']:
        if message['role'] == current_role:
            current_content += message['content']
        else:
            if current_role is not None:
                new_messages.append({
                    "role": current_role,
                    "content": current_content
                })
            current_role = message['role']
            current_content = message['content']
    
    if current_role is not None:
        new_messages.append({
            "role": current_role,
            "content": current_content
        })
    data['messages'] = new_messages
    return data

def trans2json(input_txt_path):
    data = []
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n" or line == "":
                continue
            spk, content = line.strip().split(':')
            data.append({
                "role": spk.strip(),
                "content": content.strip()
            })
    standard_json = {}
    standard_json['messages'] = data
    return standard_json

def trans2txt(json_data):
    output = ""
    for item in json_data['messages']:
        spk = item['role']
        content = item['content']
        temp = f"{spk}: {content}\n"
        output += temp
    return output