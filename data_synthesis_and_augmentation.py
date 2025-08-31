import json
import os
import sys
import numpy as np
import pandas as pd
from utils import save_json, load_json, check_txt, read_txt, trans2standard_json
from llm_api import generate_text_by_llm_api_via_openai  

SYN_SYS_PROMPT = """
## 角色设定
您是一位专注于儿童自闭症对话干预的专家，现在需要您基于提供的参考对话生成一段对话风格和说话方式相似、对话主题为{new_topic}的多轮对话，请在适当的地方插入[儿童无响应]以模拟儿童不回应的状态并保证对话自然流畅，以医生或儿童开头的形式逐行展示。
## 参考对话
{ref_dialogue}
## 生成对话
"""

topic_list = [        
        "如何穿衣",  
        "如何洗漱",  
        "如何洗手",  
        "如何洗澡",  
        "选择交通工具",
        "打招呼",  
        "介绍自己",  
        "学习社交礼仪",  
        "理解自我概念",
        "故事复述",  
        "日常生活分享",  
        "理解顺序和时间线",  
        "了解节日习俗",  
        "寓言故事的内容理解",
        "角色扮演医生和病人",  
        "角色扮演收营员和顾客",  
        "角色扮演餐馆服务员和顾客",  
        "性别认知",  
        "季节认知",
        "学习公共场所行为规范",
        "学习交通安全常识",
        "颜色",  
        "食物",  
        "水果",  
        "天气",  
        "动物",
        "职业"   
    ]

"""
1. Given the topic and dialogue content, enable the LLM to match the corresponding theme.
2. Under this theme, utilize DTT and NET to generate synthetic dialogues that share a similar style but differ in theme matter.
"""

def synthesis_data_via_api():
    input_dir = './sft_dataset/real_txt'
    input_txt_list = os.listdir(input_dir)
    index = 0
    for input_txt_path in sorted(input_txt_list):
        input_txt_path = os.path.join(input_dir, input_txt_path)
        basename = os.path.basename(input_txt_path)
        file_name, file_extension = os.path.splitext(basename)
        output_dir = './sft_dataset/synthesis_txt/' + file_name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ref_dialogue = read_txt(input_txt_path)
        for new_topic in topic_list:
            messages = [
                {
                    "role": "system",
                    "content": SYN_SYS_PROMPT.format(new_topic=new_topic, ref_dialogue=ref_dialogue)
                },
            ]
            model_name = "gpt-4.1"
            response = generate_text_by_llm_api_via_openai(messages, model_name)
            response_content = response.choices[0].message.content
            # print(response_content)
            output_dir_name = file_name + f'_{new_topic}' + '.txt'
            output_txt_file = os.path.join(output_dir, output_dir_name)
            with open(output_txt_file, 'w', encoding='utf-8') as f:
                f.write(response_content)
            print("{} : {} topic finished".format(index, new_topic))
        index += 1


if __name__ == '__main__':
    synthesis_data_via_api()