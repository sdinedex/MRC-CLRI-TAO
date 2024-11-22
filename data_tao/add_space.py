# import os
# import json

# def process_sentence(sentence):
#     """
#     在单词和标点符号之间插入空格
#     """
#     import re
#     # 在标点符号前后添加空格
#     sentence = re.sub(r"([.,!?;:])", r" \1 ", sentence)
#     # 替换多余的空格
#     sentence = re.sub(r"\s{2,}", " ", sentence)
#     return sentence.strip()

# def process_file(file_path):
#     """
#     处理单个 JSONL 文件
#     """
#     processed_data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             data['sentence'] = process_sentence(data['sentence'])
#             processed_data.append(data)
    
#     with open(file_path, 'w', encoding='utf-8') as f:
#         for data in processed_data:
#             f.write(json.dumps(data, ensure_ascii=False) + '\n')

# def process_folder(folder_path):
#     """
#     处理文件夹中的所有 JSONL 文件
#     """
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.jsonl'):
#             file_path = os.path.join(folder_path, file_name)
#             print(f"Processing {file_path}...")
#             process_file(file_path)

# # 设置文件夹路径
# folder_path = './MRC-CLRI-TAO/data_tao/finentity'  # 替换为你的文件夹路径
# process_folder(folder_path)

#------------------------------------------('Near-term outlook seen strong for Nifty'分离连接符号)
import re
import json

def preprocess_text(sentence):
    """
    对句子进行预处理，分离标点符号和连字符。
    """
    # 使用正则表达式将标点符号与单词分离
    sentence = re.sub(r'([.,!?;:()"\'])', r' \1 ', sentence)  # 分离常见标点符号
    sentence = re.sub(r'([\-])', r' \1 ', sentence)  # 分离连字符 '-'
    sentence = re.sub(r'([&])', r' \1 ', sentence)  # 分离和符号 '&'
    sentence = re.sub(r'\s{2,}', ' ', sentence)  # 去掉多余的空格
    return sentence.strip()

def process_dataset(input_file, output_file):
    """
    对整个数据集进行处理，分离标点符号，并保存到新的文件。
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    processed_data = []
    for item in data:
        # 对句子进行标点符号分离
        original_sentence = item['sentence']
        item['sentence'] = preprocess_text(item['sentence'])
        print(f"Original: {original_sentence}")
        print(f"Processed: {item['sentence']}")
        processed_data.append(item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Written: {item}")

# 输入文件路径
input_train_file = './MRC-CLRI-TAO/data_tao/sentfin/train.jsonl'
input_dev_file = './MRC-CLRI-TAO/data_tao/sentfin/dev.jsonl'
input_test_file = './MRC-CLRI-TAO/data_tao/sentfin/test.jsonl'

# 输出文件路径
output_train_file = './MRC-CLRI-TAO/data_tao/sentfin/train.jsonl'
output_dev_file = './MRC-CLRI-TAO/data_tao/sentfin/dev.jsonl'
output_test_file = './MRC-CLRI-TAO/data_tao/sentfin/test.jsonl'

# 对数据集进行批量处理
process_dataset(input_train_file, output_train_file)
process_dataset(input_dev_file, output_dev_file)
process_dataset(input_test_file, output_test_file)

