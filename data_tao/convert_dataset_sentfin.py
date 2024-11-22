# from difflib import SequenceMatcher
# import pandas as pd
# import json
# import re

# # 读取原始数据
# data = pd.read_csv('./MRC-TAO/SEntFiN-v1.1.csv')  # 替换为您的文件路径

# # 定义一个清洗函数，用于处理 Decisions 列中的 JSON 数据
# def clean_decisions(decisions):
#     try:
#         # 替换单引号为双引号，确保合法 JSON 格式
#         cleaned = decisions.replace("'", '"')
#         # 修正键值对的双引号问题
#         cleaned = re.sub(r'([a-zA-Z0-9_\s.]+):', r'"\1":', cleaned)  # 给键补充双引号
#         cleaned = re.sub(r':\s*"([^\"]*?)"(?=\s*[,\}])', r': "\1"', cleaned)  # 修正值的双引号
#         cleaned = re.sub(r'(?<=\w)"(?=\w)', r'', cleaned)  # 移除值中间的非法双引号

#         # 将清洗后的字符串解析为 JSON
#         return json.loads(cleaned)
#     except json.JSONDecodeError as e:
#         print(f"JSON 解析错误: {decisions}\n错误信息: {e}")
#         return {}

# # 定义模糊匹配函数
# def find_similar_entity(entity, units):
#     """
#     使用模糊匹配找到与实体最相似的句子部分。
#     """
#     best_match = None
#     highest_similarity = 0.8  # 设置相似度阈值为 80%
#     for i in range(len(units)):
#         for j in range(i + 1, len(units) + 1):
#             candidate = ' '.join(units[i:j])
#             similarity = SequenceMatcher(None, entity.lower(), candidate.lower()).ratio()
#             if similarity > highest_similarity:
#                 best_match = (i, j)
#                 highest_similarity = similarity
#     return best_match

# # 定义转换函数
# def convert_row(row):
#     sentence = row['Title']
#     decisions = clean_decisions(row['Decisions'])  # 清理并解析 JSON
#     labels = []

#     # 使用正则表达式将句子拆分为单词和标点符号
#     units = re.findall(r'\w+|[^\s\w]', sentence)  # 单词和标点符号作为独立单位
#     units_lower = [unit.lower() for unit in units]  # 转换为小写

#     for entity, sentiment in decisions.items():
#         # 将实体也拆分为单词和标点符号
#         entity_units = re.findall(r'\w+|[^\s\w]', entity.lower())
#         start_idx = -1

#         # 精确匹配
#         for i in range(len(units_lower) - len(entity_units) + 1):
#             if units_lower[i:i + len(entity_units)] == entity_units:
#                 start_idx = i
#                 break

#         # 如果精确匹配失败，尝试模糊匹配
#         if start_idx == -1:
#             match = find_similar_entity(entity, units)
#             if match:
#                 start_idx, end_idx = match
#                 labels.append([(start_idx, end_idx), sentiment])
#                 print(f"模糊匹配成功: 实体 '{entity}' 映射到 '{' '.join(units[start_idx:end_idx])}'")
#             else:
#                 print(f"实体 '{entity}' 未找到于句子中，跳过。")
#         else:
#             end_idx = start_idx + len(entity_units)  # 计算结束位置
#             labels.append([(start_idx, end_idx), sentiment])

#     return {'sentence': sentence, 'labels': labels}

# # 应用转换函数到整个数据集
# converted_data = [convert_row(row) for _, row in data.iterrows()]

# # 保存为 JSON 文件
# with open('./MRC-TAO/converted_dataset.json', 'w', encoding='utf-8') as f:
#     json.dump(converted_data, f, ensure_ascii=False, indent=4)

# print("数据已成功转换并保存到 'converted_dataset.json' 文件中。")



import json

# 读取当前 JSON 文件
with open('./MRC-TAO/converted_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 定义转换为单行格式的函数
def convert_to_single_line(data):
    formatted_data = []
    for entry in data:
        sentence = entry['sentence']
        labels = entry['labels']
        single_line_entry = {
            'sentence': sentence,
            'labels': labels
        }
        formatted_data.append(single_line_entry)
    return formatted_data

# 转换数据格式
single_line_data = convert_to_single_line(data)

# 保存为新的 JSON 文件
with open('./MRC-TAO/converted_dataset_plus.json', 'w', encoding='utf-8') as f:
    for entry in single_line_data:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')  # 添加换行符以保持每行一个对象

print("已将 JSON 转换为单行格式并保存为 'converted_dataset_plus.json'")
