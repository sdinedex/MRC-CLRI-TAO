import json

# 读取原始 JSON 数据集
with open('./MRC-TAO/FinEntity.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 定义转换函数
def convert_json_format(data):
    result = []
    for item in data:
        # 提取句子
        sentence = item['content']
        labels = []

        # 提取标注内容
        for annotation in item['annotations']:
            start = annotation['start']
            end = annotation['end']
            sentiment = annotation['tag']  # 或者 'label'，根据实际字段名称
            labels.append([(start, end), sentiment])

        # 整理为目标格式
        result.append({
            'sentence': sentence,
            'labels': labels
        })

    return result

# 转换数据格式
converted_data = convert_json_format(data)

# 保存为新的 JSON 文件
with open('./MRC-TAO/converted_dataset_finentity_plus.json', 'w', encoding='utf-8') as f:
    for item in converted_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')  # 每行一个 JSON 对象

print("转换完成！数据已保存到 'converted_dataset.json' 文件中。")
