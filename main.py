import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import json
import  csv
from openai import OpenAI

# 配置
MODEL_NAME = 'BAAI/bge-small-zh'   # 更强的中文 embedding
TXT_PATH = '红楼梦原文.txt'         # 你的文学作品文本
INDEX_PATH = 'faiss.index'
CHUNKS_PATH = 'chunks.pkl'
EMBS_PATH   = 'embeddings.npy'

# 一次性离线预处理
def preprocess_and_save():
    # 分片
    with open(TXT_PATH, 'r', encoding='utf-8') as f:
        raw = f.read().replace('\n', '')
    # 按句号、问号、叹号分句并做滑窗重叠
    import re
    sents = re.split('(?<=[。！？])', raw)
    chunks, window, step = [], 10, 2

    for i in range(0, len(sents), step):
        chunk = ''.join(sents[i:i+window]).strip()
        if chunk:
            chunks.append(chunk)
    # 生成 embeddings
    embedder = SentenceTransformer(MODEL_NAME)
    embs = embedder.encode(chunks, convert_to_numpy=True)
    # 建索引
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    # 持久化
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(chunks, f)
    import numpy as np
    np.save(EMBS_PATH, embs)
    print("✅ 离线索引和分片加载完成")

# 启动时加载
def load_index_and_chunks():
    import pickle, numpy as np
    import faiss
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        chunks = pickle.load(f)
    embs = np.load(EMBS_PATH)
    embedder = SentenceTransformer(MODEL_NAME)
    return index, chunks, embedder

# 如果不存在，就先跑离线
if not os.path.exists(INDEX_PATH):
    preprocess_and_save()

# 每次启动只需这行，秒级加载
index, chunks, embedder = load_index_and_chunks()

# 统计 token，大致用字数/2 估算
def filter_too_long(chunks, max_chars=3000):
    return [c for c in chunks if len(c) <= max_chars]
chunks = filter_too_long(chunks)

def search(query, k=3, threshold=None):
    qv = embedder.encode([query], convert_to_numpy=True)
    dists, idxs = index.search(qv, k)
    results = []
    for dist, i in zip(dists[0], idxs[0]):
        if threshold is None or dist < threshold:
            results.append(chunks[i])
    return results


def extract_kg_from_text(text):
    client = OpenAI(api_key="your api_key", base_url="https://api.deepseek.com")

    system_prompt = """你是一个知识图谱专家，请从以下文学文本中抽取实体和关系，仔细一点，不要遗漏，严格按以下JSON格式输出：
    {
      "人物表": [
        {"person_id": "唯一数字ID", "姓名": "标准姓名（如‘宝玉’需转为‘贾宝玉’）"}
      ],
      "人物关系表": [
        {"source_id": "来源人物ID", "target_id": "目标人物ID", "关系": "亲属（需要具体的关系）/朋友/敌对等"}
      ],
      "地点表": [
        {"location_id": "唯一数字ID", "地点名称": "标准名称"}
      ],
      "地点交互表": [
        {"person_id": "人物ID", "location_id": "地点ID", "交互类型": "居住/到访/途经等"}
      ],
      "事件表": [
        {"person_id": "人物ID", "事件名称": "事件描述"}
      ]
    }

    规则：
    1. 所有ID必须为自增数字（从1开始），同一实体在不同表中使用相同ID
    2. 交互类型仅限：居住、到访、途经、其他
    3. 事件名称需简洁，不超过8个字
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": f"文本：{text}\n请提取人物、地点、事件实体及其关系，严格按照要求格式输出，使用中文关系类型"},
        ],
        stream=False
    )

    response_data = response.choices[0].message.content
    l = len(response_data)
    response_data = response_data[7:l - 3]
    data = json.loads(response_data)

    return data


if __name__ == "__main__":
    # 加载索引和分片
    index, chunks, embedder = load_index_and_chunks()

    # 用户查询
    query = input("请输入情节关键词：")

    # 检索（只要最相关1段）
    matched = search(query, k=1)
    if not matched:
        print("❌ 未检索到足够相关内容，请换个关键词。")
        exit(1)
    text = matched[0]
    print(text)

    # 知识图谱抽取
    kg = extract_kg_from_text(text)
    if kg:
        # 生成 CSV
        for tab in ["人物表","地点表","人物关系表","地点交互表","事件表"]:
            with open(f"{tab}.csv","w",encoding="utf-8",newline="") as f:
                writer = csv.DictWriter(f, fieldnames=kg[tab][0].keys())
                writer.writeheader()
                writer.writerows(kg[tab])
        print("✅ 完成，输出 CSV 在当前目录")

