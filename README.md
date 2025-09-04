# Literature RAG Knowledge Graph

This is a coursework project that demonstrates how **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLM)** can be used to build a **knowledge graph** from long literary works.  
The system supports keyword-based retrieval, entity & relation extraction, and CSV export for visualization (e.g., in **Neo4j Aura**).

---

## ✨ Features
- **Text preprocessing**: Split the novel into overlapping chunks and generate embeddings  
- **Semantic retrieval**: Use FAISS and Sentence-Transformers for fast, relevant passage search  
- **Entity & relation extraction**: Call **DeepSeek API** to identify characters, locations, events, and relationships  
- **Structured output**: Save extracted knowledge into CSV files for further analysis and graph visualization  

---

## 📖 Example Workflow
1. Place the target literary text (e.g. *红楼梦原文.txt*) in the project folder.  
2. Run the script.  
3. Enter a keyword (e.g. *“晴雯撕扇”*).  
4. The system retrieves the most relevant passage.  
5. DeepSeek API extracts entities and relations.  
6. CSV files are generated automatically.  

Output files include:  
- `人物表.csv` (Characters)  
- `地点表.csv` (Locations)  
- `人物关系表.csv` (Character relationships)  
- `地点交互表.csv` (Character-location interactions)  
- `事件表.csv` (Events)  

These can be directly imported into **Neo4j Aura** for visualization.

---

## 🚀 How to Run

### Prerequisites
- Python 3.9+  
- DeepSeek API key  
- Dependencies:  
  - `faiss-cpu`  
  - `sentence-transformers`  
  - `openai`  
  - `numpy`  
  - `pickle` (standard library)  
  - `csv` (standard library)  

### Installation
```bash
git clone https://github.com/Arashi250/literature-rag-kg.git
cd literature-rag-kg
```

### Run
```bash
python main.py
```

## 📜 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this code with proper attribution.

## 🙏 Acknowledgements

- DeepSeek API: for entity & relation extraction
- Neo4j Aura: for graph visualization
- Sentence-Transformers and FAISS: for vector search
