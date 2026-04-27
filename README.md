# ⚡ 电力检修智能问答助手

基于 RAG（检索增强生成）架构，结合智谱 GLM-4-Flash 大模型，针对电力检修领域的智能问答系统。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的智谱 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：
```
ZHIPUAI_API_KEY=你的API Key
```

> 去 [bigmodel.cn](https://bigmodel.cn) 注册并获取免费 API Key

### 3. 放入知识文档

将电力检修相关的 PDF、Word（.docx）或 TXT 文件放入 `data/` 目录。

目前已有一份示例文档 `data/sample_transformer.txt`，可直接测试。

### 4. 构建知识库

```bash
python ingest.py
```

### 5. 启动问答界面

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`

## 项目结构

```
power-qa-bot/
├── data/                  # 放入电力检修文档（PDF/Word/TXT）
├── chroma_db/             # 向量数据库（自动生成）
├── ingest.py              # 文档解析 & 向量入库
├── rag_chain.py           # RAG检索 + 问答链
├── app.py                 # Streamlit 前端
├── requirements.txt
├── .env.example
└── README.md
```

## 示例问题

- 变压器过热如何处理？
- 断路器检修周期是多久？
- 电缆单相接地故障怎么定位？
- 测量绝缘电阻时吸收比的判断标准是什么？
