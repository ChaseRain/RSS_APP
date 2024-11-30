# 资讯问答机器人 (Langchain + Ollama + RSSHub 实现 RAG)

## 一、项目背景

在信息爆炸的时代，人工筛选有价值的信息变得愈加困难。为了帮助快速提取和整合关键信息，我们开发了一个本地部署的资讯问答机器人，能够基于 A 股行情等多种领域的资讯，提供智能化的问答服务。

通过结合 Langchain、Ollama 和 RSSHub，我们创建了一个可以从海量市场数据中提取相关信息并进行分析的系统。该系统能够在本地运行，并通过 RSSHub 提供的 RSS 源来获取最新的市场资讯。

## 二、技术栈

- **Langchain**: 用于构建基于大语言模型 (LLM) 的应用程序，提供数据检索、链式处理和生成任务等功能。
- **Ollama**: 本地运行的多种大型语言模型框架，支持如 Qwen、Gemma、Mistral 等多种模型，能够进行高效的文本生成和推理任务。
- **RSSHub**: 一个开源项目，能够为各大网站生成 RSS 源，轻松获取资讯数据。


## 三、项目结构

```
|-- src/
|   |-- __init__.py
|   |-- content_fetcher.py      # 获取 RSS 内容
|   |-- vectorizer.py           # 将文档转为向量
|   |-- rag_chain.py            # 实现 RAG 功能
|   |-- app.py                  # 启动 Gradio UI
|
|-- .gitignore                  # Git 忽略文件
|-- README.md                   # 项目说明文件
|-- requirements.txt            # 项目依赖
```

## 四、环境配置

### 1. 创建 Python 虚拟环境

为了确保依赖的兼容性，建议创建一个 Python 虚拟环境：

```bash
conda create -n finance_bot python=3.10
conda activate finance_bot
```

### 2. 安装依赖库

在虚拟环境中安装以下依赖：

```bash
pip install ollama langchain faiss-cpu gradio feedparser sentence-transformers lxml
```

- **ollama**: 本地运行的语言模型。
- **langchain**: 用于构建和管理链式数据处理。
- **faiss-cpu**: 向量检索库，用于存储和检索文档向量。
- **gradio**: 创建用户界面的工具。
- **feedparser**: 用于解析 RSS 源。
- **sentence-transformers**: 用于生成文本的向量表示。
- **lxml**: 用于处理 XML 和 HTML 数据。

### 3. 启动 Ollama 服务

确保 Ollama 服务已经启动。在 macOS 上可以直接启动 Ollama 应用程序；在 Linux 上，使用以下命令启动服务：

```bash
ollama serve
```

### 4. 下载模型

通过 Ollama 下载您所需的模型。例如，下载 Qwen 模型：

```bash
ollama pull qwen:7b
```

## 五、使用方法

### 1. 获取数据

本项目通过 RSSHub 获取财经新闻和 A 股行情数据。您可以自定义 RSS 源 URL 来获取更多的数据。数据会被解析并通过 `RecursiveCharacterTextSplitter` 拆分成较小的文档块，方便后续处理。

```python
import feedparser
from lxml import etree

def get_content(url):
    data = feedparser.parse(url)
    docs = []
    for news in data['entries']:
        summary = etree.HTML(text=news['summary']).xpath('string(.)')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        split_docs = text_splitter.create_documents(texts=[summary], metadatas=[{k: news[k] for k in ('title', 'published', 'link')}])
        docs.extend(split_docs)
                
    return data, docs
```

### 2. 向量化文档

使用 HuggingFace 的模型将文档转化为向量，存储在 FAISS 向量数据库中，方便后续检索。

```python
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def create_docs_vector(docs):
    embeddings = HuggingFaceEmbeddings(model_name="/path/to/bge-m3", encode_kwargs={'normalize_embeddings': True})
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store
```

### 3. 实现 RAG（检索增强生成）

RAG 通过从向量数据库中检索相关文档并生成答案，结合上下文信息来提升回答的质量。

```python
def rag_chain(question, vector_store, model='qwen', threshold=0.3):
    related_docs = vector_store.similarity_search_with_relevance_scores(question)
    related_docs = list(filter(lambda x: x[1] > threshold, related_docs))
    context = "\n\n".join([f'[citation:{i}] {doc[0].page_content}' for i, doc in enumerate(related_docs)])
    metadata = {str(i): doc[0].metadata for i, doc in enumerate(related_docs)}

    system_prompt = f"""
    当你收到用户的问题时，请编写清晰、简洁、准确的回答。
    你会收到一组与问题相关的上下文，每个上下文都以参考编号开始，如[citation:x]，其中x是一个数字。
    请使用这些上下文，并在适当的情况下在每个句子的末尾引用上下文。
    """

    user_prompt = f"用户的问题是：{question}"

    response = ollama.chat(model=model, messages=[
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': user_prompt
        }
    ])

    return response['message']['content'], context
```

### 4. 创建 Gradio UI

通过 Gradio 创建一个简单的 Web 界面，用户可以输入问题并获得模型的回答。

```python
import gradio as gr

def launch_ui():
    interface = gr.Interface(
        fn=lambda question, model, threshold: rag_chain(question, vector_store, model, threshold),
        inputs=[
            gr.Textbox(lines=2, placeholder="请输入你的问题...", label="问题"),
            gr.Dropdown(['gemma', 'mistral', 'mixtral', 'qwen:7b'], label="选择模型", value='gemma'),
            gr.Number(label="检索阈值", value=0.3)
        ],
        outputs=[
            gr.Text(label="回答"),
            gr.Text(label="相关上下文")
        ],
        title="资讯问答Bot",
        description="输入问题，我会查找相关资料，然后整合并给你生成回复"
    )

    interface.launch()

if __name__ == "__main__":
    launch_ui()
```

## 六、测试与评估

您可以测试不同的模型（如 Qwen、Gemma、Mistral、Mixtral）来评估它们在回答问题时的表现。不同模型的性能可能有所差异，具体取决于您的需求。

## 七、总结

本项目展示了如何使用 Langchain 和 Ollama 在本地搭建一个资讯问答机器人。通过结合 RSSHub 提供的动态资讯源，系统能够根据用户的问题，从海量数据中检索相关内容并生成高质量的答案。

### 项目特点：

- 本地部署，无需依赖外部服务，保障隐私。
- 可根据需要定制更多 RSS 源。
- 支持多个大型语言模型（Qwen、Gemma、Mistral、Mixtral 等）。

您可以根据自己的需求进行扩展，进一步优化模型效果或添加新的数据源。

---

希望这个 README 能够帮助您顺利完成项目部署。如果有任何问题，随时可以向我询问！