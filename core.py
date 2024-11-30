import feedparser
import ollama


from lxml import etree
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 从订阅源获取内容
def get_content_from_feed(feed_url="https://www.zhihu.com/rss"):
    # 使用 feedparser 解析订阅源 ,通常用于读取rss feed
    data = feedparser.parse(feed_url)

    docs = []
    for news in data["entries"]:
        # 通过xpath 提取干净的文本内容
        summary = etree.HTML(news["summary"]).xpath("string(.)")

        # 初始化文档拆分器，设定大小和重叠大小
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=10, length_function=len)

        # 拆分文档
        split_docs = text_splitter.create_documents(
            texts=[summary],
            metadatas=[{k: news[k] for k in ["title", "published", "link"]}]
        )[:10]

        # 合并拆分后的文档
        docs.extend(split_docs)

    return data, docs


def create_docs_vector(docs):
    # 手动加载模型并设置为 CPU
    model_name = "~/huggingface/bge-m3"

    # 基于 embeddings，为 docs 创建向量
    embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={'normalize_embeddings': True})
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store




def rag_chain(question, vector_store, model='qwen', threshold=0.3):
    # 确保 vector_store 是正确的向量数据库对象
    if not hasattr(vector_store, 'similarity_search_with_relevance_scores'):
        raise ValueError("vector_store 必须是一个有效的向量数据库对象，比如 FAISS 或 Chroma 实例")
    
    # 从向量数据库中检索与 question 相关的文档
    related_docs = vector_store.similarity_search_with_relevance_scores(question)

    # 过滤掉小于设定阈值的文档
    related_docs = list(filter(lambda x: x[1] > threshold, related_docs))

    # 格式化检索到的文档
    context = "\n\n".join([f'[citation:{i}] {doc[0].page_content}' for i, doc in enumerate(related_docs)])

    # 保存文档的 meta 信息，如 title、link 等
    # metadata = {str(i): doc[0].metadata for i, doc in enumerate(related_docs)}

    # 设定系统提示词
    system_prompt = f"""
    当你收到用户的问题时，请编写清晰、简洁、准确的回答。
    你会收到一组与问题相关的上下文，每个上下文都以参考编号开始，如[citation:x]，其中x是一个数字。
    请使用这些上下文，并在适当的情况下在每个句子的末尾引用上下文。

    你的答案必须是正确的，并且使用公正和专业的语气写作。请限制在1024个tokens之内。
    不要提供与问题无关的信息，也不要重复。
    不允许在答案中添加编造成分，如果给定的上下文没有提供足够的信息，就说“缺乏关于xx的信息”。

    请用参考编号引用上下文，格式为[citation:x]。
    如果一个句子来自多个上下文，请列出所有适用的引用，如[citation:3][citation:5]。
    除了代码和特定的名字和引用，你的答案必须用与问题相同的语言编写，如果问题是中文，则回答也是中文。

    这是一组上下文：

    {context}

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

    print(system_prompt + user_prompt)

    return response['message']['content'], context



