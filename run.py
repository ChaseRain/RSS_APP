import gradio as gr
from core import create_docs_vector, get_content_from_feed, rag_chain

if __name__ == "__main__":

    # 财联社 RSS
    url = "https://www.zhihu.com/rss"
    data, docs = get_content_from_feed(url)

    # 创建文档向量
    # 使用 create_docs_vector 函数创建向量存储
    # docs 是从 RSS 获取的文章内容列表
    vector_store = create_docs_vector(docs)
    
    # 打印处理的文章数量
    # len(docs) 获取文章列表的长度
    print(f"已成功处理 {len(docs)} 篇文章")
    
    # 打印向量存储中的向量数量
    # vector_store.index.ntotal 获取 FAISS 索引中的向量总数
    print(f"向量存储中包含 {vector_store.index.ntotal} 个向量")

    # 创建 Gradio 界面
    interface = gr.Interface(
        fn=lambda question, model, threshold: rag_chain(question, vector_store, model, threshold),
        inputs=[
            gr.Textbox(lines=2, placeholder="请输入你的问题...", label="问题"),
            gr.Dropdown(['gemma2', 'mistral', 'mixtral', 'qwen2'], label="选择模型", value='gemma'),
            gr.Number(label="检索阈值", value=0.3)
        ],
        outputs=[
            gr.Text(label="回答"),
            gr.Text(label="相关上下文")
        ],
        title="资讯问答Bot",
        description="输入问题，我会查找相关资料，然后整合并给你生成回复"
    )

    # 运行界面
    interface.launch()
