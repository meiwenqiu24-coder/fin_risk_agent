"""
src/engine/vector_store.py
财务向量数据库：使用本地 HuggingFace Embedding（无外部 API 调用）+ Chroma 持久化存储。
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 默认持久化目录
_DEFAULT_PERSIST_DIR = Path(__file__).parents[2] / "data" / "processed" / "chroma_db"

# 多语言模型：支持中文（简/繁）、英文语义对齐，约 570MB
_DEFAULT_EMBED_MODEL = "BAAI/bge-m3"


class FinancialVectorDB:
    """
    基于 Chroma + 本地 Sentence-Transformers 的财务向量数据库。

    设计原则：
    - 纯本地 Embedding，金融数据不出内网
    - 数据持久化到 data/processed/chroma_db，重启后无需重新 ingest
    - collection_name 支持多份财报隔离存储

    Usage::

        db = FinancialVectorDB()
        db.add_documents(chunks)          # chunks 来自 FinancialDocumentParser.parse()
        results = db.similarity_search("流动比率是多少", k=3)
    """

    def __init__(
        self,
        persist_dir: str | Path = _DEFAULT_PERSIST_DIR,
        embed_model: str = _DEFAULT_EMBED_MODEL,
        collection_name: str = "financial_reports",
    ) -> None:
        """
        Args:
            persist_dir:      Chroma 数据库持久化目录，不存在时自动创建。
            embed_model:      HuggingFace 模型名称或本地路径，默认使用支持中文的轻量模型。
            collection_name:  Chroma collection 名称，不同财报可用不同名称隔离。
        """
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        self._embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            # 推理时使用 CPU 即可；有 GPU 时 device 改为 "cuda"
            model_kwargs={"device": "cpu"},
            # 批量 encode 时标准化向量，提升余弦相似度检索精度
            encode_kwargs={"normalize_embeddings": True},
        )

        self._db = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(self._persist_dir),
        )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """
        将 FinancialDocumentParser 生成的 Document chunk 列表写入向量库。

        Args:
            documents: 带 metadata（至少含 page、source）的 Document 列表。

        Raises:
            ValueError: 传入空列表时抛出。
        """
        if not documents:
            raise ValueError("documents 列表不能为空")

        self._db.add_documents(documents)
        self._db.persist()

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        """
        根据风控问题检索最相关的 top-k 文本块。

        Args:
            query: 自然语言风控问题，例如"公司的流动比率是多少"。
            k:     返回的最大结果数，默认 3。

        Returns:
            list[Document]: 按相似度降序排列，每条 metadata 含 page、source 等溯源信息。

        Raises:
            ValueError: query 为空字符串时抛出。
        """
        if not query.strip():
            raise ValueError("query 不能为空字符串")

        return self._db.similarity_search(query, k=k)
