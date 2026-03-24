"""
src/engine/agent.py
财务风控 RAG Agent：串联 PDF 解析、向量检索与结构化 LLM 输出三大环节。
"""

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from src.engine.pdf_parser import FinancialDocumentParser
from src.engine.vector_store import FinancialVectorDB
from src.schema.risk_metrics import FinancialRiskReport

# ── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """你是一个顶级的量化风控官和精算师。请根据以下提供的财报片段（Context），提取相关财务数值并严格按照预设的格式输出。

你可以处理英文、繁体、简体文本，但请统一使用简体中文输出最终的 JSON 风险报告。

规则：
1. 绝对不能脑补，如果 Context 中找不到对应指标，数值请填 0.0，并在 source_text 中注明"未找到"，page_number 填 0。
2. source_text 必须直接引用 Context 中的原文片段，不得改写。
3. page_number 必须与所引用原文对应的页码一致。
4. 根据提取的数值综合判定风险等级（RED/YELLOW/GREEN）：
   - GREEN：流动比率 ≥ 1.5 且速动比率 ≥ 1.0 且 ROE ≥ 8% 且存货周转率 ≥ 4
   - YELLOW：未达到 GREEN 但未触发 RED
   - RED：流动比率 < 1.0，或速动比率 < 0.5，或 ROE < 0，或存货周转率 < 2

【计算授权与公式字典】
如果你在上下文中找到了原始的财务行项目数据（如流动资产合计、净利润等），但没有直接找到比率数值，你**必须**使用以下标准财务公式进行推算，绝不能直接填 0：

- 流动比率 (Current Ratio)   = 流动资产合计 / 流动负债合计
- 速动比率 (Quick Ratio)     = (流动资产合计 - 存货账面价值) / 流动负债合计
- ROE (净资产收益率)          = 净利润 / 归属于母公司所有者权益
- 净利率 (Net Profit Margin) = 净利润 (或 年内溢利) / 营业收入 (或 收入)
- 存货周转率 (Inv. Turnover) = 营业成本 / 存货账面价值

【防幻觉补丁】
在执行推算时，请确保你使用的原始数字确切来自于检索到的上下文。如果上下文缺失计算所需的某一项关键原始数据，该指标才允许输出 0.0。
"""

_USER_PROMPT_TEMPLATE = """请根据以下财报片段，提取财务风控指标并输出结构化报告。

<context>
{context}
</context>

请严格按照 JSON Schema 输出完整的财务风险报告。"""

# 检索查询：语义问题 + 原始财务术语，覆盖流动性、盈利能力、营运健康度三个维度
_RETRIEVAL_QUERIES = [
    # 语义层
    "企业的流动比率、速动比率是多少？",
    "企业的ROE和净利率情况如何？",
    "存货周转率是多少？",
    # 原始财务术语层——直接匹配报表行项目
    "流动资产合计 流动负债合计",
    "归属于母公司股东的净利润",
    "存货账面价值 营业成本",
    "资产负债表 合并",
    "营业收入 净利润 净利率",
]


# ── Agent ─────────────────────────────────────────────────────────────────────

class FinancialRiskAgent:
    """
    财务风控 RAG Agent。

    执行流：
        PDF → Parser → VectorDB → 三维度检索 → ChatAnthropic(structured_output) → FinancialRiskReport
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        retrieval_k: int = 10,
    ) -> None:
        """
        Args:
            persist_dir:  向量库持久化目录，None 时使用 FinancialVectorDB 默认路径。
            retrieval_k:  每条查询检索的 top-k 数量，合并后去重作为最终 context。
        """
        self._parser = FinancialDocumentParser()

        vector_db_kwargs = {}
        if persist_dir is not None:
            vector_db_kwargs["persist_dir"] = persist_dir
        self._vector_db = FinancialVectorDB(**vector_db_kwargs)

        self._retrieval_k = retrieval_k

        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0,      # 金融风控场景，确定性优先
            max_tokens=4096,
        )
        # 强制 LLM 按 FinancialRiskReport Schema 输出，消除格式幻觉
        self._structured_llm = llm.with_structured_output(FinancialRiskReport, method="json_mode")

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def analyze_report(
        self,
        pdf_path: str | Path,
        override_start: int | None = None,
        override_end: int | None = None,
    ) -> FinancialRiskReport:
        """
        端到端分析一份财务 PDF，返回结构化风控报告。

        Args:
            pdf_path:       财务 PDF 文件路径。
            override_start: 手动指定起始页（跳过 TOC 路由）。
            override_end:   手动指定结束页（跳过 TOC 路由）。

        Returns:
            FinancialRiskReport: 经 Pydantic 校验的结构化风控指标报告。
        """
        # 0. 路由：优先书签秒级定位，失败再回退 OCR TOC 路由
        if override_start is None or override_end is None:
            print(f"[0/3] Metadata 书签路由：尝试通过 PDF Outline 秒级定位财务报表...")
            bookmark_page = self._parser.find_pages_via_bookmarks(str(pdf_path))
            if bookmark_page is not None:
                print(f"      [Metadata Router] 成功通过书签秒级定位到财务主表页码: {bookmark_page}")
                override_start = bookmark_page
                override_end = bookmark_page + 10
                print(f"      [Metadata Router] 解析窗口：第 {override_start}–{override_end} 页")
            else:
                print(f"      [Metadata Router] 书签未命中，回退至 Agentic TOC 路由...")
                start_page, _ = self._parser.find_target_pages_via_toc(str(pdf_path))
                override_start = start_page - 2
                override_end = start_page + 8
                print(f"      [TOC Router] 自动设定解析窗口：第 {override_start}–{override_end} 页")

        # a. 解析 PDF → Document chunks，存入向量库
        print(f"[1/3] 解析 PDF 并写入向量数据库：{pdf_path}")
        chunks = self._parser.parse(
            pdf_path,
            override_start=override_start,
            override_end=override_end,
        )
        print(f"      共生成 {len(chunks)} 个文本块")
        self._vector_db.add_documents(chunks)

        # b. 分三个金融问题检索，合并所有 Document 内容与页码为长文本 context
        print("[2/3] 向量检索（流动性 / 盈利能力 / 营运健康度）...")
        context_docs = self._retrieve_context()
        context_text = self._format_context(context_docs)
        print(f"      合并后共 {len(context_docs)} 条上下文片段")

        # c & d. 构造 Prompt，调用结构化 LLM
        print("[3/3] 调用 LLM 提取结构化风控指标...")
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_USER_PROMPT_TEMPLATE.format(context=context_text)),
        ]
        report: FinancialRiskReport = self._structured_llm.invoke(messages)

        print(f"      风险等级：{report.overall_risk_level.value}")
        print(f"      评估总结：{report.assessment_summary}")
        return report

    def run(
        self,
        pdf_path: str | Path,
        override_start: int | None = None,
        override_end: int | None = None,
    ) -> dict:
        """
        UI 调用入口：执行完整分析流程，返回可序列化的 dict。

        每次调用使用独立的 Chroma collection，避免多份财报向量相互污染。

        Args:
            pdf_path:       财务 PDF 文件路径。
            override_start: 专家干预起始页（1-based）；传入时跳过 Strided Search。
            override_end:   专家干预结束页（1-based）；传入时跳过 Strided Search。

        Returns:
            dict: FinancialRiskReport 的可 JSON 序列化字典。
        """
        import uuid
        tmp_db = FinancialVectorDB(collection_name=f"run_{uuid.uuid4().hex[:8]}")

        # 0. 路由：优先书签秒级定位，失败再回退 OCR TOC 路由
        if override_start is None or override_end is None:
            print(f"[0/3] Metadata 书签路由：尝试通过 PDF Outline 秒级定位财务报表...")
            bookmark_page = self._parser.find_pages_via_bookmarks(str(pdf_path))
            if bookmark_page is not None:
                print(f"      [Metadata Router] 成功通过书签秒级定位到财务主表页码: {bookmark_page}")
                override_start = bookmark_page
                override_end = bookmark_page + 10
                print(f"      [Metadata Router] 解析窗口：第 {override_start}–{override_end} 页")
            else:
                print(f"      [Metadata Router] 书签未命中，回退至 Agentic TOC 路由...")
                start_page, _ = self._parser.find_target_pages_via_toc(str(pdf_path))
                override_start = start_page - 2
                override_end = start_page + 8
                print(f"      [TOC Router] 自动设定解析窗口：第 {override_start}–{override_end} 页")

        print(f"[1/3] 解析 PDF 并写入向量数据库：{pdf_path}")
        chunks = self._parser.parse(
            pdf_path,
            override_start=override_start,
            override_end=override_end,
        )
        print(f"      共生成 {len(chunks)} 个文本块")
        tmp_db.add_documents(chunks)

        print("[2/3] 向量检索（流动性 / 盈利能力 / 营运健康度）...")
        context_docs = self._retrieve_context(vector_db=tmp_db)
        context_text = self._format_context(context_docs)
        print(f"      合并后共 {len(context_docs)} 条上下文片段")

        print("[3/3] 调用 LLM 提取结构化风控指标...")
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=_USER_PROMPT_TEMPLATE.format(context=context_text)),
        ]
        report: FinancialRiskReport = self._structured_llm.invoke(messages)
        print(f"      风险等级：{report.overall_risk_level.value}")
        print(f"      评估总结：{report.assessment_summary}")

        return report.model_dump(mode="json")

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _retrieve_context(self, vector_db=None) -> list[Document]:
        """对三个金融问题分别检索，按页码升序去重后返回。"""
        db = vector_db if vector_db is not None else self._vector_db
        seen: set[str] = set()
        results: list[Document] = []

        for query in _RETRIEVAL_QUERIES:
            for doc in db.similarity_search(query, k=self._retrieval_k):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    results.append(doc)

        results.sort(key=lambda d: d.metadata.get("page", 0))
        return results

    @staticmethod
    def _format_context(docs: list[Document]) -> str:
        """将 Document 列表格式化为带页码标注的连续上下文字符串。"""
        segments = [
            f"[第 {doc.metadata.get('page', '?')} 页]\n{doc.page_content}"
            for doc in docs
        ]
        return "\n\n---\n\n".join(segments)


# ── 顶层 API 入口 ─────────────────────────────────────────────────────────────

def run_agent(
    pdf_path: str,
    override_start: int = None,
    override_end: int = None,
) -> dict:
    """
    前端调用入口：每次运行前清空 ChromaDB，防止脏数据干扰，
    然后完整执行 PDF 解析 → 向量检索 → LLM 提取流程。

    Args:
        pdf_path:       财务 PDF 文件路径。
        override_start: 专家干预起始页（1-based）；传入时跳过 Strided Search。
        override_end:   专家干预结束页（1-based）；传入时跳过 Strided Search。

    Returns:
        dict: FinancialRiskReport 的可 JSON 序列化字典。
    """
    import shutil

    chroma_dir = Path("data/processed/chroma_db")
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        print(f"[准备] 已清空旧 ChromaDB：{chroma_dir}")

    agent = FinancialRiskAgent()
    result = agent.run(
        pdf_path,
        override_start=override_start,
        override_end=override_end,
    )
    return result


# ── 手动测试入口 ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_agent("data/raw/report.pdf")
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
