"""
src/engine/pdf_parser.py
财务 PDF 文档解析器：逐页提取文本并按语义分块，同时将页码绑定到每个 chunk 的 metadata，
以便与 FinancialRiskReport Schema 中的 page_number 字段精确对接。

提取策略——跳跃式自适应寻星 OCR 算法 (Strided Search & Local Expansion)：
  阶段 1：大步幅跳跃抽样
    从倒数第 stride 页开始，以 stride 为步长向前抽样，对每个候选页做 OCR，
    检测是否命中财务锚点词（审计报告、资产负债表、Balance Sheet 等）。
    命中即记录 anchor_page 并停止跳跃；未命中则用默认锚点。
  阶段 2：局部全量展开
    以 anchor_page 为中心，向前 2 页、向后 18 页定义黄金窗口（共 20 页），
    对窗口内所有页无条件 OCR，输出 chunk 强制标记为财务报表页。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# ── 财务报表关键词 ────────────────────────────────────────────────────────────

_FINANCIAL_TABLE_KEYWORDS = [
    # 简体
    "资产负债表", "利润表", "现金流量表", "合并资产", "合并利润",
    "流动资产", "流动负债", "非流动资产", "非流动负债",
    "营业收入", "营业成本", "净利润", "归属于母公司",
    "存货", "应收账款", "货币资金", "所有者权益",
    "综合资产负债表", "审计报告",
    # 繁体
    "資產負債表", "損益表", "現金流量表", "流動資產", "流動負債",
    "營業收入", "營業成本", "淨利潤", "存貨", "應收帳款",
    "獨立核數師報告", "綜合資產負債表",
    # 英文
    "Balance Sheet", "Income Statement", "Cash Flow Statement",
    "Current Assets", "Current Liabilities", "Non-current Assets",
    "Revenue", "Cost of Revenue", "Net Income", "Gross Profit",
    "Inventories", "Accounts Receivable", "Stockholders Equity",
    "Consolidated Balance Sheet",
]

# ── 跳跃式寻星锚点词库 ───────────────────────────────────────────────────────
# 用于 Strided Search 阶段的 OCR 文本命中检测（大小写不敏感）

_ANCHOR_KEYWORDS = [
    "审计报告", "核数师报告", "核數師報告", "Auditor's Report",
    "auditor's report", "auditor report",
    "资产负债表", "資產負債表", "Balance Sheet",
    "balance sheet",
    "综合损益", "綜合損益",
    "利润表", "利潤表", "損益表",
    "损益表", "Income Statement",
    "income statement",
]

# CID 占位符正则，如 (cid:123)
_CID_PATTERN = re.compile(r"\(cid:\d+\)")


# ── 乱码检测 ─────────────────────────────────────────────────────────────────

def _chinese_ratio(text: str) -> float:
    """返回文本中 CJK 字符占总字符数的比例（忽略空白符）。"""
    stripped = text.replace(" ", "").replace("\n", "")
    if not stripped:
        return 0.0
    cjk = sum(1 for ch in stripped if "\u4e00" <= ch <= "\u9fff")
    return cjk / len(stripped)


def _is_garbled(text: str, min_chinese_ratio: float = 0.05) -> bool:
    """
    判断文本是否为乱码：
      - 含有 CID 占位符，且数量超过 5 个
      - 或：文本长度超过 20 且中文字符比例低于阈值
    """
    cid_count = len(_CID_PATTERN.findall(text))
    if cid_count > 5:
        return True
    if len(text) > 20 and _chinese_ratio(text) < min_chinese_ratio:
        return True
    return False


# ── OCR 降级（延迟导入，避免无 OCR 场景的启动开销）───────────────────────────

def _ocr_page(pdf_path: Path, page_index: int, dpi: int = 200) -> str:
    """
    将 PDF 指定页渲染为图片并用 easyocr 识别文字，返回识别文本。

    Args:
        pdf_path:   PDF 文件路径。
        page_index: 0-based 页码。
        dpi:        渲染分辨率，越高越准但越慢，200 是速度与精度的平衡点。
    """
    from pdf2image import convert_from_path  # type: ignore
    import easyocr  # type: ignore
    import numpy as np

    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page_index + 1,
        last_page=page_index + 1,
    )
    if not images:
        return ""

    img_array = np.array(images[0])

    # 复用进程内的 Reader 实例（首次初始化会下载模型，约 1-2 秒）
    if not hasattr(_ocr_page, "_reader"):
        # ch_tra 模型同时覆盖繁体与简体（字符集高度重叠），不可与 ch_sim 混用
        _ocr_page._reader = easyocr.Reader(["ch_tra", "en"], gpu=False, verbose=False)

    results = _ocr_page._reader.readtext(img_array, detail=0, paragraph=True)
    return "\n".join(results)


# ── Parser ────────────────────────────────────────────────────────────────────

class FinancialDocumentParser:
    """
    读取财务 PDF 报告，逐页提取文本后使用 RecursiveCharacterTextSplitter 分块。

    提取优先级：pypdf 文本层 → （乱码时）easyocr OCR 降级。

    每个输出的 Document chunk 均携带以下 metadata：
        - page               (int)  : 原始页码（1-based）
        - source             (str)  : PDF 文件路径
        - page_char_count    (int)  : 提取后的字符数
        - is_financial_table (bool) : 是否包含财务报表关键词
        - ocr_fallback       (bool) : 是否经过 OCR 降级处理
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        financial_table_chunk_size: int = 3000,
        ocr_dpi: int = 200,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        self._financial_splitter = RecursiveCharacterTextSplitter(
            chunk_size=financial_table_chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        self._ocr_dpi = ocr_dpi

    @staticmethod
    def _is_financial_table_page(text: str) -> bool:
        return any(kw in text for kw in _FINANCIAL_TABLE_KEYWORDS)

    def parse(
        self,
        pdf_path: str,
        override_start: int = None,
        override_end: int = None,
    ) -> list[Document]:
        """
        解析指定 PDF，返回带 metadata 的 Document chunk 列表。

        若传入 override_start / override_end，则跳过自动寻星，直接对指定页码区间
        全量 OCR 并返回 chunk（短路模式）。

        否则对超过 100 页的长文档执行两阶段定位：
          阶段 1（Strided Search）：从倒数第 20 页起以 20 页为步长向前跳跃，
            逐页 OCR 并检测锚点词，命中即停止。
          阶段 2（Local Expansion）：以 anchor_page 为中心展开黄金窗口，
            对窗口内所有页无条件 OCR，输出 chunk 强制标记为财务报表页。

        Args:
            pdf_path:       PDF 文件路径。
            override_start: 专家干预起始页（1-based，含）。
            override_end:   专家干预结束页（1-based，含）。
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        if total_pages == 0:
            raise ValueError(f"PDF 文件不含任何页面：{pdf_path}")

        # ── 短路逻辑：专家干预模式 ────────────────────────────────────────
        if override_start is not None and override_end is not None:
            print(
                f"      [专家干预] 已跳过自动寻星，"
                f"强制 OCR 第 {override_start}–{override_end} 页"
            )
            pages_to_ocr = list(range(override_start - 1, override_end))
            return self._ocr_page_list(pdf_path, pages_to_ocr)

        # ── 阶段 1：大步幅跳跃抽样 (Strided Search) ──────────────────────
        strided_threshold = 100
        stride = 20
        default_anchor = 190
        expansion_before = 5
        expansion_after = 25

        start_page: int | None = None
        end_page: int | None = None
        force_ocr_all = False

        if total_pages > strided_threshold:
            anchor_page: int | None = None
            candidate = total_pages - stride

            while candidate > 50:
                print(f"      [Strided Search] 抽样检查第 {candidate} 页...")
                sample_text = _ocr_page(pdf_path, candidate - 1, dpi=self._ocr_dpi)
                lower = sample_text.lower()
                if any(kw.lower() in lower for kw in _ANCHOR_KEYWORDS):
                    anchor_page = candidate
                    print(f"      [Strided Search] 在第 {candidate} 页命中核心财务锚点！")
                    break
                candidate -= stride

            if anchor_page is None:
                anchor_page = default_anchor
                print(
                    f"      [Strided Search] 未在任何抽样页命中锚点，"
                    f"使用默认锚点（第 {anchor_page} 页）"
                )

            # ── 阶段 2：局部全量展开 (Local Expansion) ──────────────────
            start_page = max(1, anchor_page - expansion_before)
            end_page = min(total_pages, anchor_page + expansion_after)
            force_ocr_all = True
            print(
                f"      [Local Expansion] 开始对目标窗口 "
                f"(第 {start_page} 页至第 {end_page} 页) 进行深度 OCR 解析..."
            )

        # ── 归一化为 0-based 索引区间 ─────────────────────────────────────
        idx_start = (start_page - 1) if start_page else 0
        idx_end = end_page if end_page else total_pages
        actual_start = idx_start + 1
        actual_end = idx_end

        chunks: list[Document] = []
        financial_page_count = 0
        ocr_page_count = 0

        for page_index in range(idx_start, idx_end):
            page_num = page_index + 1
            print(f"      [Parsing] 正在处理第 {page_num} 页...")

            if force_ocr_all:
                print(f"      [OCR] 识别中...")
                text = _ocr_page(pdf_path, page_index, dpi=self._ocr_dpi).strip()
                ocr_used = True
                ocr_page_count += 1
                is_financial = True
                financial_page_count += 1
            else:
                raw_text = (reader.pages[page_index].extract_text() or "").strip()
                ocr_used = False
                text = raw_text

                if self._is_financial_table_page(raw_text):
                    print(f"      [OCR] 识别中...")
                    text = _ocr_page(pdf_path, page_index, dpi=self._ocr_dpi).strip()
                    ocr_used = True
                    ocr_page_count += 1

                is_financial = self._is_financial_table_page(text)
                if is_financial:
                    financial_page_count += 1

            if not text:
                continue

            splitter = self._financial_splitter if is_financial else self._splitter
            page_chunks = splitter.create_documents(
                texts=[text],
                metadatas=[{
                    "page": page_num,
                    "source": str(pdf_path),
                    "page_char_count": len(text),
                    "is_financial_table": is_financial,
                    "ocr_fallback": ocr_used,
                }],
            )
            chunks.extend(page_chunks)

        if not chunks:
            raise ValueError(
                f"指定页面范围内未能提取到任何文本（含 OCR 降级）：{pdf_path}"
            )

        print(
            f"      [完成] 财务报表页：{financial_page_count} 页 | "
            f"OCR 识别页：{ocr_page_count} 页（解析范围：第 {actual_start}–{actual_end} 页）"
        )
        return chunks

    def find_pages_via_bookmarks(self, pdf_path: str) -> int | None:
        """
        Metadata 元数据书签路由：直接读取 PDF 内嵌 Outline（书签），
        秒级定位综合财务状况表（资产负债表）所在的绝对页码。

        匹配优先级：
            ['綜合財務狀況表', '资产负债表', 'Balance Sheet']

        Returns:
            目标书签的 1-based 绝对页码，未找到时返回 None。
        """
        _BOOKMARK_TARGETS = ["綜合財務狀況表", "资产负债表", "Balance Sheet"]

        reader = PdfReader(str(pdf_path))
        outline = reader.outline

        def _search(items) -> int | None:
            for item in items:
                if isinstance(item, list):
                    result = _search(item)
                    if result is not None:
                        return result
                else:
                    title = getattr(item, "title", "") or ""
                    if any(kw in title for kw in _BOOKMARK_TARGETS):
                        page_index = reader.get_destination_page_number(item)
                        return page_index + 1  # 转换为 1-based 人类页码
            return None

        return _search(outline)

    def find_target_pages_via_toc(self, pdf_path: str) -> tuple:
        """
        Agentic TOC 目录路由：对 PDF 第 2–15 页（目录高频区）做 OCR，
        将提取的目录文本交给 LLM 定位财务报表所在的绝对页码。

        Returns:
            (start_page: int, toc_text: str)
            start_page 为 LLM 推断的绝对页码（已含 +4 封面偏移）。
        """
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        _TOC_SYSTEM_PROMPT = (
            "你是一个金融分析师。请在这段财报目录中，找出【合并财务状况表 / "
            "Consolidated Statement of Financial Position / 资产负债表】"
            "以及【合并损益表 / 利润表】所在的起始页码。"
            "考虑到封面通常不计入纸质页码，请在提取的纸质页码基础上 自动 +4 作为绝对页码。"
            '请严格只返回一个 JSON 格式，如 {"start_page": 190}，'
            '找不到则返回 {"start_page": 170}。'
        )

        pdf_path = Path(pdf_path)
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)

        # 只扫描第 2–15 页（1-based），即 0-based 索引 1–14
        toc_end_index = min(15, total_pages)
        print(f"  [TOC Router] 正在 OCR 第 2–{toc_end_index} 页提取目录...")

        segments: list[str] = []
        for page_index in range(1, toc_end_index):
            page_num = page_index + 1
            print(f"  [TOC Router] OCR 第 {page_num} 页...")
            text = _ocr_page(pdf_path, page_index, dpi=self._ocr_dpi).strip()
            if text:
                segments.append(f"[第 {page_num} 页]\n{text}")

        toc_text = "\n\n".join(segments)

        # ── LLM 路由决策 ──────────────────────────────────────────────────
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=0,
            max_tokens=128,
        )
        response = llm.invoke([
            SystemMessage(content=_TOC_SYSTEM_PROMPT),
            HumanMessage(content=toc_text or "（目录页未提取到文字）"),
        ])
        raw = response.content.strip()
        print(f"  [TOC Router] LLM 原始响应：{raw}")

        start_page = 170  # 默认兜底
        try:
            match = re.search(r"\{[^}]+\}", raw)
            data = json.loads(match.group() if match else raw)
            start_page = int(data.get("start_page", 170))
        except (json.JSONDecodeError, AttributeError, ValueError, KeyError):
            print(f"  [TOC Router] JSON 解析失败，使用默认页码 {start_page}")

        print(f"  [TOC Router] 定位到财务报表起始绝对页码：{start_page}")
        return (start_page, toc_text)

    def _ocr_page_list(self, pdf_path: Path, page_indices: list[int]) -> list[Document]:
        """对指定的 0-based 页码列表全量 OCR，返回 chunk 列表（强制标记为财务页）。"""
        chunks: list[Document] = []
        for page_index in page_indices:
            page_num = page_index + 1
            print(f"      [Parsing] 正在处理第 {page_num} 页...")
            print(f"      [OCR] 识别中...")
            text = _ocr_page(pdf_path, page_index, dpi=self._ocr_dpi).strip()
            if not text:
                continue
            page_chunks = self._financial_splitter.create_documents(
                texts=[text],
                metadatas=[{
                    "page": page_num,
                    "source": str(pdf_path),
                    "page_char_count": len(text),
                    "is_financial_table": True,
                    "ocr_fallback": True,
                }],
            )
            chunks.extend(page_chunks)

        if not chunks:
            raise ValueError("指定页面范围内未能提取到任何文本（含 OCR 降级）")

        print(
            f"      [完成] OCR 识别页：{len(page_indices)} 页（"
            f"第 {page_indices[0]+1}–{page_indices[-1]+1} 页）"
        )
        return chunks
