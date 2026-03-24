"""
vision_diagnostic.py — 使用视觉模型从财报 PDF 中提取财务数据

用法:
    python -m tools.vision_diagnostic <pdf_path> <page_number>

示例:
    python -m tools.vision_diagnostic ~/Projects/fin_risk_agent/data/raw/report.pdf 160

注意: 原始需求指定模型 claude-3-5-sonnet-20240620，该模型已于 2025-10-28 退役。
      如遇 404/NotFoundError，请将 MODEL 改为 claude-sonnet-4-6。
"""

import sys
import json
import base64
import tempfile
from pathlib import Path

import anthropic
from pdf2image import convert_from_path

# ── 配置 ──────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"  # 原需求指定 claude-3-5-sonnet-20240620，已于 2025-10-28 退役，改用当前等效视觉模型
DPI = 300
SYSTEM_PROMPT = (
    "你是一个专业的金融数据提取专家。"
    "这是一张美团财报的【合并财务状况表 / 资产负债表】图片。"
    "请无视繁体字或可能的排版乱序，利用图片中的空间对齐关系，"
    "精准提取以下科目：流動資產總值、流動負債總值、年内溢利（净利润）。 "
    '只返回 JSON 格式，如 {"current_assets": 124500, "current_liabilities": 108200, "net_profit": 15400}。'
)
# ─────────────────────────────────────────────────────────────────────────────


def pdf_page_to_base64(pdf_path: str, page_number: int) -> str:
    """将 PDF 指定页转换为 300 DPI PNG，返回 base64 字符串。"""
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    print(f"[1/3] 转换 PDF 第 {page_number} 页 (300 DPI)...")
    images = convert_from_path(
        str(pdf_path),
        dpi=DPI,
        first_page=page_number,
        last_page=page_number,
    )
    if not images:
        raise ValueError(f"无法从 PDF 提取第 {page_number} 页，请确认页码正确。")

    image = images[0]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        image.save(tmp_path, "PNG")

    print(f"   图片尺寸: {image.size[0]}x{image.size[1]} px  ->  {tmp_path}")

    with open(tmp_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("utf-8")

    Path(tmp_path).unlink(missing_ok=True)
    return b64


def extract_financials(image_b64: str) -> dict:
    """将 base64 图片传给 Claude 视觉模型，返回解析后的 JSON dict。"""
    client = anthropic.Anthropic()

    print(f"[2/3] 调用视觉模型 {MODEL} 解析财务数据...")
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "请根据图片提取财务数据，只返回 JSON。",
                    },
                ],
            }
        ],
    )

    raw_text = next(
        (block.text for block in response.content if block.type == "text"), ""
    ).strip()

    print(f"[3/3] 模型原始返回:\n{raw_text}\n")

    # 提取 JSON（兼容模型在 JSON 外附带说明文字的情况）
    start = raw_text.find("{")
    end = raw_text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"模型未返回有效 JSON，原始输出: {raw_text}")

    return json.loads(raw_text[start:end])


def main():
    if len(sys.argv) != 3:
        print("用法: python -m tools.vision_diagnostic <pdf_path> <page_number>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    try:
        page_number = int(sys.argv[2])
    except ValueError:
        print(f"错误: 页码必须为整数，收到: {sys.argv[2]}")
        sys.exit(1)

    try:
        image_b64 = pdf_page_to_base64(pdf_path, page_number)
        result = extract_financials(image_b64)

        print("=" * 50)
        print("提取结果 (JSON):")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("=" * 50)

        # 友好展示
        assets = result.get("current_assets", "N/A")
        liabilities = result.get("current_liabilities", "N/A")
        profit = result.get("net_profit", "N/A")
        print(f"流動資產總值:  {assets:,}" if isinstance(assets, (int, float)) else f"流動資產總值:  {assets}")
        print(f"流動負債總值:  {liabilities:,}" if isinstance(liabilities, (int, float)) else f"流動負債總值:  {liabilities}")
        print(f"年内溢利:      {profit:,}" if isinstance(profit, (int, float)) else f"年内溢利:      {profit}")

    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except anthropic.NotFoundError:
        print(
            f"错误: 模型 '{MODEL}' 不可用（可能已退役）。\n"
            "请将 tools/vision_diagnostic.py 中的 MODEL 改为 'claude-sonnet-4-6' 后重试。"
        )
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
