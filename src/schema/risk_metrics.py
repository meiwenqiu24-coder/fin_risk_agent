"""
财务风控指标 Schema
用于 RAG 提取结构化财务数据，每个子模型均附带原文溯源字段以防止幻觉。
"""

from enum import Enum
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """整体风险等级枚举"""
    RED = "RED"        # 高风险
    YELLOW = "YELLOW"  # 中风险
    GREEN = "GREEN"    # 低风险


class LiquidityRisk(BaseModel):
    """流动性风险指标"""

    liquidity_ratio: float = Field(
        description="流动比率：流动资产 / 流动负债，衡量企业短期偿债能力，一般认为大于 2 较为健康"
    )
    quick_ratio: float = Field(
        description="速动比率：(流动资产 - 存货) / 流动负债，剔除存货后的短期偿债能力，一般认为大于 1 较为健康"
    )
    source_text: str = Field(
        description="从财报中提取上述流动性指标的原始文本片段，用于溯源核验"
    )
    page_number: int = Field(
        description="上述原始文本片段所在财报页码"
    )


class Profitability(BaseModel):
    """盈利能力指标"""

    roe: float = Field(
        description="净资产收益率（ROE）：净利润 / 平均股东权益，衡量股东权益的回报水平，越高代表盈利能力越强"
    )
    net_profit_margin: float = Field(
        description="净利率：净利润（或年内溢利）/ 营业收入（或收入），反映企业最终盈利能力，取值范围 0~1"
    )
    source_text: str = Field(
        description="从财报中提取上述盈利能力指标的原始文本片段，用于溯源核验"
    )
    page_number: int = Field(
        description="上述原始文本片段所在财报页码"
    )


class OperationalHealth(BaseModel):
    """营运健康度指标"""

    inventory_turnover: float = Field(
        description="存货周转率：营业成本 / 平均存货，衡量企业存货管理效率，数值越高表示存货周转越快、占用资金越少"
    )
    source_text: str = Field(
        description="从财报中提取上述营运健康度指标的原始文本片段，用于溯源核验"
    )
    page_number: int = Field(
        description="上述原始文本片段所在财报页码"
    )


class FinancialRiskReport(BaseModel):
    """顶层财务风险报告，整合流动性风险、盈利能力与营运健康度三个维度"""

    liquidity_risk: LiquidityRisk = Field(
        description="流动性风险维度指标"
    )
    profitability: Profitability = Field(
        description="盈利能力维度指标"
    )
    operational_health: OperationalHealth = Field(
        description="营运健康度维度指标"
    )
    overall_risk_level: RiskLevel = Field(
        description="综合风险等级：GREEN（低风险）/ YELLOW（中风险）/ RED（高风险）"
    )
    assessment_summary: str = Field(
        description="基于上述三个维度的综合风险评估总结，应简明描述主要风险点及建议"
    )
