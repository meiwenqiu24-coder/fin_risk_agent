"""
tests/test_schema.py
单元测试：验证 FinancialRiskReport Schema 的正常校验与类型错误拦截。
"""

import pytest
from pydantic import ValidationError

from src.schema.risk_metrics import (
    FinancialRiskReport,
    LiquidityRisk,
    OperationalHealth,
    Profitability,
    RiskLevel,
)

# ---------------------------------------------------------------------------
# 共享 fixture：构造一份合法的模拟财报数据
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_report_data() -> dict:
    return {
        "liquidity_risk": {
            "liquidity_ratio": 2.35,
            "quick_ratio": 1.18,
            "source_text": "截至报告期末，公司流动资产合计 47.2 亿元，流动负债合计 20.1 亿元。",
            "page_number": 42,
        },
        "profitability": {
            "roe": 0.156,
            "gross_margin": 0.382,
            "source_text": "本报告期净利润为 8.3 亿元，加权平均净资产收益率为 15.6%，毛利率为 38.2%。",
            "page_number": 67,
        },
        "operational_health": {
            "inventory_turnover": 6.4,
            "source_text": "报告期内存货周转率为 6.4 次，较上年同期提升 0.8 次。",
            "page_number": 55,
        },
        "overall_risk_level": RiskLevel.GREEN,
        "assessment_summary": (
            "公司流动性充裕，短期偿债压力较小；盈利能力稳健，ROE 处于行业中上水平；"
            "存货周转效率持续改善，整体运营健康度良好，综合风险等级为绿色。"
        ),
    }


# ---------------------------------------------------------------------------
# 测试 1：合法数据 —— 正常构建并校验字段值
# ---------------------------------------------------------------------------

class TestValidReport:
    def test_build_success(self, valid_report_data):
        """使用完整合法数据可以成功构建 FinancialRiskReport 实例。"""
        report = FinancialRiskReport(**valid_report_data)
        assert isinstance(report, FinancialRiskReport)

    def test_liquidity_fields(self, valid_report_data):
        """流动性风险子模型字段值与输入一致。"""
        report = FinancialRiskReport(**valid_report_data)
        assert report.liquidity_risk.liquidity_ratio == pytest.approx(2.35)
        assert report.liquidity_risk.quick_ratio == pytest.approx(1.18)
        assert report.liquidity_risk.page_number == 42

    def test_profitability_fields(self, valid_report_data):
        """盈利能力子模型字段值与输入一致。"""
        report = FinancialRiskReport(**valid_report_data)
        assert report.profitability.roe == pytest.approx(0.156)
        assert report.profitability.gross_margin == pytest.approx(0.382)

    def test_operational_health_fields(self, valid_report_data):
        """营运健康度子模型字段值与输入一致。"""
        report = FinancialRiskReport(**valid_report_data)
        assert report.operational_health.inventory_turnover == pytest.approx(6.4)

    def test_risk_level_enum(self, valid_report_data):
        """overall_risk_level 应为 RiskLevel.GREEN 枚举值。"""
        report = FinancialRiskReport(**valid_report_data)
        assert report.overall_risk_level == RiskLevel.GREEN

    def test_serialization_roundtrip(self, valid_report_data):
        """model_dump / model_validate 往返序列化结果一致。"""
        report = FinancialRiskReport(**valid_report_data)
        dumped = report.model_dump()
        restored = FinancialRiskReport.model_validate(dumped)
        assert restored == report


# ---------------------------------------------------------------------------
# 测试 2：类型错误 —— 验证 Pydantic 能正确拦截非法数据
# ---------------------------------------------------------------------------

class TestTypeValidationErrors:
    def test_liquidity_ratio_string_rejected(self, valid_report_data):
        """liquidity_ratio 传入无法转换为 float 的字符串时，应抛出 ValidationError。"""
        valid_report_data["liquidity_risk"]["liquidity_ratio"] = "不是数字"
        with pytest.raises(ValidationError) as exc_info:
            FinancialRiskReport(**valid_report_data)

        errors = exc_info.value.errors()
        locs = [e["loc"] for e in errors]
        assert ("liquidity_risk", "liquidity_ratio") in locs

    def test_roe_string_rejected(self, valid_report_data):
        """roe 传入无法转换为 float 的字符串时，应抛出 ValidationError。"""
        valid_report_data["profitability"]["roe"] = "高收益"
        with pytest.raises(ValidationError) as exc_info:
            FinancialRiskReport(**valid_report_data)

        errors = exc_info.value.errors()
        locs = [e["loc"] for e in errors]
        assert ("profitability", "roe") in locs

    def test_inventory_turnover_string_rejected(self, valid_report_data):
        """inventory_turnover 传入无法转换为 float 的字符串时，应抛出 ValidationError。"""
        valid_report_data["operational_health"]["inventory_turnover"] = "快"
        with pytest.raises(ValidationError) as exc_info:
            FinancialRiskReport(**valid_report_data)

        errors = exc_info.value.errors()
        locs = [e["loc"] for e in errors]
        assert ("operational_health", "inventory_turnover") in locs

    def test_invalid_risk_level_rejected(self, valid_report_data):
        """overall_risk_level 传入枚举之外的值时，应抛出 ValidationError。"""
        valid_report_data["overall_risk_level"] = "BLUE"
        with pytest.raises(ValidationError) as exc_info:
            FinancialRiskReport(**valid_report_data)

        errors = exc_info.value.errors()
        locs = [e["loc"] for e in errors]
        assert ("overall_risk_level",) in locs

    def test_missing_required_field_rejected(self, valid_report_data):
        """缺少必填字段 assessment_summary 时，应抛出 ValidationError。"""
        del valid_report_data["assessment_summary"]
        with pytest.raises(ValidationError) as exc_info:
            FinancialRiskReport(**valid_report_data)

        errors = exc_info.value.errors()
        locs = [e["loc"] for e in errors]
        assert ("assessment_summary",) in locs
