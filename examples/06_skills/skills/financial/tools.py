import json
import sys
from pathlib import Path

from hypertic.tools import tool

skill_dir = Path(__file__).parent
if str(skill_dir) not in sys.path:
    sys.path.insert(0, str(skill_dir))

import calculate_ratio
import interpret_ratio

calculate_ratios_from_data = calculate_ratio.calculate_ratios_from_data
perform_comprehensive_analysis = interpret_ratio.perform_comprehensive_analysis


@tool
def calculate_financial_ratios(financial_data: str) -> str:
    """Calculate financial ratios from financial statement data.

    Takes financial data as JSON string and calculates all key financial ratios
    including profitability, liquidity, leverage, efficiency, and valuation ratios.

    Args:
        financial_data: JSON string containing financial statement data with keys:
            - income_statement: revenue, cost_of_goods_sold, net_income, etc.
            - balance_sheet: total_assets, current_assets, shareholders_equity, etc.
            - cash_flow: operating_cash_flow, etc.
            - market_data: stock_price, shares_outstanding, etc.

    Returns:
        JSON string with calculated ratios, interpretations, and summary
    """
    try:
        # Parse JSON string to dict
        if isinstance(financial_data, str):
            data = json.loads(financial_data)
        else:
            data = financial_data

        # Calculate ratios using the actual Python function
        result = calculate_ratios_from_data(data)

        # Return as formatted JSON
        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({"error": "Invalid JSON in financial_data", "details": str(e)}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Error calculating ratios: {e}", "details": str(e)}, indent=2)


@tool
def analyze_financial_ratios(ratios: str, industry: str = "general", historical_data: str | None = None) -> str:
    """Perform comprehensive analysis of financial ratios with interpretations.

    Analyzes calculated ratios, provides industry benchmarks, trend analysis,
    and recommendations.

    Args:
        ratios: JSON string of calculated financial ratios
        industry: Industry sector for benchmarking (default: "general")
        historical_data: Optional JSON string with historical ratio data for trends

    Returns:
        JSON string with comprehensive analysis including interpretations and recommendations
    """
    try:
        # Parse JSON strings
        if isinstance(ratios, str):
            ratios_dict = json.loads(ratios)
        else:
            ratios_dict = ratios

        hist_data = None
        if historical_data:
            if isinstance(historical_data, str):
                hist_data = json.loads(historical_data)
            else:
                hist_data = historical_data

        # Perform analysis using the actual Python function
        result = perform_comprehensive_analysis(ratios_dict, industry, hist_data)

        # Return as formatted JSON
        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({"error": "Invalid JSON in ratios or historical_data", "details": str(e)}, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Error analyzing ratios: {e}", "details": str(e)}, indent=2)
