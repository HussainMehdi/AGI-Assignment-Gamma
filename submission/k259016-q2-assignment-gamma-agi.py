from crewai import Agent, LLM
from crewai.tools import tool
from typing import Optional

llm_model = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434",
    
)

# ==================== KNOWLEDGE BASES ====================

# Stock Market Knowledge Base
STOCK_MARKET_KB = {
    "HBL": {
        "name": "Habib Bank Limited",
        "current_price": 125.50,
        "price_change": 2.30,
        "price_change_percent": 1.87,
        "volume": 1500000,
        "market_cap": "450 billion PKR",
        "sector": "Banking",
        "trend": "upward",
        "recommendation": "buy",
        "52_week_high": 135.00,
        "52_week_low": 98.50
    },
    "UBL": {
        "name": "United Bank Limited",
        "current_price": 185.75,
        "price_change": -1.25,
        "price_change_percent": -0.67,
        "volume": 980000,
        "market_cap": "380 billion PKR",
        "sector": "Banking",
        "trend": "sideways",
        "recommendation": "hold",
        "52_week_high": 195.00,
        "52_week_low": 165.00
    },
    "OGDC": {
        "name": "Oil and Gas Development Company",
        "current_price": 142.30,
        "price_change": 3.20,
        "price_change_percent": 2.30,
        "volume": 2100000,
        "market_cap": "620 billion PKR",
        "sector": "Oil & Gas",
        "trend": "upward",
        "recommendation": "buy",
        "52_week_high": 155.00,
        "52_week_low": 120.00
    },
    "PSO": {
        "name": "Pakistan State Oil",
        "current_price": 198.50,
        "price_change": -2.10,
        "price_change_percent": -1.05,
        "volume": 1750000,
        "market_cap": "420 billion PKR",
        "sector": "Oil & Gas",
        "trend": "downward",
        "recommendation": "sell",
        "52_week_high": 210.00,
        "52_week_low": 185.00
    },
    "KSE-100": {
        "index_value": 45230.50,
        "change": 245.30,
        "change_percent": 0.54,
        "trend": "upward",
        "top_gainers": ["HBL", "OGDC", "ENGRO"],
        "top_losers": ["PSO", "FFC", "FFBL"]
    }
}

# Property Market Knowledge Base
PROPERTY_MARKET_KB = {
    "Karachi DHA": {
        "location": "Defence Housing Authority, Karachi",
        "3-bedroom_apartment": {
            "price_range": "25-40 million PKR",
            "average_price": "32 million PKR",
            "price_per_sqft": "15000-20000 PKR",
            "trend": "upward",
            "demand": "high",
            "recommendation": "good investment"
        },
        "4-bedroom_house": {
            "price_range": "80-120 million PKR",
            "average_price": "95 million PKR",
            "price_per_sqft": "18000-25000 PKR",
            "trend": "upward",
            "demand": "very high",
            "recommendation": "excellent investment"
        },
        "commercial_plot": {
            "price_range": "150-300 million PKR",
            "average_price": "200 million PKR",
            "price_per_marla": "25-50 million PKR",
            "trend": "stable",
            "demand": "moderate"
        }
    },
    "Lahore DHA": {
        "location": "Defence Housing Authority, Lahore",
        "3-bedroom_apartment": {
            "price_range": "20-35 million PKR",
            "average_price": "28 million PKR",
            "price_per_sqft": "12000-18000 PKR",
            "trend": "upward",
            "demand": "high",
            "recommendation": "good investment"
        },
        "4-bedroom_house": {
            "price_range": "70-110 million PKR",
            "average_price": "85 million PKR",
            "price_per_sqft": "15000-22000 PKR",
            "trend": "upward",
            "demand": "high",
            "recommendation": "good investment"
        }
    },
    "Islamabad F-7": {
        "location": "F-7 Sector, Islamabad",
        "3-bedroom_apartment": {
            "price_range": "30-45 million PKR",
            "average_price": "37 million PKR",
            "price_per_sqft": "18000-25000 PKR",
            "trend": "upward",
            "demand": "very high",
            "recommendation": "excellent investment"
        }
    }
}

# Gold Price Knowledge Base
GOLD_PRICE_KB = {
    "24k_gold": {
        "price_per_tola": 245000,
        "price_per_gram": 21000,
        "price_per_ounce": 653000,
        "currency": "PKR",
        "trend": "upward",
        "change_24h": 2500,
        "change_percent_24h": 1.03,
        "recommendation": "buy",
        "market_sentiment": "bullish"
    },
    "22k_gold": {
        "price_per_tola": 224500,
        "price_per_gram": 19250,
        "trend": "upward",
        "change_24h": 2300,
        "change_percent_24h": 1.03
    },
    "21k_gold": {
        "price_per_tola": 214375,
        "price_per_gram": 18375,
        "trend": "upward",
        "change_24h": 2200,
        "change_percent_24h": 1.04
    },
    "historical_trend": {
        "1_month": "upward",
        "3_months": "upward",
        "6_months": "volatile",
        "1_year": "upward"
    }
}

# Car Market Knowledge Base
CAR_MARKET_KB = {
    "Toyota Corolla": {
        "model_year_2024": {
            "price_new": "6500000-7500000 PKR",
            "price_used_2023": "5500000-6500000 PKR",
            "price_used_2022": "4800000-5800000 PKR",
            "price_used_2021": "4200000-5200000 PKR",
            "depreciation_rate": "15-20% per year",
            "demand": "very high",
            "resale_value": "excellent",
            "recommendation": "good buy"
        },
        "model_year_2023": {
            "price_used": "5500000-6500000 PKR",
            "condition": "excellent",
            "resale_value": "high"
        }
    },
    "Honda Civic": {
        "model_year_2024": {
            "price_new": "7000000-8000000 PKR",
            "price_used_2023": "6000000-7000000 PKR",
            "price_used_2022": "5200000-6200000 PKR",
            "depreciation_rate": "15-18% per year",
            "demand": "very high",
            "resale_value": "excellent",
            "recommendation": "good buy"
        }
    },
    "Suzuki Alto": {
        "model_year_2024": {
            "price_new": "2500000-3000000 PKR",
            "price_used_2023": "2200000-2700000 PKR",
            "price_used_2022": "1900000-2400000 PKR",
            "depreciation_rate": "12-15% per year",
            "demand": "very high",
            "resale_value": "good",
            "recommendation": "budget-friendly option"
        }
    },
    "Toyota Camry": {
        "model_year_2024": {
            "price_new": "12000000-15000000 PKR",
            "price_used_2023": "10000000-13000000 PKR",
            "depreciation_rate": "18-22% per year",
            "demand": "moderate",
            "resale_value": "good"
        }
    },
    "market_trends": {
        "overall_demand": "high",
        "popular_segments": ["sedan", "SUV", "hatchback"],
        "import_trend": "increasing",
        "local_manufacturing": "stable"
    }
}


# ==================== TOOLS ====================

@tool("Stock Market Information Tool")
def stock_market_tool(stock_symbol: str) -> str:
    """
    Retrieve information about Pakistani stocks from the knowledge base.
    
    Args:
        stock_symbol: Stock symbol to look up (e.g., 'HBL', 'UBL', 'OGDC', 'PSO', 'KSE-100')
    
    Returns:
        Detailed stock information including price, trend, and recommendations
    """
    stock_symbol = stock_symbol.upper()
    if STOCK_MARKET_KB[stock_symbol]:
        return str(STOCK_MARKET_KB[stock_symbol])
    else:
        #  return keys like {available stocks: HBL, UBL, OGDC, PSO, KSE-100}
        return f"Available stocks: {', '.join(STOCK_MARKET_KB.keys())}"

@tool("Property Market Information Tool")
def property_market_tool(location: str, property_type: Optional[str] = None) -> str:
    """
    Retrieve property market information from the knowledge base.
    
    Args:
        location: Location name (e.g., 'Karachi DHA', 'Lahore DHA', 'Islamabad F-7')
        property_type: Type of property (e.g., '3-bedroom apartment', '4-bedroom house', 'commercial plot')
    
    Returns:
        Detailed property market information including prices, trends, and recommendations
    """
    location = location.title()
    
    if location in PROPERTY_MARKET_KB:
        location_info = PROPERTY_MARKET_KB[location]
        
        if property_type:
            return str(location_info[property_type])
        else:
            return f"Available property types for {location}: {', '.join(location_info.keys())}"
    else:
        return f"Available locations: {', '.join(PROPERTY_MARKET_KB.keys())}"

@tool("Gold Price Information Tool")
def gold_price_tool() -> str:
    """
    Retrieve current gold prices from the knowledge base.
    
    Args:
        None
    
    Returns:
        Current gold prices per tola, per gram, and market trends like {24k_gold, 22k_gold, 21k_gold, historical_trend}
    """
    
    return f"Current gold prices: {GOLD_PRICE_KB}"
    
@tool("Car Market Information Tool")
def car_market_tool(car_model: str, model_year: Optional[str] = None) -> str:
    """
    Retrieve car market information from the knowledge base.
    
    Args:
        car_model: Car model name (e.g., 'Toyota Corolla', 'Honda Civic', 'Suzuki Alto')
        model_year: Specific model year (e.g., '2024', '2023', '2022')
    
    Returns:
        Car prices, depreciation rates, and market information
    """
    car_model = car_model.title()
    
    if car_model in CAR_MARKET_KB:
        car_info = CAR_MARKET_KB[car_model]
        return str(car_info)
    else:
        return f"Available car models: {', '.join(CAR_MARKET_KB.keys())}"


# Portfolio Manager Agent (has access to all tools)
portfolio_manager_agent = Agent(
    role="Senior Portfolio Manager and Investment Advisor",
    goal="Analyze all available investment options using stock market, property market, gold prices, and car market tools to provide data-driven, comprehensive investment recommendations tailored to the investor's budget and goals",
    backstory="An experienced Pakistani financial advisor with deep expertise across multiple asset classes. You systematically use all available market analysis tools - stock_market_tool, property_market_tool, gold_price_tool, and car_market_tool - to gather current market data before making recommendations. You compare returns, risks, liquidity, and market trends across stocks, real estate, precious metals, and vehicles to provide holistic investment advice.",
    llm=llm_model,
    tools=[stock_market_tool, property_market_tool, gold_price_tool, car_market_tool],
    verbose=True,
    max_iter=10,
    allow_delegation=False
)
response = portfolio_manager_agent.kickoff("""You are a Pakistani portfolio manager advising an individual investor with a budget of 5,000,000 PKR.

IMPORTANT: You MUST use ALL available tools to gather current market information before making your recommendation:
1. Use stock_market_tool to check HBL stock prices, trends, and recommendations
2. Use property_market_tool to check 3-bedroom apartment prices and trends in Karachi DHA
3. Use gold_price_tool to check current 24k gold prices and market trends
4. Use car_market_tool to check Toyota Corolla 2024 prices, depreciation rates, and resale value

After gathering data from all tools, provide a comprehensive analysis comparing:
- HBL Stocks
- 3-bedroom apartment in Karachi DHA
- 24k gold
- Toyota Corolla 2024

For each option, analyze:
- Current market price/value
- Expected returns and growth potential
- Risk factors
- Liquidity
- Market trends
- Fit within the 5M PKR budget

Finally, provide your recommendation with clear reasoning based on the data you gathered from all tools.""")
print(response)