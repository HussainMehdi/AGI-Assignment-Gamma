from crewai import Agent, LLM
from crewai.tools import tool
import os
import sys
from datetime import datetime
from typing import Optional

llm_model = LLM(
    model="ollama/llama3.2",
    base_url="http://localhost:11434"
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
    
    if stock_symbol in STOCK_MARKET_KB:
        stock_info = STOCK_MARKET_KB[stock_symbol]
        
        if stock_symbol == "KSE-100":
            return f"KSE-100 Index: Value = {stock_info['index_value']}, Change = {stock_info['change']} ({stock_info['change_percent']}%), Trend = {stock_info['trend']}. Top Gainers: {', '.join(stock_info['top_gainers'])}. Top Losers: {', '.join(stock_info['top_losers'])}."
        else:
            return (f"{stock_info['name']} ({stock_symbol}): Current Price = {stock_info['current_price']} PKR, "
                   f"Change = {stock_info['price_change']} ({stock_info['price_change_percent']}%), "
                   f"Volume = {stock_info['volume']:,}, Market Cap = {stock_info['market_cap']}, "
                   f"Sector = {stock_info['sector']}, Trend = {stock_info['trend']}, "
                   f"Recommendation = {stock_info['recommendation']}, "
                   f"52W High = {stock_info['52_week_high']}, 52W Low = {stock_info['52_week_low']}")
    else:
        available_stocks = ", ".join([k for k in STOCK_MARKET_KB.keys() if k != "KSE-100"])
        return f"Stock '{stock_symbol}' not found in knowledge base. Available stocks: {available_stocks}, KSE-100"

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
        result = f"Property Market - {location_info['location']}:\n"
        
        if property_type:
            # Search for property type in the location data
            for key, value in location_info.items():
                if property_type.lower() in key.lower() or key.lower() in property_type.lower():
                    prop_data = value
                    result += f"\n{key.replace('_', ' ').title()}:\n"
                    for k, v in prop_data.items():
                        result += f"  {k.replace('_', ' ').title()}: {v}\n"
                    return result
            
            return f"Property type '{property_type}' not found for {location}. Available types: {', '.join([k for k in location_info.keys() if k != 'location'])}"
        else:
            # Return all property types for the location
            for key, value in location_info.items():
                if key != "location":
                    result += f"\n{key.replace('_', ' ').title()}:\n"
                    if isinstance(value, dict):
                        for k, v in value.items():
                            result += f"  {k.replace('_', ' ').title()}: {v}\n"
            return result
    else:
        available_locations = ", ".join(PROPERTY_MARKET_KB.keys())
        return f"Location '{location}' not found in knowledge base. Available locations: {available_locations}"

@tool("Gold Price Information Tool")
def gold_price_tool(gold_type: Optional[str] = "24k_gold") -> str:
    """
    Retrieve current gold prices from the knowledge base.
    
    Args:
        gold_type: Type of gold (e.g., '24k_gold', '22k_gold', '21k_gold')
    
    Returns:
        Current gold prices per tola, per gram, and market trends
    """
    gold_type = gold_type.lower()
    
    if gold_type in GOLD_PRICE_KB:
        gold_info = GOLD_PRICE_KB[gold_type]
        result = f"{gold_type.upper().replace('_', ' ')} Gold Prices:\n"
        result += f"  Price per Tola: {gold_info['price_per_tola']:,} {gold_info.get('currency', 'PKR')}\n"
        if 'price_per_gram' in gold_info:
            result += f"  Price per Gram: {gold_info['price_per_gram']:,} {gold_info.get('currency', 'PKR')}\n"
        if 'price_per_ounce' in gold_info:
            result += f"  Price per Ounce: {gold_info['price_per_ounce']:,} {gold_info.get('currency', 'PKR')}\n"
        result += f"  Trend: {gold_info['trend']}\n"
        if 'change_24h' in gold_info:
            result += f"  24h Change: {gold_info['change_24h']:,} ({gold_info.get('change_percent_24h', 0)}%)\n"
        if 'recommendation' in gold_info:
            result += f"  Recommendation: {gold_info['recommendation']}\n"
        if 'market_sentiment' in gold_info:
            result += f"  Market Sentiment: {gold_info['market_sentiment']}\n"
        return result
    elif gold_type == "historical":
        hist_info = GOLD_PRICE_KB.get("historical_trend", {})
        return f"Gold Historical Trends: 1 Month = {hist_info.get('1_month')}, 3 Months = {hist_info.get('3_months')}, 6 Months = {hist_info.get('6_months')}, 1 Year = {hist_info.get('1_year')}"
    else:
        available_types = ", ".join([k for k in GOLD_PRICE_KB.keys() if k != "historical_trend"])
        return f"Gold type '{gold_type}' not found. Available types: {available_types}, historical"

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
        result = f"{car_model} Market Information:\n"
        
        if model_year:
            year_key = f"model_year_{model_year}"
            if year_key in car_info:
                year_data = car_info[year_key]
                result += f"\n{model_year} Model:\n"
                for k, v in year_data.items():
                    result += f"  {k.replace('_', ' ').title()}: {v}\n"
                return result
            else:
                available_years = [k.replace("model_year_", "") for k in car_info.keys() if k.startswith("model_year_")]
                return f"Model year '{model_year}' not found for {car_model}. Available years: {', '.join(available_years)}"
        else:
            # Return all available model years
            for key, value in car_info.items():
                if key.startswith("model_year_"):
                    year = key.replace("model_year_", "")
                    result += f"\n{year} Model:\n"
                    if isinstance(value, dict):
                        for k, v in value.items():
                            result += f"  {k.replace('_', ' ').title()}: {v}\n"
            if "market_trends" in car_info:
                result += f"\nMarket Trends:\n"
                for k, v in car_info["market_trends"].items():
                    result += f"  {k.replace('_', ' ').title()}: {v}\n"
            return result
    else:
        available_models = ", ".join([k for k in CAR_MARKET_KB.keys() if k != "market_trends"])
        return f"Car model '{car_model}' not found in knowledge base. Available models: {available_models}"

def safe_print(*args, **kwargs):
    """Safe print function that avoids recursion issues with CrewAI's output system"""
    try:
        # Use sys.stdout.write directly to avoid rich proxy issues
        message = ' '.join(str(arg) for arg in args)
        sys.stdout.write(message + '\n')
        sys.stdout.flush()
    except Exception:
        # Fallback to regular print if sys.stdout fails
        try:
            print(*args, **kwargs)
        except Exception:
            pass  # Silently fail if printing is not possible

# Stock Market Agent (agent will call the stock tool)
stock_market_agent = Agent(
    role="Stock Market Analyst",
    goal="Analyze Pakistani stock market trends, prices, and provide buy/sell recommendations",
    backstory="Expert in PSX (Pakistan Stock Exchange) with deep knowledge of market trends, company fundamentals, and technical analysis. Uses the stock market tool to fetch data before advising.",
    llm=llm_model,
    tools=[stock_market_tool],
    verbose=True,
    max_iter=3,
    allow_delegation=False
)

# Property Market Agent (agent will call the property tool)
property_market_agent = Agent(
    role="Property Market Specialist",
    goal="Analyze Pakistani property market prices, trends, and investment opportunities",
    backstory="Real estate expert specializing in Karachi, Lahore, and Islamabad property markets with knowledge of current rates and trends. Uses the property market tool to fetch local pricing and demand.",
    llm=llm_model,
    tools=[property_market_tool],
    verbose=True,
    max_iter=3,
    allow_delegation=False
)

# Gold Price Agent (agent will call the gold tool)
gold_price_agent = Agent(
    role="Gold Price Analyst",
    goal="Track and analyze gold prices in Pakistani market (per tola/gram)",
    backstory="Specialist in precious metals market, tracking daily gold rates, trends, and providing investment advice for gold trading. Uses the gold price tool to retrieve rates.",
    llm=llm_model,
    tools=[gold_price_tool],
    verbose=True,
    max_iter=3,
    allow_delegation=False
)

# Car Market Agent (agent will call the car tool)
car_market_agent = Agent(
    role="Automotive Market Analyst",
    goal="Analyze car prices, market trends, and provide buy/sell recommendations for vehicles",
    backstory="Expert in Pakistani automotive market, familiar with local and imported car prices, depreciation rates, and market demand. Uses the car market tool for pricing and depreciation.",
    llm=llm_model,
    tools=[car_market_tool],
    verbose=True,
    max_iter=3,
    allow_delegation=False
)

# Portfolio Manager Agent (has access to all tools)
portfolio_manager_agent = Agent(
    role="Portfolio Manager",
    goal="Coordinate asset analysis and provide comprehensive investment recommendations",
    backstory="Financial advisor who synthesizes information from stocks, property, gold, and car markets, using available tools to form holistic strategies.",
    llm=llm_model,
    tools=[stock_market_tool, property_market_tool, gold_price_tool, car_market_tool],
    verbose=True,
    max_iter=2,
    allow_delegation=False
)

def generate_transaction_id():
    """Generate a unique transaction ID for asset operations"""
    counter_file = "transaction_counter.txt"
    
    # Create file if not exists
    if not os.path.exists(counter_file):
        with open(counter_file, "w") as f:
            f.write("0")
    
    # Read counter
    with open(counter_file, "r") as f:
        counter = int(f.read().strip())
    
    # Increment and save
    counter += 1
    with open(counter_file, "w") as f:
        f.write(str(counter))
    
    # Return formatted id
    return f"TXN{counter:04d}"

def update_transaction_file(transaction_id, agent_name, note):
    """Update transaction file with agent notes"""
    file_path = f"transaction_{transaction_id}.md"
    
    # Create file with header if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Transaction Report: {transaction_id}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"## {agent_name}\n")
        f.write(f"{note}\n\n")

def check_stock_market(transaction_id, stock_symbol=None, action="check"):
    """Check stock market prices and trends using the stock_market_tool directly, then ask the agent to interpret it."""
    # 1) Use the tool deterministically
    stock_key = stock_symbol if stock_symbol is not None else "KSE-100"
    try:
        tool_output = stock_market_tool(stock_key)
    except Exception as e:
        tool_output = f"Tool error: {e}"

    # 2) Let the agent interpret the tool output
    if stock_symbol:
        user_goal = f"for stock {stock_symbol} and whether to {action}"
    else:
        user_goal = f"for the overall PSX market and whether to {action}"

    prompt = (
        "You are a Pakistani stock market analyst.\n"
        "You already have the tool output below—use it as your evidence.\n\n"
        "Tool output data:\n"
        f"{tool_output}\n\n"
        f"Task: Based ONLY on the tool output data above, explain the situation in simple terms and give a clear recommendation {user_goal}.\n"
        "Provide your final answer now."
    )

    try:
        response = stock_market_agent.kickoff(prompt)
        analysis = str(response) if response else "No analysis generated by agent."
    except Exception as e:
        analysis = f"Agent error while analyzing stock data: {e}"

    note = "TOOL OUTPUT:\n" + tool_output + "\n\nAGENT ANALYSIS:\n" + analysis

    safe_print(f"Stock Market Agent response ({transaction_id}): {analysis}")
    update_transaction_file(transaction_id, "Stock Market Analyst", note)

    return note

def check_property_market(transaction_id, location=None, property_type=None, action="check"):
    """Check property market prices and trends using the property_market_tool directly, then ask the agent to interpret it."""
    target_location = location if location is not None else "Karachi DHA"
    try:
        tool_output = property_market_tool(target_location, property_type)
    except Exception as e:
        tool_output = f"Tool error: {e}"

    loc_desc = f"in {target_location}"
    if property_type:
        loc_desc += f" for a {property_type}"

    prompt = (
        "You are a Pakistani property market specialist.\n"
        "You already have the tool output below—use it as your evidence.\n\n"
        "Tool output data:\n"
        f"{tool_output}\n\n"
        f"Task: Based ONLY on the tool output data above, explain the property market situation {loc_desc} and give a clear recommendation whether the user should {action}, hold, or avoid.\n"
        "Provide your final answer now.\n"
        "Mention approximate price ranges and demand in simple language."
    )

    try:
        response = property_market_agent.kickoff(prompt)
        analysis = str(response) if response else "No analysis generated by agent."
    except Exception as e:
        analysis = f"Agent error while analyzing property data: {e}"

    note = "TOOL OUTPUT:\n" + tool_output + "\n\nAGENT ANALYSIS:\n" + analysis

    safe_print(f"Property Market Agent response ({transaction_id}): {analysis}")
    update_transaction_file(transaction_id, "Property Market Specialist", note)

    return note

def check_gold_price(transaction_id, action="check"):
    """Check current gold prices in Pakistani market using the gold_price_tool, then ask the agent to interpret it."""
    try:
        tool_output = gold_price_tool()
    except Exception as e:
        tool_output = f"Tool error: {e}"

    prompt = (
        "You are a Pakistani gold price analyst.\n"
        "You already have the tool output below—use it as your evidence.\n\n"
        "Tool output data:\n"
        f"{tool_output}\n\n"
        f"Task: Based ONLY on the tool output data above, summarize the current gold prices and trend in PKR and give a clear recommendation whether the user should {action}, hold, or avoid.\n"
        "Keep it short and practical."
    )

    try:
        response = gold_price_agent.kickoff(prompt)
        analysis = str(response) if response else "No analysis generated by agent."
    except Exception as e:
        analysis = f"Agent error while analyzing gold data: {e}"

    note = "TOOL OUTPUT:\n" + tool_output + "\n\nAGENT ANALYSIS:\n" + analysis

    safe_print(f"Gold Price Agent response ({transaction_id}): {analysis}")
    update_transaction_file(transaction_id, "Gold Price Analyst", note)

    return note

def check_car_market(transaction_id, car_model=None, action="check"):
    """Check car market prices and trends using the car_market_tool, then ask the agent to interpret it."""
    target_model = car_model if car_model is not None else "Toyota Corolla"
    try:
        tool_output = car_market_tool(target_model)
    except Exception as e:
        tool_output = f"Tool error: {e}"

    prompt = (
        "You are an automotive market analyst for Pakistan.\n"
        "You already have the tool output below—use it as your evidence.\n\n"
        "Tool output data:\n"
        f"{tool_output}\n\n"
        f"Task: Based ONLY on the tool output data above, explain the price range, depreciation, and resale value for {target_model} and give a clear recommendation whether the user should {action}, hold, or avoid.\n"
        "Keep the answer focused on practical advice."
    )

    try:
        response = car_market_agent.kickoff(prompt)
        analysis = str(response) if response else "No analysis generated by agent."
    except Exception as e:
        analysis = f"Agent error while analyzing car market data: {e}"

    note = "TOOL OUTPUT:\n" + tool_output + "\n\nAGENT_ANALYSIS:\n" + analysis

    safe_print(f"Car Market Agent response ({transaction_id}): {analysis}")
    update_transaction_file(transaction_id, "Automotive Market Analyst", note)

    return note

def get_portfolio_recommendation(transaction_id, asset_type=None, budget=None, action="check"):
    """Get comprehensive portfolio recommendation by explicitly calling all tools, then asking the portfolio manager to synthesize."""
    # 1) Collect deterministic data from all tools
    try:
        stock_view = stock_market_tool("KSE-100")
    except Exception as e:
        stock_view = f"Stock tool error: {e}"

    try:
        property_view = property_market_tool("Karachi DHA", "3-bedroom apartment")
    except Exception as e:
        property_view = f"Property tool error: {e}"

    try:
        gold_view = gold_price_tool()
    except Exception as e:
        gold_view = f"Gold tool error: {e}"

    try:
        car_view = car_market_tool("Toyota Corolla")
    except Exception as e:
        car_view = f"Car tool error: {e}"

    asset_str = f" for {asset_type}" if asset_type else ""
    budget_str = f" with budget of {budget}" if budget else ""

    # 2) Ask portfolio manager to reason over all tool outputs
    prompt = (
        "You are a Pakistani portfolio manager advising an individual investor.\n"
        "You already have the tool outputs below—use them as your evidence.\n\n"
        "Tool output data from four tools:\n\n"
        f"=== STOCK MARKET (KSE-100 / key stocks) ===\n{stock_view}\n\n"
        f"=== PROPERTY MARKET (Karachi DHA, 3-bedroom apartment) ===\n{property_view}\n\n"
        f"=== GOLD MARKET (24k baseline) ===\n{gold_view}\n\n"
        f"=== CAR MARKET (Toyota Corolla) ===\n{car_view}\n\n"
        f"Task: The user wants a recommendation{asset_str}{budget_str} in Pakistan.\n"
        f"Based ONLY on the tool data provided above, suggest how they should allocate or rebalance between stocks, property, gold, and car(s) and what their best {action} strategy is.\n"
        "Be concise but specific with percentages or rough ranges."
    )

    try:
        response = portfolio_manager_agent.kickoff(prompt)
        analysis = str(response) if response else "No analysis generated by agent."
    except Exception as e:
        analysis = f"Agent error while building portfolio recommendation: {e}"

    note_parts = [
        "TOOLS SUMMARY:",
        "=== STOCK TOOL OUTPUT ===",
        stock_view,
        "=== PROPERTY TOOL OUTPUT ===",
        property_view,
        "=== GOLD TOOL OUTPUT ===",
        gold_view,
        "=== CAR TOOL OUTPUT ===",
        car_view,
        "",
        "PORTFOLIO MANAGER ANALYSIS:",
        analysis,
    ]
    note = "\n".join(note_parts)

    safe_print(f"Portfolio Manager response ({transaction_id}): {analysis}")
    update_transaction_file(transaction_id, "Portfolio Manager", note)

    return note

def buy_asset(transaction_id, asset_type, asset_details):
    """Execute buy operation for an asset using the appropriate tool output, then ask the relevant agent to advise."""
    asset_type_lower = asset_type.lower()

    # 1) Get tool data for the specific asset
    if asset_type_lower == "stock":
        try:
            tool_output = stock_market_tool(asset_details.split()[0])
        except Exception as e:
            tool_output = f"Tool error: {e}"
        agent = stock_market_agent
        agent_name = "Stock Market Analyst"
    elif asset_type_lower in ["property", "real estate"]:
        try:
            tool_output = property_market_tool("Karachi DHA", "3-bedroom apartment")
        except Exception as e:
            tool_output = f"Tool error: {e}"
        agent = property_market_agent
        agent_name = "Property Market Specialist"
    elif asset_type_lower == "gold":
        try:
            tool_output = gold_price_tool()
        except Exception as e:
            tool_output = f"Tool error: {e}"
        agent = gold_price_agent
        agent_name = "Gold Price Analyst"
    elif asset_type_lower == "car":
        try:
            tool_output = car_market_tool(asset_details)
        except Exception as e:
            tool_output = f"Tool error: {e}"
        agent = car_market_agent
        agent_name = "Automotive Market Analyst"
    else:
        # Fallback to portfolio manager
        try:
            tool_output = "Using all tools via portfolio manager context."
        except Exception as e:
            tool_output = f"Tool error: {e}"
        agent = portfolio_manager_agent
        agent_name = "Portfolio Manager"

    # 2) Ask the agent whether buying is a good idea
    prompt = (
        f"You are the {agent_name}.\n"
        f"The user is considering a BUY operation for: {asset_details} ({asset_type}).\n\n"
        "You already have the tool output below—use it as your evidence.\n\n"
        "Tool output data:\n"
        f"{tool_output}\n\n"
        "Task: Based ONLY on the tool output data above, explain whether buying now is a good idea, what risks exist, "
        "and at what price range or quantity it would be reasonable to buy.\n"
        "Keep it concise and practical."
    )

    try:
        response = agent.kickoff(prompt)
        analysis = str(response) if response else "No analysis generated by agent."
    except Exception as e:
        analysis = f"Agent error while evaluating buy operation: {e}"

    note = "BUY OPERATION\n\nTOOL DATA:\n" + tool_output + "\n\nAGENT ANALYSIS:\n" + analysis

    safe_print(f"{agent_name} BUY response ({transaction_id}): {analysis}")
    update_transaction_file(transaction_id, f"{agent_name} - BUY", note)

    return note

def sell_asset(transaction_id, asset_type, asset_details):
    """Execute sell operation for an asset using the appropriate tool output, then ask the relevant agent to advise."""
    asset_type_lower = asset_type.lower()

    # 1) Get tool data for the specific asset
    if asset_type_lower == "stock":
        try:
            tool_output = stock_market_tool(asset_details.split()[0])
        except Exception as e:
            tool_output = f"Tool error: {e}"
        agent = stock_market_agent
        agent_name = "Stock Market Analyst"
    elif asset_type_lower in ["property", "real estate"]:
        try:
            tool_output = property_market_tool("Karachi DHA", "3-bedroom apartment")
        except Exception as e:
            tool_output = f"Tool error: {e}"
        agent = property_market_agent
        agent_name = "Property Market Specialist"
    elif asset_type_lower == "gold":
        try:
            tool_output = gold_price_tool()
        except Exception as e:
            tool_output = f"Tool error: {e}"
        agent = gold_price_agent
        agent_name = "Gold Price Analyst"
    elif asset_type_lower == "car":
        try:
            tool_output = car_market_tool(asset_details)
        except Exception as e:
            tool_output = f"Tool error: {e}"
        agent = car_market_agent
        agent_name = "Automotive Market Analyst"
    else:
        try:
            tool_output = "Using all tools via portfolio manager context."
        except Exception as e:
            tool_output = f"Tool error: {e}"
        agent = portfolio_manager_agent
        agent_name = "Portfolio Manager"

    # 2) Ask the agent whether selling is a good idea
    prompt = (
        f"You are the {agent_name}.\n"
        f"The user is considering a SELL operation for: {asset_details} ({asset_type}).\n\n"
        "You already have the tool output below—use it as your evidence.\n\n"
        "Tool output data:\n"
        f"{tool_output}\n\n"
        "Task: Based ONLY on the tool output data above, explain whether selling now is a good idea, what risks exist, "
        "and at what price range or timing it would be reasonable to sell.\n"
        "Keep it concise and practical."
    )

    try:
        response = agent.kickoff(prompt)
        analysis = str(response) if response else "No analysis generated by agent."
    except Exception as e:
        analysis = f"Agent error while evaluating sell operation: {e}"

    note = "SELL OPERATION\n\nTOOL DATA:\n" + tool_output + "\n\nAGENT ANALYSIS:\n" + analysis

    safe_print(f"{agent_name} SELL response ({transaction_id}): {analysis}")
    update_transaction_file(transaction_id, f"{agent_name} - SELL", note)

    return note

# Example usage
if __name__ == "__main__":
    # Generate a transaction ID
    txn_id = generate_transaction_id()
    safe_print(f"\n=== Starting Asset Market Analysis: {txn_id} ===\n")
    
    # Check all markets
    safe_print("1. Checking Stock Market...")
    check_stock_market(txn_id, stock_symbol="HBL", action="buy")
    
    safe_print("\n2. Checking Property Market...")
    check_property_market(txn_id, location="Karachi DHA", property_type="3-bedroom apartment", action="buy")
    
    safe_print("\n3. Checking Gold Prices...")
    check_gold_price(txn_id, action="buy")
    
    safe_print("\n4. Checking Car Market...")
    check_car_market(txn_id, car_model="Toyota Corolla", action="buy")
    
    safe_print("\n5. Getting Portfolio Recommendation...")
    get_portfolio_recommendation(txn_id, budget="5000000 PKR", action="investment")
    
    safe_print("\n6. Executing Buy Operation...")
    buy_asset(txn_id, "stock", "HBL - 100 shares")
    
    safe_print("\n7. Executing Sell Operation...")
    sell_asset(txn_id, "gold", "5 tola gold bars")
    
    safe_print(f"\n=== Transaction {txn_id} Complete ===\n")

