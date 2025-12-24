from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from typing import Optional
from langchain_core.messages import AIMessage

# Note: Install required packages with:
# pip install -U langchain langchain-ollama langchain-community faiss-cpu langchain-text-splitters langchain-huggingface sentence-transformers
# 
# Install and run Ollama locally:
# 1. Download Ollama from: https://ollama.com/download
# 2. Pull a model: ollama pull llama3.2
# 3. Make sure Ollama is running: ollama serve

# Initialize Ollama LLM (local, offline, free)
# Available models: "llama3.2", "llama3.1", "mistral", "phi3", etc.
# Make sure you've pulled the model first: ollama pull llama3.2
llm_model = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",  # Default Ollama URL
    temperature=0.7
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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


# ==================== RAG IMPLEMENTATION ====================

# Convert knowledge bases to documents
def create_documents_from_kb():
    """Convert all knowledge bases into Document objects for RAG."""
    documents = []
    
    # Stock Market documents
    for symbol, data in STOCK_MARKET_KB.items():
        content = f"Stock: {symbol}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "stock_market", "symbol": symbol, "category": "stocks"}
        )
        documents.append(doc)
    
    # Property Market documents
    for location, data in PROPERTY_MARKET_KB.items():
        content = f"Location: {location}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    content += f"{key}:\n"
                    for sub_key, sub_value in value.items():
                        content += f"  {sub_key}: {sub_value}\n"
                else:
                    content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "property_market", "location": location, "category": "property"}
        )
        documents.append(doc)
    
    # Gold Price documents
    for gold_type, data in GOLD_PRICE_KB.items():
        content = f"Gold Type: {gold_type}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "gold_prices", "gold_type": gold_type, "category": "gold"}
        )
        documents.append(doc)
    
    # Car Market documents
    for car_model, data in CAR_MARKET_KB.items():
        content = f"Car Model: {car_model}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    content += f"{key}:\n"
                    for sub_key, sub_value in value.items():
                        content += f"  {sub_key}: {sub_value}\n"
                else:
                    content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "car_market", "car_model": car_model, "category": "cars"}
        )
        documents.append(doc)
    
    return documents

# Create documents from knowledge bases
all_documents = create_documents_from_kb() 

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(all_documents)

# Create vector store with embeddings
# Reference: https://docs.langchain.com/oss/python/langchain/rag#build-a-rag-agent-with-langchain
vector_store = FAISS.from_documents(documents=all_splits, embedding=embeddings)

# Create retrieval tool for RAG
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant information from the investment knowledge base to help answer queries about stocks, property, gold, and cars."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'unknown')} | Category: {doc.metadata.get('category', 'unknown')}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Define system prompt for the portfolio manager agent
SYSTEM_PROMPT = """You are a Senior Portfolio Manager and Investment Advisor with deep expertise across multiple asset classes.

You have access to a retrieval tool that can search through a comprehensive knowledge base containing information about:
- Pakistani stock market (HBL, UBL, OGDC, PSO, KSE-100)
- Property market (Karachi DHA, Lahore DHA, Islamabad F-7)
- Gold prices (24k, 22k, 21k gold)
- Car market (Toyota Corolla, Honda Civic, Suzuki Alto, Toyota Camry)

Use the retrieve_context tool to gather relevant market information before making recommendations. The tool will automatically find the most relevant information based on the user's query.

You are advising an individual investor with a budget of 5,000,000 PKR.

When analyzing investment options, consider:
- Current market price/value
- Expected returns and growth potential
- Risk factors
- Liquidity
- Market trends
- Fit within the 5M PKR budget

Provide comprehensive analysis and clear recommendations based on the retrieved information."""

# Create LangChain agent with RAG retrieval tool
# Reference: https://docs.langchain.com/oss/python/langchain/rag#build-a-rag-agent-with-langchain
portfolio_manager_agent = create_agent(
    model=llm_model,
    tools=[retrieve_context],
    system_prompt=SYSTEM_PROMPT,
)

# Run the agent
response = portfolio_manager_agent.invoke(
    {"messages": [{"role": "user", "content": "Provide investment recommendations for a budget of 5,000,000 PKR."}]}
)

# Extract and print AIMessage from response
# Get the last AIMessage which contains the final response
if "messages" in response:
    # Find all AIMessages and get the last one (which has the final response)
    ai_messages = [msg for msg in response["messages"] if isinstance(msg, AIMessage)]
    if ai_messages:
        # Get the last AIMessage which should have the actual content
        final_ai_message = ai_messages[-1]
        if final_ai_message.content:
            print(final_ai_message.content)
        else:
            print("AIMessage found but content is empty")
            print(final_ai_message)
    else:
        # If no AIMessage found, print the last message
        if response["messages"]:
            print(response["messages"][-1])
        else:
            print(response)
else:
    print(response)