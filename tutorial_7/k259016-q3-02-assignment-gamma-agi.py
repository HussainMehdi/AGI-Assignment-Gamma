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

# Flood Impact Data Knowledge Base (Latest 2024-2025 Data)
FLOOD_IMPACT_KB = {
    "overall_statistics": {
        "total_deaths": 1006,
        "children_deaths": 275,
        "total_displaced": 3000000,
        "evacuation_centers": 150000,
        "total_affected": 6000000,
        "economic_losses_pkr": "822 billion PKR",
        "economic_losses_usd": "2.9 billion USD",
        "agriculture_losses": "430 billion PKR",
        "infrastructure_damage": "307 billion PKR",
        "houses_damaged": 12500,
        "bridges_destroyed": 239,
        "schools_impacted": 437
    },
    "Punjab": {
        "province": "Punjab",
        "people_affected": 5100000,
        "people_evacuated": 1900000,
        "deaths": 260,
        "children_deaths": 98,
        "affected_rivers": ["Chenab", "Ravi", "Sutlej"],
        "severity": "critical",
        "priority_areas": ["Multan", "Lahore outskirts", "Sialkot", "Gujranwala"],
        "rehabilitation_status": "ongoing",
        "immediate_needs": ["shelter", "clean_water", "medical_supplies", "food_rations"]
    },
    "Khyber_Pakhtunkhwa": {
        "province": "Khyber Pakhtunkhwa (KP)",
        "people_affected": 1570000,
        "people_urgent_need": 604000,
        "deaths": 437,
        "schools_impacted": 437,
        "severity": "critical",
        "priority_areas": ["Swat", "Dir", "Chitral", "Peshawar", "Mardan"],
        "rehabilitation_status": "urgent",
        "immediate_needs": ["emergency_shelter", "road_repair", "school_reconstruction", "healthcare_facilities"]
    },
    "Sindh": {
        "province": "Sindh",
        "people_affected": 180000,
        "severity": "moderate",
        "priority_areas": ["Karachi outskirts", "Hyderabad", "Sukkur"],
        "rehabilitation_status": "in_progress",
        "immediate_needs": ["drainage_systems", "housing_reconstruction", "agricultural_recovery"]
    },
    "Balochistan": {
        "province": "Balochistan",
        "deaths": 30,
        "houses_fully_destroyed": 3006,
        "severity": "moderate",
        "priority_areas": ["Quetta", "Sibi", "Naseerabad"],
        "rehabilitation_status": "ongoing",
        "immediate_needs": ["housing_reconstruction", "water_supply", "livestock_recovery"]
    },
    "Azad_Kashmir": {
        "province": "Azad Kashmir",
        "deaths": 38,
        "houses_fully_destroyed": 2078,
        "severity": "moderate",
        "priority_areas": ["Muzaffarabad", "Mirpur", "Kotli"],
        "rehabilitation_status": "ongoing",
        "immediate_needs": ["housing_reconstruction", "road_repair", "landslide_prevention"]
    },
    "Gilgit_Baltistan": {
        "province": "Gilgit-Baltistan",
        "deaths": 41,
        "houses_fully_destroyed": 481,
        "severity": "moderate",
        "priority_areas": ["Gilgit", "Skardu", "Hunza"],
        "rehabilitation_status": "ongoing",
        "immediate_needs": ["housing_reconstruction", "bridge_repair", "mountain_road_restoration"]
    }
}

# Rehabilitation Activities Knowledge Base
REHABILITATION_ACTIVITIES_KB = {
    "housing_reconstruction": {
        "priority": "high",
        "estimated_cost_per_unit": "500000-800000 PKR",
        "total_units_needed": 15000,
        "timeline": "6-12 months",
        "responsible_agencies": ["NDMA", "PDMA", "UNHCR", "Red Crescent"],
        "status": "planning_phase",
        "challenges": ["funding_shortage", "material_availability", "skilled_labor"]
    },
    "infrastructure_repair": {
        "bridges": {
            "destroyed": 239,
            "repair_cost_per_bridge": "50-200 million PKR",
            "priority": "critical",
            "responsible_agencies": ["NHA", "Provincial PWD", "Army Engineers"]
        },
        "roads": {
            "damaged_km": 5000,
            "repair_cost_per_km": "5-15 million PKR",
            "priority": "high",
            "responsible_agencies": ["NHA", "Provincial PWD"]
        },
        "schools": {
            "impacted": 437,
            "reconstruction_cost_per_school": "10-30 million PKR",
            "priority": "high",
            "responsible_agencies": ["Education Department", "UNICEF", "NGOs"]
        }
    },
    "healthcare_services": {
        "mobile_health_units": 150,
        "temporary_clinics": 75,
        "medical_supplies_cost": "500 million PKR",
        "priority": "critical",
        "responsible_agencies": ["Health Department", "WHO", "UNICEF", "Red Crescent"],
        "focus_areas": ["disease_prevention", "maternal_health", "child_health", "mental_health"]
    },
    "water_sanitation": {
        "water_filtration_units": 500,
        "latrine_construction": 10000,
        "cost": "300 million PKR",
        "priority": "critical",
        "responsible_agencies": ["WASA", "UNICEF", "WaterAid"],
        "focus_areas": ["clean_water_access", "sanitation_facilities", "hygiene_education"]
    },
    "agricultural_recovery": {
        "total_loss": "430 billion PKR",
        "farmers_affected": 2000000,
        "crop_replacement_cost": "200 billion PKR",
        "livestock_recovery": "50 billion PKR",
        "priority": "high",
        "responsible_agencies": ["Agriculture Department", "FAO", "World Bank"],
        "support_programs": ["seed_distribution", "livestock_replacement", "irrigation_repair", "financial_assistance"]
    },
    "livelihood_restoration": {
        "employment_programs": {
            "cash_for_work": "500000 beneficiaries",
            "skill_training": "200000 people",
            "microfinance": "100000 loans",
            "cost": "150 billion PKR"
        },
        "priority": "high",
        "responsible_agencies": ["BISP", "Punjab Skills Development", "KP TEVTA", "NGOs"],
        "timeline": "12-24 months"
    }
}

# Government Agencies and Stakeholders Knowledge Base
AGENCIES_STAKEHOLDERS_KB = {
    "NDMA": {
        "name": "National Disaster Management Authority",
        "role": "coordination_and_oversight",
        "responsibilities": ["national_coordination", "resource_mobilization", "international_aid_coordination"],
        "contact": "ndma.gov.pk"
    },
    "PDMA": {
        "name": "Provincial Disaster Management Authority",
        "role": "provincial_coordination",
        "responsibilities": ["provincial_response", "local_coordination", "resource_distribution"],
        "provinces": ["Punjab", "Sindh", "KP", "Balochistan"]
    },
    "UN_agencies": {
        "UNICEF": {
            "focus": ["child_protection", "education", "health", "water_sanitation"],
            "budget_allocation": "50 million USD"
        },
        "UNHCR": {
            "focus": ["shelter", "displacement", "protection"],
            "budget_allocation": "30 million USD"
        },
        "WHO": {
            "focus": ["healthcare", "disease_prevention", "medical_supplies"],
            "budget_allocation": "25 million USD"
        },
        "FAO": {
            "focus": ["agricultural_recovery", "livestock", "food_security"],
            "budget_allocation": "40 million USD"
        }
    },
    "NGOs": {
        "Red_Crescent": {
            "focus": ["emergency_relief", "shelter", "healthcare"],
            "coverage": "nationwide"
        },
        "Edhi_Foundation": {
            "focus": ["emergency_response", "medical_services", "shelter"],
            "coverage": "nationwide"
        },
        "WaterAid": {
            "focus": ["water_sanitation", "hygiene"],
            "coverage": "flood_affected_areas"
        }
    },
    "international_donors": {
        "World_Bank": {
            "commitment": "500 million USD",
            "focus": ["infrastructure", "agriculture", "housing"]
        },
        "ADB": {
            "commitment": "300 million USD",
            "focus": ["infrastructure", "disaster_resilience"]
        },
        "USAID": {
            "commitment": "100 million USD",
            "focus": ["healthcare", "education", "livelihoods"]
        }
    }
}


# ==================== RAG IMPLEMENTATION ====================

# Convert knowledge bases to documents
def create_documents_from_kb():
    """Convert all knowledge bases into Document objects for RAG."""
    documents = []
    
    # Flood Impact documents
    for region, data in FLOOD_IMPACT_KB.items():
        content = f"Region/Area: {region}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    content += f"{key}: {', '.join(map(str, value))}\n"
                else:
                    content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "flood_impact", "region": region, "category": "flood_data"}
        )
        documents.append(doc)
    
    # Rehabilitation Activities documents
    for activity, data in REHABILITATION_ACTIVITIES_KB.items():
        content = f"Activity: {activity}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    content += f"{key}:\n"
                    for sub_key, sub_value in value.items():
                        content += f"  {sub_key}: {sub_value}\n"
                elif isinstance(value, list):
                    content += f"{key}: {', '.join(map(str, value))}\n"
                else:
                    content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "rehabilitation_activities", "activity": activity, "category": "rehabilitation"}
        )
        documents.append(doc)
    
    # Agencies and Stakeholders documents
    for agency_type, data in AGENCIES_STAKEHOLDERS_KB.items():
        content = f"Agency Type: {agency_type}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    content += f"{key}:\n"
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list):
                            content += f"  {sub_key}: {', '.join(map(str, sub_value))}\n"
                        else:
                            content += f"  {sub_key}: {sub_value}\n"
                elif isinstance(value, list):
                    content += f"{key}: {', '.join(map(str, value))}\n"
                else:
                    content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "agencies_stakeholders", "agency_type": agency_type, "category": "coordination"}
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
    """Retrieve relevant information from the flood rehabilitation knowledge base to help answer queries about flood impact, rehabilitation activities, and coordination between government agencies and stakeholders."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'unknown')} | Category: {doc.metadata.get('category', 'unknown')}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Define system prompt for the flood rehabilitation coordination agent
SYSTEM_PROMPT = """You are a Senior Disaster Management Coordinator and Rehabilitation Specialist working with the Pakistani government, agencies, and stakeholders to coordinate post-flood rehabilitation activities.

You have access to a retrieval tool that can search through a comprehensive knowledge base containing information about:
- Flood impact data across provinces (Punjab, Khyber Pakhtunkhwa, Sindh, Balochistan, Azad Kashmir, Gilgit-Baltistan)
- Rehabilitation activities (housing reconstruction, infrastructure repair, healthcare services, water sanitation, agricultural recovery, livelihood restoration)
- Government agencies and stakeholders (NDMA, PDMA, UN agencies, NGOs, international donors)

Use the retrieve_context tool to gather relevant information before providing recommendations. The tool will automatically find the most relevant information based on the user's query.

Your role is to assist in coordinating post-flood rehabilitation efforts by:
- Analyzing flood impact data and identifying priority areas
- Recommending appropriate rehabilitation activities based on regional needs
- Coordinating between government agencies, UN agencies, NGOs, and international donors
- Providing strategic guidance on resource allocation and timeline planning
- Identifying gaps in current rehabilitation efforts

When analyzing rehabilitation needs, consider:
- Severity of flood impact in each region
- Immediate needs (shelter, water, healthcare, food)
- Long-term recovery requirements (housing, infrastructure, agriculture, livelihoods)
- Available resources and funding from various agencies
- Coordination mechanisms between different stakeholders
- Timeline and priority of activities

Provide comprehensive analysis and clear recommendations based on the retrieved information to support effective post-flood rehabilitation coordination."""

# Create LangChain agent with RAG retrieval tool
# Reference: https://docs.langchain.com/oss/python/langchain/rag#build-a-rag-agent-with-langchain
flood_rehabilitation_agent = create_agent(
    model=llm_model,
    tools=[retrieve_context],
    system_prompt=SYSTEM_PROMPT,
)

# Run the agent
response = flood_rehabilitation_agent.invoke(
    {"messages": [{"role": "user", "content": "Provide a comprehensive rehabilitation plan for Punjab province, including priority activities, required resources, and coordination between agencies."}]}
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