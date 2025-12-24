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

# Disease Statistics Knowledge Base (Latest 2024-2025 Data)
DISEASE_STATISTICS_KB = {
    "overall_statistics_2025": {
        "dengue_cases_total": 12000,
        "dengue_cases_official": 819,
        "malaria_cases_total": 215000,
        "malaria_cases_sindh": 65000,
        "malaria_cases_kp": 54000,
        "peak_season": "monsoon_months",
        "mortality_rate_dengue": "0.5-1%",
        "mortality_rate_malaria": "0.3-0.5%",
        "high_risk_areas": ["urban_centers", "stagnant_water_areas", "poor_sanitation_zones"]
    },
    "Sindh": {
        "province": "Sindh",
        "dengue_cases": 819,
        "dengue_cases_hospital_data": 12000,
        "malaria_cases": 65000,
        "karachi_division": {
            "dengue_cases": 85,
            "severity": "high",
            "hotspots": ["Gulshan-e-Iqbal", "Malir", "Korangi"]
        },
        "hyderabad_division": {
            "dengue_cases": 48,
            "malaria_cases": 2325,
            "severity": "moderate"
        },
        "mirpurkhas_division": {
            "dengue_cases": 37,
            "malaria_cases": 618,
            "severity": "moderate"
        },
        "sukkur_division": {
            "dengue_cases": 5,
            "malaria_cases": 488,
            "severity": "low"
        },
        "larkana_division": {
            "malaria_cases": 1294,
            "severity": "moderate"
        },
        "shaheed_benazirabad": {
            "malaria_cases": 595,
            "severity": "moderate"
        },
        "risk_factors": ["poor_drainage", "stagnant_water", "garbage_accumulation", "weak_mosquito_control"]
    },
    "Khyber_Pakhtunkhwa": {
        "province": "Khyber Pakhtunkhwa",
        "malaria_cases_9_months": 54000,
        "severity": "high",
        "high_risk_districts": ["Peshawar", "Mardan", "Swat", "Dir"],
        "risk_factors": ["mountainous_terrain", "irrigation_channels", "seasonal_rainfall"]
    },
    "Punjab": {
        "province": "Punjab",
        "dengue_cases": 2500,
        "malaria_cases": 45000,
        "severity": "moderate",
        "high_risk_cities": ["Lahore", "Faisalabad", "Multan", "Rawalpindi"],
        "risk_factors": ["urban_density", "construction_sites", "water_storage"]
    },
    "Balochistan": {
        "province": "Balochistan",
        "malaria_cases": 35000,
        "dengue_cases": 150,
        "severity": "moderate",
        "risk_factors": ["irrigation_systems", "seasonal_water_accumulation"]
    }
}

# Symptoms and Diagnosis Knowledge Base
SYMPTOMS_DIAGNOSIS_KB = {
    "dengue": {
        "disease": "Dengue Fever",
        "common_symptoms": ["high_fever_104F", "severe_headache", "pain_behind_eyes", "joint_muscle_pain", "nausea_vomiting", "rash", "mild_bleeding"],
        "severe_symptoms": ["severe_abdominal_pain", "persistent_vomiting", "rapid_breathing", "bleeding_gums", "fatigue_restlessness", "blood_in_vomit_stool"],
        "incubation_period": "4-10_days",
        "diagnostic_tests": ["NS1_antigen_test", "IgM_IgG_antibody_test", "PCR_test", "complete_blood_count"],
        "differential_diagnosis": ["malaria", "chikungunya", "typhoid", "influenza"],
        "warning_signs": ["decreasing_platelet_count", "increasing_hematocrit", "abdominal_tenderness", "clinical_fluid_accumulation"]
    },
    "malaria": {
        "disease": "Malaria",
        "common_symptoms": ["fever_chills", "sweating", "headache", "nausea_vomiting", "body_aches", "fatigue"],
        "severe_symptoms": ["severe_anemia", "cerebral_malaria", "respiratory_distress", "organ_failure", "hypoglycemia"],
        "incubation_period": "7-30_days",
        "diagnostic_tests": ["blood_smear_microscopy", "rapid_diagnostic_test_RDT", "PCR_test", "antigen_detection"],
        "parasite_types": ["Plasmodium_vivax", "Plasmodium_falciparum", "Plasmodium_malariae", "Plasmodium_ovale"],
        "differential_diagnosis": ["dengue", "typhoid", "viral_fever", "septicemia"]
    }
}

# Treatment Protocols Knowledge Base
TREATMENT_PROTOCOLS_KB = {
    "dengue_treatment": {
        "mild_to_moderate": {
            "approach": "supportive_care",
            "medications": ["paracetamol_acetaminophen", "oral_rehydration_solution"],
            "avoid": ["aspirin", "ibuprofen", "NSAIDs"],
            "monitoring": ["daily_blood_tests", "platelet_count", "hematocrit"],
            "fluid_management": "adequate_hydration",
            "duration": "5-7_days"
        },
        "severe_dengue": {
            "hospitalization": "required",
            "intravenous_fluids": "crystalloid_solutions",
            "blood_transfusion": "if_platelets_below_20000",
            "platelet_transfusion": "if_bleeding_or_platelets_below_10000",
            "monitoring": ["vital_signs", "urine_output", "hematocrit", "platelet_count"],
            "icu_care": "if_shock_or_organ_failure"
        },
        "discharge_criteria": ["afebrile_24_hours", "platelet_count_rising", "no_bleeding", "stable_vitals"]
    },
    "malaria_treatment": {
        "uncomplicated_vivax": {
            "first_line": "Chloroquine_3_days",
            "radical_cure": "Primaquine_14_days",
            "alternative": "Artemether_Lumefantrine",
            "duration": "3_days_acute_14_days_radical"
        },
        "uncomplicated_falciparum": {
            "first_line": "Artemether_Lumefantrine_3_days",
            "alternative": "Artesunate_Amodiaquine",
            "duration": "3_days"
        },
        "severe_malaria": {
            "hospitalization": "required",
            "intravenous": "Artesunate_IV",
            "monitoring": ["parasite_count", "blood_glucose", "renal_function", "hematocrit"],
            "supportive_care": ["blood_transfusion_if_needed", "anticonvulsants", "respiratory_support"]
        },
        "pregnancy_safety": {
            "safe": ["Chloroquine", "Quinine"],
            "avoid": ["Primaquine", "Tetracycline", "Doxycycline"]
        }
    },
    "medication_dosages": {
        "paracetamol": "500-1000mg_every_4-6_hours_max_4g_day",
        "chloroquine": "600mg_base_then_300mg_6_hours_then_300mg_daily_2_days",
        "artemether_lumefantrine": "4_tablets_twice_daily_3_days",
        "primaquine": "15mg_daily_14_days_vivax_45mg_single_dose_falciparum"
    }
}

# Prevention and Control Knowledge Base
PREVENTION_CONTROL_KB = {
    "personal_protection": {
        "mosquito_repellents": ["DEET_20-30%", "Picaridin", "Lemon_eucalyptus_oil"],
        "protective_clothing": ["long_sleeves", "long_pants", "light_colors"],
        "mosquito_nets": ["insecticide_treated_nets", "bed_nets", "window_screens"],
        "indoor_protection": ["mosquito_coils", "electric_vaporizers", "air_conditioning"]
    },
    "environmental_control": {
        "eliminate_breeding_sites": ["remove_stagnant_water", "cover_water_containers", "clean_gutters", "dispose_garbage"],
        "larvicidal_treatment": ["temephos_abate", "biological_larvicides", "mosquito_fish"],
        "adult_mosquito_control": ["indoor_residual_spraying", "space_spraying", "fogging"],
        "community_actions": ["cleanup_campaigns", "drainage_improvement", "public_awareness"]
    },
    "public_health_measures": {
        "surveillance": ["case_reporting", "vector_monitoring", "outbreak_detection"],
        "health_education": ["symptom_recognition", "prevention_messages", "early_seeking_care"],
        "vaccination": ["dengue_vaccine_available_limited", "malaria_vaccine_research_phase"],
        "coordination": ["health_department", "local_government", "community_organizations"]
    }
}

# Hospital Resources and Protocols Knowledge Base
HOSPITAL_RESOURCES_KB = {
    "diagnostic_capabilities": {
        "dengue_tests": ["NS1_RDT", "IgM_IgG_ELISA", "PCR_limited_availability", "CBC_automated"],
        "malaria_tests": ["blood_smear_microscopy", "RDT_rapid_tests", "PCR_reference_labs"],
        "critical_monitoring": ["platelet_counters", "hematocrit_analyzers", "blood_chemistry_analyzers"]
    },
    "treatment_facilities": {
        "isolation_wards": "dengue_dedicated_beds",
        "icu_capabilities": "severe_cases_management",
        "blood_bank": "platelet_blood_availability",
        "pharmacy_stock": ["antimalarials", "antipyretics", "iv_fluids", "supportive_medications"]
    },
    "referral_protocols": {
        "primary_care": "mild_cases_outpatient",
        "secondary_care": "moderate_cases_ward_admission",
        "tertiary_care": "severe_cases_icu_referral",
        "specialist_consultation": ["hematologist", "infectious_disease_specialist", "critical_care"]
    },
    "major_hospitals": {
        "Karachi": ["Aga_Khan_Hospital", "Jinnah_Postgraduate_Medical_Centre", "Civil_Hospital", "Liaquat_National_Hospital"],
        "Lahore": ["Mayo_Hospital", "Services_Hospital", "Shaukat_Khanum_Memorial", "Ganga_Ram_Hospital"],
        "Islamabad": ["Pakistan_Institute_of_Medical_Sciences", "Shifa_International", "Holy_Family_Hospital"],
        "Peshawar": ["Lady_Reading_Hospital", "Khyber_Teaching_Hospital", "Hayatabad_Medical_Complex"]
    }
}

# Public Health Information Knowledge Base
PUBLIC_HEALTH_KB = {
    "when_to_seek_care": {
        "immediate_medical_attention": ["high_fever_persistent", "severe_headache", "bleeding_signs", "decreased_urine", "severe_abdominal_pain"],
        "emergency_signs": ["difficulty_breathing", "severe_dehydration", "altered_consciousness", "severe_bleeding"],
        "self_care_appropriate": ["mild_fever", "mild_symptoms", "no_warning_signs"]
    },
    "myths_and_facts": {
        "myths": ["dengue_only_affects_children", "mosquitoes_only_bite_at_night", "vaccination_prevents_all_cases"],
        "facts": ["adults_equally_affected", "Aedes_mosquito_bites_during_day", "prevention_best_approach"]
    },
    "high_risk_groups": {
        "dengue": ["children", "elderly", "pregnant_women", "people_with_chronic_diseases"],
        "malaria": ["pregnant_women", "children_under_5", "immunocompromised", "travelers_to_endemic_areas"]
    },
    "seasonal_patterns": {
        "dengue_peak": "post_monsoon_september_november",
        "malaria_peak": "monsoon_july_september",
        "year_round_risk": "urban_areas_with_poor_sanitation"
    }
}


# ==================== RAG IMPLEMENTATION ====================

# Convert knowledge bases to documents
def create_documents_from_kb():
    """Convert all knowledge bases into Document objects for RAG."""
    documents = []
    
    # Disease Statistics documents
    for region, data in DISEASE_STATISTICS_KB.items():
        content = f"Region/Area: {region}\n"
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
            metadata={"source": "disease_statistics", "region": region, "category": "epidemiology"}
        )
        documents.append(doc)
    
    # Symptoms and Diagnosis documents
    for disease, data in SYMPTOMS_DIAGNOSIS_KB.items():
        content = f"Disease: {disease}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    content += f"{key}: {', '.join(map(str, value))}\n"
                else:
                    content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "symptoms_diagnosis", "disease": disease, "category": "clinical"}
        )
        documents.append(doc)
    
    # Treatment Protocols documents
    for treatment_type, data in TREATMENT_PROTOCOLS_KB.items():
        content = f"Treatment Type: {treatment_type}\n"
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
            metadata={"source": "treatment_protocols", "treatment_type": treatment_type, "category": "treatment"}
        )
        documents.append(doc)
    
    # Prevention and Control documents
    for prevention_type, data in PREVENTION_CONTROL_KB.items():
        content = f"Prevention Type: {prevention_type}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    content += f"{key}: {', '.join(map(str, value))}\n"
                else:
                    content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "prevention_control", "prevention_type": prevention_type, "category": "prevention"}
        )
        documents.append(doc)
    
    # Hospital Resources documents
    for resource_type, data in HOSPITAL_RESOURCES_KB.items():
        content = f"Resource Type: {resource_type}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    content += f"{key}: {', '.join(map(str, value))}\n"
                elif isinstance(value, dict):
                    content += f"{key}:\n"
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list):
                            content += f"  {sub_key}: {', '.join(map(str, sub_value))}\n"
                        else:
                            content += f"  {sub_key}: {sub_value}\n"
                else:
                    content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "hospital_resources", "resource_type": resource_type, "category": "healthcare"}
        )
        documents.append(doc)
    
    # Public Health documents
    for health_topic, data in PUBLIC_HEALTH_KB.items():
        content = f"Topic: {health_topic}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    content += f"{key}: {', '.join(map(str, value))}\n"
                else:
                    content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "public_health", "topic": health_topic, "category": "public_health"}
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
    """Retrieve relevant information from the dengue and malaria healthcare knowledge base to help answer queries about disease statistics, symptoms, diagnosis, treatment protocols, prevention measures, hospital resources, and public health information."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'unknown')} | Category: {doc.metadata.get('category', 'unknown')}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Define system prompt for the healthcare coordination agent
SYSTEM_PROMPT = """You are a Senior Healthcare Coordinator and Medical Information Specialist working with hospitals, doctors, and the public to address dengue and malaria disease-related issues in Pakistan.

You have access to a retrieval tool that can search through a comprehensive knowledge base containing information about:
- Disease statistics and epidemiology (provincial data, case counts, risk factors)
- Symptoms and diagnosis (common symptoms, severe symptoms, diagnostic tests, differential diagnosis)
- Treatment protocols (mild/moderate/severe cases, medication dosages, hospitalization criteria)
- Prevention and control measures (personal protection, environmental control, public health measures)
- Hospital resources and protocols (diagnostic capabilities, treatment facilities, referral protocols, major hospitals)
- Public health information (when to seek care, myths and facts, high-risk groups, seasonal patterns)

Use the retrieve_context tool to gather relevant healthcare information before providing recommendations. The tool will automatically find the most relevant information based on the user's query.

Your role is to assist:
- Hospitals: Provide information on diagnostic protocols, treatment guidelines, resource management, and referral procedures
- Doctors: Offer clinical guidance on symptoms recognition, diagnostic approaches, treatment protocols, and case management
- Public/Patients: Provide accessible information on symptoms, when to seek medical care, prevention measures, and general health education

When providing information, consider:
- Accuracy and evidence-based recommendations
- Urgency and severity of symptoms
- Appropriate level of care (self-care, primary care, hospital admission, ICU)
- Regional variations in disease patterns and healthcare resources
- Prevention strategies tailored to local conditions
- Clear, accessible language for public queries

Always emphasize the importance of seeking professional medical care for proper diagnosis and treatment. Provide comprehensive, accurate information based on the retrieved knowledge base."""

# Create LangChain agent with RAG retrieval tool
# Reference: https://docs.langchain.com/oss/python/langchain/rag#build-a-rag-agent-with-langchain
healthcare_coordination_agent = create_agent(
    model=llm_model,
    tools=[retrieve_context],
    system_prompt=SYSTEM_PROMPT,
)

# Run the agent
response = healthcare_coordination_agent.invoke(
    {"messages": [{"role": "user", "content": "I have a high fever, severe headache, and body aches. What should I do? Is this dengue or malaria? When should I seek medical attention?"}]}
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