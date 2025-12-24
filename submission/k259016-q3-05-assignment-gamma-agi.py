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

# Traditional Pakistani Recipes Knowledge Base
PAKISTANI_RECIPES_KB = {
    "Biryani": {
        "dish_type": "Rice_Dish",
        "origin": "Hyderabad_Deccan",
        "popular_varieties": ["Chicken_Biryani", "Beef_Biryani", "Mutton_Biryani", "Vegetable_Biryani"],
        "key_ingredients": ["basmati_rice", "meat_chicken", "yogurt", "onions", "ginger_garlic_paste", "biryani_masala", "saffron", "mint_leaves", "coriander"],
        "cooking_method": "layered_cooking_dum_style",
        "cooking_time": "60-90_minutes",
        "servings": "4-6_people",
        "difficulty": "moderate",
        "spice_level": "medium_to_high",
        "regional_variations": {
            "Sindhi_Biryani": "uses_prunes_and_potatoes",
            "Hyderabadi_Biryani": "uses_marinated_meat_and_saffron",
            "Lahore_Biryani": "uses_fried_onions_and_yogurt"
        }
    },
    "Karahi": {
        "dish_type": "Curry",
        "popular_varieties": ["Chicken_Karahi", "Mutton_Karahi", "Beef_Karahi", "Prawn_Karahi"],
        "key_ingredients": ["meat", "tomatoes", "ginger", "garlic", "green_chilies", "coriander", "karahi_masala", "ginger_strips"],
        "cooking_method": "high_heat_wok_cooking",
        "cooking_time": "30-45_minutes",
        "servings": "3-4_people",
        "difficulty": "easy_to_moderate",
        "spice_level": "medium_to_high",
        "serving_style": "served_in_karahi_pan_with_naan"
    },
    "Nihari": {
        "dish_type": "Stew",
        "origin": "Mughlai",
        "key_ingredients": ["beef_shank", "bone_marrow", "nihari_masala", "ginger_garlic", "wheat_flour", "ghee", "garam_masala"],
        "cooking_method": "slow_cooking_overnight",
        "cooking_time": "6-8_hours",
        "servings": "4-6_people",
        "difficulty": "moderate",
        "spice_level": "medium",
        "serving_style": "garnished_with_ginger_cilantro_lemon",
        "traditional_time": "breakfast_dish"
    },
    "Haleem": {
        "dish_type": "Stew",
        "key_ingredients": ["lentils", "wheat", "barley", "meat", "ginger_garlic", "spices", "ghee"],
        "cooking_method": "slow_cooking_until_mushy",
        "cooking_time": "4-6_hours",
        "servings": "6-8_people",
        "difficulty": "moderate",
        "spice_level": "medium",
        "seasonal": "popular_during_Ramadan",
        "nutritional_value": "high_protein_high_fiber"
    },
    "Kebab": {
        "dish_type": "Grilled_Meat",
        "popular_varieties": ["Seekh_Kebab", "Chapli_Kebab", "Shami_Kebab", "Reshmi_Kebab"],
        "key_ingredients": ["minced_meat", "spices", "onions", "ginger_garlic", "herbs"],
        "cooking_method": "grilling_tandoor",
        "cooking_time": "20-30_minutes",
        "servings": "4-6_people",
        "difficulty": "moderate",
        "spice_level": "medium_to_high",
        "serving_style": "with_chutney_and_salad"
    },
    "Pulao": {
        "dish_type": "Rice_Dish",
        "popular_varieties": ["Yakhni_Pulao", "Matar_Pulao", "Chicken_Pulao"],
        "key_ingredients": ["basmati_rice", "meat_broth", "whole_spices", "onions", "ginger_garlic"],
        "cooking_method": "rice_cooked_in_broth",
        "cooking_time": "45-60_minutes",
        "servings": "4-6_people",
        "difficulty": "easy_to_moderate",
        "spice_level": "mild_to_medium"
    }
}

# Regional Specialties Knowledge Base
REGIONAL_SPECIALTIES_KB = {
    "Punjab": {
        "province": "Punjab",
        "famous_dishes": ["Sarson_Da_Saag", "Makki_Di_Roti", "Butter_Chicken", "Tandoori_Chicken", "Lassi"],
        "cooking_style": "rich_and_buttery",
        "spice_level": "moderate",
        "staple_foods": ["wheat", "lentils", "dairy"]
    },
    "Sindh": {
        "province": "Sindh",
        "famous_dishes": ["Sindhi_Biryani", "Sai_Bhaji", "Dal_Pakwan", "Koki", "Thadal"],
        "cooking_style": "aromatic_and_spicy",
        "spice_level": "high",
        "staple_foods": ["rice", "fish", "vegetables"]
    },
    "Khyber_Pakhtunkhwa": {
        "province": "Khyber_Pakhtunkhwa",
        "famous_dishes": ["Chapli_Kebab", "Peshawari_Chapli", "Namkeen_Gosht", "Kabuli_Pulao", "Kahwa"],
        "cooking_style": "meat_centric_minimal_spices",
        "spice_level": "mild_to_medium",
        "staple_foods": ["meat", "bread", "rice"]
    },
    "Balochistan": {
        "province": "Balochistan",
        "famous_dishes": ["Sajji", "Kaak", "Khaddi_Kebab", "Dampukht"],
        "cooking_style": "simple_roasted_meat",
        "spice_level": "mild",
        "staple_foods": ["meat", "bread", "dates"]
    }
}

# Cooking Techniques and Ingredients Knowledge Base
COOKING_TECHNIQUES_KB = {
    "spice_blends": {
        "garam_masala": ["cinnamon", "cardamom", "cloves", "black_pepper", "cumin", "coriander"],
        "biryani_masala": ["star_anise", "mace", "nutmeg", "fennel", "bay_leaves"],
        "karahi_masala": ["cumin", "coriander", "red_chili", "turmeric", "garam_masala"],
        "nihari_masala": ["fennel", "cumin", "coriander", "turmeric", "red_chili", "ginger_powder"]
    },
    "cooking_methods": {
        "dum_cooking": "slow_cooking_in_sealed_pot",
        "tandoor_cooking": "clay_oven_high_temperature",
        "bhuna": "dry_roasting_spices_and_meat",
        "tadka": "tempering_with_whole_spices"
    },
    "essential_ingredients": {
        "spices": ["turmeric", "red_chili_powder", "cumin", "coriander", "garam_masala", "cardamom", "cinnamon"],
        "aromatics": ["onions", "ginger", "garlic", "green_chilies"],
        "herbs": ["coriander_leaves", "mint_leaves", "curry_leaves"],
        "dairy": ["yogurt", "ghee", "cream", "paneer"],
        "grains": ["basmati_rice", "wheat_flour", "lentils"]
    }
}

# Food Quality Standards Knowledge Base
FOOD_QUALITY_STANDARDS_KB = {
    "hygiene_standards": {
        "personal_hygiene": ["clean_hands", "proper_attire", "hair_covering", "no_jewelry"],
        "kitchen_hygiene": ["clean_surfaces", "separate_cutting_boards", "proper_storage", "regular_cleaning"],
        "food_handling": ["proper_temperature_control", "avoid_cross_contamination", "use_clean_utensils"]
    },
    "temperature_control": {
        "hot_food": "above_60_degrees_celsius",
        "cold_food": "below_4_degrees_celsius",
        "danger_zone": "4_to_60_degrees_celsius",
        "cooking_temperatures": {
            "poultry": "75_degrees_celsius",
            "beef_mutton": "70_degrees_celsius",
            "fish": "63_degrees_celsius"
        }
    },
    "storage_guidelines": {
        "refrigeration": "below_4_degrees_celsius",
        "freezing": "below_minus_18_degrees_celsius",
        "dry_storage": "cool_dry_place_below_21_degrees",
        "shelf_life": {
            "cooked_meat": "3-4_days_refrigerated",
            "raw_meat": "1-2_days_refrigerated",
            "rice_dishes": "1-2_days_refrigerated",
            "curries": "3-4_days_refrigerated"
        }
    },
    "food_safety_regulations": {
        "pakistan_standards": "PSQCA_Pakistan_Standards_and_Quality_Control_Authority",
        "haccp": "Hazard_Analysis_Critical_Control_Points",
        "fssa": "Food_Safety_and_Standards_Act",
        "licensing": "restaurants_require_food_license"
    }
}

# Quality Checking Methods Knowledge Base
QUALITY_CHECKING_METHODS_KB = {
    "visual_inspection": {
        "freshness_indicators": ["bright_color", "firm_texture", "no_discoloration", "no_mold"],
        "meat_quality": ["bright_red_color", "firm_texture", "no_off_odors", "minimal_fat_separation"],
        "spice_quality": ["aromatic_smell", "vibrant_color", "no_clumping", "no_insects"]
    },
    "sensory_evaluation": {
        "smell": "fresh_aromatic_no_off_odors",
        "taste": "balanced_flavors_no_rancidity",
        "texture": "appropriate_consistency_not_slimy",
        "appearance": "appealing_color_presentation"
    },
    "physical_tests": {
        "temperature_check": "use_food_thermometer",
        "ph_testing": "for_pickled_foods",
        "moisture_content": "for_dry_spices",
        "texture_analysis": "firmness_consistency"
    },
    "microbiological_tests": {
        "total_plate_count": "bacterial_load_assessment",
        "coliform_test": "fecal_contamination",
        "e_coli_test": "pathogen_detection",
        "salmonella_test": "foodborne_illness_prevention"
    },
    "chemical_analysis": {
        "pesticide_residue": "for_fruits_vegetables",
        "heavy_metals": "for_seafood",
        "additives_check": "preservatives_colors",
        "adulteration_detection": "milk_oil_spices"
    }
}

# Storage and Preservation Knowledge Base
STORAGE_PRESERVATION_KB = {
    "refrigeration": {
        "meat": "store_in_sealed_container_1-2_days",
        "cooked_food": "cool_before_refrigeration_3-4_days",
        "dairy": "store_in_original_packaging_check_expiry",
        "vegetables": "store_in_crisper_drawer_remove_plastic"
    },
    "freezing": {
        "meat": "wrap_tightly_freezer_bag_3-6_months",
        "cooked_dishes": "freeze_in_airtight_containers_2-3_months",
        "bread": "slice_before_freezing_2-3_months",
        "spices": "freeze_to_preserve_flavor_6-12_months"
    },
    "dry_storage": {
        "spices": "cool_dark_place_airtight_containers",
        "rice_lentils": "airtight_containers_cool_dry_place",
        "flour": "airtight_container_cool_dry_place",
        "oil": "cool_dark_place_away_from_light"
    },
    "preservation_methods": {
        "pickling": "use_vinegar_salt_preserve_vegetables",
        "drying": "sun_dry_or_dehydrate_fruits_vegetables",
        "canning": "sterilize_and_seal_jars",
        "fermentation": "yogurt_pickles_traditional_methods"
    }
}

# Nutritional Information Knowledge Base
NUTRITIONAL_INFO_KB = {
    "common_dishes": {
        "Biryani_serving_300g": {
            "calories": "450-550_kcal",
            "protein": "25-30g",
            "carbs": "60-70g",
            "fat": "15-20g"
        },
        "Karahi_serving_250g": {
            "calories": "350-450_kcal",
            "protein": "30-35g",
            "carbs": "10-15g",
            "fat": "20-25g"
        },
        "Nihari_serving_300g": {
            "calories": "400-500_kcal",
            "protein": "35-40g",
            "carbs": "20-25g",
            "fat": "20-25g"
        }
    },
    "health_considerations": {
        "spice_tolerance": "adjust_spice_levels_for_sensitivity",
        "dietary_restrictions": ["halal_requirements", "vegetarian_options", "gluten_free_alternatives"],
        "nutritional_balance": "combine_with_salads_yogurt_for_balanced_meal",
        "portion_control": "moderate_portions_for_healthy_eating"
    }
}

# Quality Control Department Operations Knowledge Base
QC_OPERATIONS_KB = {
    "inspection_procedures": {
        "receiving_inspection": {
            "raw_materials": ["check_expiry_dates", "verify_certificates", "visual_quality_check", "temperature_verification"],
            "packaging": ["check_integrity", "verify_labels", "check_for_damage", "verify_batch_numbers"],
            "documentation": ["verify_supplier_certificates", "check_halal_certification", "verify_origin_certificates"]
        },
        "in_process_inspection": {
            "cooking_process": ["temperature_monitoring", "cooking_time_verification", "spice_measurement_accuracy", "hygiene_compliance"],
            "preparation_area": ["surface_cleanliness", "equipment_sanitization", "cross_contamination_checks", "personal_hygiene_verification"],
            "critical_control_points": ["haccp_monitoring", "temperature_logs", "ph_measurements", "visual_quality_checks"]
        },
        "finished_product_inspection": {
            "quality_attributes": ["appearance", "texture", "taste", "aroma", "temperature"],
            "safety_checks": ["microbiological_testing", "chemical_analysis", "foreign_object_detection", "allergen_verification"],
            "packaging_checks": ["seal_integrity", "label_accuracy", "weight_verification", "expiry_date_labeling"]
        }
    },
    "quality_control_checklists": {
        "daily_checks": ["temperature_logs", "hygiene_audits", "equipment_calibration", "staff_training_verification"],
        "weekly_checks": ["deep_cleaning_verification", "supplier_audit", "waste_management_check", "pest_control_verification"],
        "monthly_checks": ["comprehensive_audit", "equipment_maintenance", "staff_certification_renewal", "regulatory_compliance_review"]
    },
    "non_conformance_management": {
        "identification": ["visual_inspection_failures", "test_result_deviations", "customer_complaints", "internal_audit_findings"],
        "documentation": ["non_conformance_report", "root_cause_analysis", "corrective_action_plan", "preventive_measures"],
        "disposition": ["reject_and_quarantine", "rework_if_possible", "dispose_if_unsafe", "traceability_records"]
    },
    "testing_protocols": {
        "microbiological": {
            "total_plate_count": "standard_plate_count_method",
            "coliform_test": "most_probable_number_method",
            "e_coli_detection": "selective_media_culture",
            "salmonella_test": "enrichment_and_isolation",
            "frequency": "daily_for_high_risk_weekly_for_low_risk"
        },
        "chemical": {
            "pesticide_residue": "chromatography_methods",
            "heavy_metals": "atomic_absorption_spectroscopy",
            "additives": "hplc_gc_methods",
            "adulteration": "specific_tests_for_milk_oil_spices",
            "frequency": "monthly_or_as_per_risk_assessment"
        },
        "physical": {
            "temperature": "calibrated_thermometers",
            "ph_level": "ph_meter_calibration",
            "moisture_content": "moisture_analyzer",
            "texture": "texture_analyzer_equipment",
            "frequency": "continuous_monitoring_for_temperature_daily_for_others"
        }
    },
    "documentation_requirements": {
        "quality_records": ["inspection_reports", "test_results", "calibration_certificates", "training_records", "audit_reports"],
        "traceability": ["batch_codes", "supplier_information", "production_dates", "distribution_records"],
        "compliance": ["psqca_certificates", "halal_certification", "food_license", "haccp_documentation"]
    },
    "regulatory_compliance": {
        "pakistan_standards": {
            "psqca": "Pakistan_Standards_and_Quality_Control_Authority",
            "requirements": ["product_standards", "labeling_requirements", "packaging_standards", "safety_standards"],
            "certification": "psqca_certification_mandatory_for_export"
        },
        "food_safety": {
            "fssa": "Food_Safety_and_Standards_Act",
            "haccp": "mandatory_for_food_businesses",
            "gmp": "Good_Manufacturing_Practices_required",
            "licensing": "food_business_license_required"
        },
        "halal_certification": {
            "authority": "Pakistan_Halal_Authority",
            "requirements": ["halal_ingredients", "halal_processing", "separation_from_non_halal", "certification_audit"],
            "renewal": "annual_certification_renewal_required"
        }
    }
}


# ==================== RAG IMPLEMENTATION ====================

# Convert knowledge bases to documents
def create_documents_from_kb():
    """Convert all knowledge bases into Document objects for RAG."""
    documents = []
    
    # Pakistani Recipes documents
    for dish, data in PAKISTANI_RECIPES_KB.items():
        content = f"Dish: {dish}\n"
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
            metadata={"source": "pakistani_recipes", "dish": dish, "category": "recipes"}
        )
        documents.append(doc)
    
    # Regional Specialties documents
    for region, data in REGIONAL_SPECIALTIES_KB.items():
        content = f"Region: {region}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    content += f"{key}: {', '.join(map(str, value))}\n"
                else:
                    content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "regional_specialties", "region": region, "category": "regional_cuisine"}
        )
        documents.append(doc)
    
    # Cooking Techniques documents
    for technique_type, data in COOKING_TECHNIQUES_KB.items():
        content = f"Technique Type: {technique_type}\n"
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
            metadata={"source": "cooking_techniques", "technique_type": technique_type, "category": "cooking"}
        )
        documents.append(doc)
    
    # Food Quality Standards documents
    for standard_type, data in FOOD_QUALITY_STANDARDS_KB.items():
        content = f"Standard Type: {standard_type}\n"
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
            metadata={"source": "food_quality_standards", "standard_type": standard_type, "category": "quality"}
        )
        documents.append(doc)
    
    # Quality Checking Methods documents
    for method_type, data in QUALITY_CHECKING_METHODS_KB.items():
        content = f"Method Type: {method_type}\n"
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
            metadata={"source": "quality_checking_methods", "method_type": method_type, "category": "quality_checking"}
        )
        documents.append(doc)
    
    # Storage and Preservation documents
    for storage_type, data in STORAGE_PRESERVATION_KB.items():
        content = f"Storage Type: {storage_type}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    content += f"{key}: {', '.join(map(str, value))}\n"
                else:
                    content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "storage_preservation", "storage_type": storage_type, "category": "preservation"}
        )
        documents.append(doc)
    
    # Nutritional Information documents
    for nutrition_type, data in NUTRITIONAL_INFO_KB.items():
        content = f"Nutrition Type: {nutrition_type}\n"
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
            metadata={"source": "nutritional_info", "nutrition_type": nutrition_type, "category": "nutrition"}
        )
        documents.append(doc)
    
    # Quality Control Operations documents
    for qc_operation, data in QC_OPERATIONS_KB.items():
        content = f"QC Operation: {qc_operation}\n"
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    content += f"{key}:\n"
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, list):
                            content += f"  {sub_key}: {', '.join(map(str, sub_value))}\n"
                        elif isinstance(sub_value, dict):
                            content += f"  {sub_key}:\n"
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                if isinstance(sub_sub_value, list):
                                    content += f"    {sub_sub_key}: {', '.join(map(str, sub_sub_value))}\n"
                                else:
                                    content += f"    {sub_sub_key}: {sub_sub_value}\n"
                        else:
                            content += f"  {sub_key}: {sub_value}\n"
                elif isinstance(value, list):
                    content += f"{key}: {', '.join(map(str, value))}\n"
                else:
                    content += f"{key}: {value}\n"
        doc = Document(
            page_content=content,
            metadata={"source": "qc_operations", "qc_operation": qc_operation, "category": "quality_control"}
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
    """Retrieve relevant information from the food quality control knowledge base to help answer queries about quality control procedures, inspection methods, testing protocols, regulatory compliance, non-conformance management, documentation requirements, and quality standards for Pakistani food products."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'unknown')} | Category: {doc.metadata.get('category', 'unknown')}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Define system prompt for the quality control department agent
SYSTEM_PROMPT = """You are a Senior Quality Control Manager and Food Safety Expert specializing in Pakistani food products. Your primary role is to assist the Quality Control Department in ensuring food safety, quality standards, and regulatory compliance.

You have access to a retrieval tool that can search through a comprehensive knowledge base containing information about:
- Quality control inspection procedures (receiving, in-process, finished product inspection)
- Quality control checklists (daily, weekly, monthly checks)
- Non-conformance management (identification, documentation, disposition)
- Testing protocols (microbiological, chemical, physical tests with frequencies)
- Documentation requirements (quality records, traceability, compliance documents)
- Regulatory compliance (PSQCA standards, Food Safety and Standards Act, HACCP, Halal certification)
- Food quality standards (hygiene standards, temperature control, storage guidelines)
- Quality checking methods (visual inspection, sensory evaluation, physical tests, microbiological tests, chemical analysis)
- Storage and preservation guidelines
- Traditional Pakistani recipes and cooking techniques (for understanding product specifications)

Use the retrieve_context tool to gather relevant information before providing recommendations. The tool will automatically find the most relevant information based on the user's query.

Your role is to assist Quality Control Department staff including:
- Quality Control Inspectors: Provide inspection procedures, checklists, testing protocols, and quality standards
- Quality Assurance Managers: Offer guidance on quality systems, compliance requirements, documentation, and audit procedures
- Laboratory Technicians: Provide testing methods, protocols, equipment calibration, and result interpretation
- Food Safety Officers: Share information on regulatory compliance, HACCP implementation, and food safety standards
- Quality Control Supervisors: Assist with non-conformance management, corrective actions, and preventive measures

When providing information, prioritize:
- Regulatory compliance and legal requirements (PSQCA, FSSA, Halal certification)
- Food safety and public health protection
- Quality standards and specifications
- Systematic inspection and testing procedures
- Proper documentation and traceability
- Non-conformance handling and corrective actions
- Risk assessment and preventive measures
- Clear, actionable procedures and checklists

Always emphasize:
- Compliance with Pakistan food safety regulations
- Importance of proper documentation and record-keeping
- Systematic approach to quality control
- Risk-based inspection and testing
- Continuous improvement and preventive actions
- Training and competency requirements

Provide comprehensive, accurate, and actionable information based on the retrieved knowledge base to support effective quality control operations."""

# Create LangChain agent with RAG retrieval tool
# Reference: https://docs.langchain.com/oss/python/langchain/rag#build-a-rag-agent-with-langchain
quality_control_agent = create_agent(
    model=llm_model,
    tools=[retrieve_context],
    system_prompt=SYSTEM_PROMPT,
)

# Run the agent
response = quality_control_agent.invoke(
    {"messages": [{"role": "user", "content": "What are the complete quality control inspection procedures for receiving raw materials for Biryani production? Include all checks, documentation requirements, and testing protocols needed."}]}
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