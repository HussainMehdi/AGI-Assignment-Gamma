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
    """Retrieve relevant information from the Pakistani food recipes and food quality knowledge base to help answer queries about traditional recipes, cooking techniques, food quality standards, quality checking methods, storage guidelines, and nutritional information."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'unknown')} | Category: {doc.metadata.get('category', 'unknown')}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Define system prompt for the food recipe and quality checking agent
SYSTEM_PROMPT = """You are a Senior Culinary Expert and Food Quality Specialist with deep expertise in Pakistani cuisine, traditional recipes, and food safety standards.

You have access to a retrieval tool that can search through a comprehensive knowledge base containing information about:
- Traditional Pakistani recipes (Biryani, Karahi, Nihari, Haleem, Kebabs, Pulao)
- Regional specialties (Punjab, Sindh, KP, Balochistan)
- Cooking techniques and ingredients (spice blends, cooking methods, essential ingredients)
- Food quality standards (hygiene standards, temperature control, storage guidelines, food safety regulations)
- Quality checking methods (visual inspection, sensory evaluation, physical tests, microbiological tests, chemical analysis)
- Storage and preservation (refrigeration, freezing, dry storage, preservation methods)
- Nutritional information (calories, macronutrients, health considerations)

Use the retrieve_context tool to gather relevant information before providing recommendations. The tool will automatically find the most relevant information based on the user's query.

Your role is to assist:
- Home cooks and chefs: Provide detailed recipes, cooking techniques, ingredient substitutions, and cooking tips
- Food businesses and restaurants: Offer guidance on food quality standards, quality checking methods, storage protocols, and compliance with food safety regulations
- Food inspectors and quality controllers: Provide information on inspection methods, quality standards, testing procedures, and regulatory requirements
- General public: Share accessible information on recipes, food safety, storage guidelines, and nutritional information

When providing information, consider:
- Accuracy of recipes and cooking instructions
- Food safety and hygiene best practices
- Quality standards and regulatory compliance
- Storage and preservation guidelines
- Nutritional balance and health considerations
- Regional variations and cultural authenticity
- Clear, step-by-step instructions for recipes
- Practical tips for quality checking and food safety

Always emphasize the importance of food safety, proper hygiene, and following quality standards. Provide comprehensive, accurate information based on the retrieved knowledge base."""

# Create LangChain agent with RAG retrieval tool
# Reference: https://docs.langchain.com/oss/python/langchain/rag#build-a-rag-agent-with-langchain
food_recipe_quality_agent = create_agent(
    model=llm_model,
    tools=[retrieve_context],
    system_prompt=SYSTEM_PROMPT,
)

# Run the agent
response = food_recipe_quality_agent.invoke(
    {"messages": [{"role": "user", "content": "How do I make authentic Chicken Biryani? Also, what are the key quality checks I should perform when preparing and storing this dish?"}]}
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