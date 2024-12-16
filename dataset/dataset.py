import pandas as pd
import ast

# Load datasets
df_recipes = pd.read_csv('cleaned_output.csv')  # Recipes with title and ingredients
df_allergen_reference = pd.read_csv('cleaned_fooddata.csv')  # Master allergen list
df_food_allergens = pd.read_csv('cleaned_output_allergen_food.csv')  # Food allergens

# Extract necessary columns
df_recipes = df_recipes[['title', 'cleaned_ingredients']]
df_allergen_reference = df_allergen_reference[['food', 'allergy']]
df_food_allergens = df_food_allergens[['food_product', 'ingredient']]

# Normalize allergen foods for consistent matching
df_allergen_reference['food'] = df_allergen_reference['food'].str.lower()
df_allergen_reference['allergy'] = df_allergen_reference['allergy'].str.strip()  # Clean whitespace
df_food_allergens['ingredient'] = df_food_allergens['ingredient'].str.lower()

# Function to clean and extract core ingredients from list-like strings
def clean_ingredients(ingredients):
    """
    Convert list-like string to a clean list of ingredients.
    Extract core keywords (e.g., butter, chicken, salt).
    """
    try:
        # Safely evaluate the list-like string into a Python list
        ingredients_list = ast.literal_eval(ingredients)
        # Extract meaningful words (remove quantities and extra text)
        clean_words = []
        for item in ingredients_list:
            # Split each item by spaces, take relevant words (e.g., last or second-to-last word)
            words = item.split()
            if len(words) > 0:
                clean_words.append(words[-1].strip(",."))  # Get the last word, strip punctuation
        return clean_words
    except:
        return []

# Function to detect allergens in the cleaned ingredients
def find_allergens(ingredients_list, allergen_list):
    """
    Identify allergens from the master list in the cleaned ingredients.
    """
    detected_allergens = [allergen for allergen in allergen_list if allergen in ingredients_list]
    return detected_allergens  # Return as a list

# Final dataset
final_data = []

# Process recipes from `cleaned_output.csv`
for _, row in df_recipes.iterrows():
    recipe_name = row['title']
    raw_ingredients = row['cleaned_ingredients']
    
    # Clean and extract core ingredients
    cleaned_ingredients = clean_ingredients(raw_ingredients)
    
    # Match allergens using the master allergen list
    detected_allergens = find_allergens(cleaned_ingredients, df_allergen_reference['food'].tolist())
    allergy_status = 'Yes' if detected_allergens else 'No'
    potential_allergens = ', '.join(detected_allergens) if detected_allergens else None
    
    # Get the type (allergy type) for each detected allergen
    allergy_types = [
        df_allergen_reference.loc[df_allergen_reference['food'] == allergen, 'allergy'].values[0]
        for allergen in detected_allergens
    ]
    allergy_type = ', '.join(allergy_types) if allergy_types else None

    final_data.append({
        'Recipe Name': recipe_name,
        'Ingredients': ', '.join(cleaned_ingredients),  # Reconstructed cleaned ingredients
        'Allergy?': allergy_status,
        'Potential Allergy': potential_allergens,
        'Type': allergy_type
    })

# Process recipes from `cleaned_output_allergen_food.csv`
for _, row in df_food_allergens.iterrows():
    recipe_name = row['food_product']
    raw_ingredients = row['ingredient']
    
    # Split ingredients into a clean list
    ingredients_list = [word.strip() for word in raw_ingredients.split(',')]
    
    # Match allergens using the master allergen list
    detected_allergens = find_allergens(ingredients_list, df_allergen_reference['food'].tolist())
    allergy_status = 'Yes' if detected_allergens else 'No'
    potential_allergens = ', '.join(detected_allergens) if detected_allergens else None

    # Get the type (allergy type) for each detected allergen
    allergy_types = [
        df_allergen_reference.loc[df_allergen_reference['food'] == allergen, 'allergy'].values[0]
        for allergen in detected_allergens
    ]
    allergy_type = ', '.join(allergy_types) if allergy_types else None

    final_data.append({
        'Recipe Name': recipe_name,
        'Ingredients': raw_ingredients,  # Raw ingredients
        'Allergy?': allergy_status,
        'Potential Allergy': potential_allergens,
        'Type': allergy_type
    })

# Convert to DataFrame and save
df_final = pd.DataFrame(final_data)
output_file = 'final_allergen_detection_dataset.csv'
df_final.to_csv(output_file, index=False)

print(f"Dataset created and saved as {output_file}")
