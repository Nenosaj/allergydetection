import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
dataset_path = 'dataset/final_allergen_detection_dataset.csv'  # Replace with your dataset path
df = pd.read_csv(dataset_path)

# Prepare the data
X = df['Ingredients']  # Input features (ingredients)
y = df['Allergy?']     # Target label (Yes/No)

# Manual allergen-to-type mapping
allergen_type_mapping = {
    "soy": "Soy Allergy",
    "peanut": "Peanut Allergy",
    "almond": "Nut Allergy",
    "wheat": "Gluten Allergy",
    "milk": "Milk Allergy / Lactose Intolerance",
    "shellfish": "Shellfish Allergy",
    "fish": "Fish Allergy",
    "egg": "Egg Allergy",
    "sesame": "Seed Allergy",
    "walnut": "Nut Allergy",
    "cashew": "Nut Allergy",
    "hazelnut": "Nut Allergy",
    "shrimp": "Shellfish Allergy",
    "lobster": "Shellfish Allergy",
    "crab": "Shellfish Allergy",
    "butter": "Milk Allergy / Lactose Intolerance",
    "cheese": "Milk Allergy / Lactose Intolerance",
    "yogurt": "Milk Allergy / Lactose Intolerance",
    "corn": "Corn Allergy",
    "barley": "Gluten Allergy",
    "nut": "Nut Allergy",
    "coconut": "Seed Allergy",
    "soybean": "Soy Allergy",
    "mustard": "Mustard Allergy",
    "celery": "Hypersensitivity",
    "apple": "Oral Allergy Syndrome",
    "kiwi": "Oral Allergy Syndrome",
    "banana": "Banana Allergy",
    "avocado": "Oral Allergy Syndrome",
    "garlic": "Allium Allergy",
    "onion": "Allium Allergy",
    "carrot": "Hypersensitivity",
    "casein": "Milk Allergy / Lactose Intolerance",
    "cattle": "Alpha-gal Syndrome",
    "cauliflower": "Cruciferous Allergy",
    "brussels sprouts": "Cruciferous Allergy",
    "broccoli": "Broccoli Allergy",
    "cucumber": "Unknown",
    "cream": "Milk Allergy / Lactose Intolerance",
    "eggplant": "Nightshade Allergy",
    "endive": "Insulin Allergy",
    "fructose": "Sugar Allergy / Intolerance",
    "ginkgo nut": "Nut Allergy",
    "horseradish": "Cruciferous Allergy",
    "grape": "LTP Allergy",
    "grapefruit": "Citrus Allergy",
    "honey": "Honey Allergy",
    "lettuce": "LTP Allergy",
    "leek": "Allium Allergy",
    "lemon": "Citrus Allergy",
    "lime": "Citrus Allergy",
    "mango": "Oral Allergy Syndrome",
    "milk powder": "Milk Allergy / Lactose Intolerance",
    "mushroom": "Mushroom Allergy",
    "okra": "Histamine Allergy",
    "olive oil": "Unknown",
    "orange": "Citrus Allergy",
    "papaya": "Oral Allergy Syndrome",
    "parsley": "Hypersensitivity",
    "parsnip": "Hypersensitivity",
    "peach": "Stone Fruit Allergy",
    "pecan": "Nut Allergy",
    "pistachio": "Nut Allergy",
    "pineapple": "Oral Allergy Syndrome",
    "potato": "Potato Allergy",
    "quince": "Oral Allergy Syndrome",
    "raspberry": "Salicylate Allergy",
    "rice": "Rice Allergy",
    "rye": "Gluten Allergy",
    "salmon": "Fish Allergy",
    "sardine": "Fish Allergy",
    "shallot": "Allium Allergy",
    "spinach": "Histamine Allergy",
    "strawberry": "Salicylate Allergy",
    "sugar": "Sugar Allergy / Intolerance",
    "squid": "Shellfish Allergy",
    "sweet potato": "Potato Allergy",
    "tomato": "Nightshade Allergy",
    "trout": "Fish Allergy",
    "tuna": "Fish Allergy",
    "turkey": "Poultry Allergy",
    "walnut": "Nut Allergy",
    "watermelon": "Unknown",
    "wheat flour": "Gluten Allergy",
    "white bean": "Legume Allergy",
    "yam": "Potato Allergy",
    "yogurt": "Milk Allergy / Lactose Intolerance",
    "zucchini": "Unknown",
    "anchovy": "Fish Allergy",
    "brie": "Milk Allergy / Lactose Intolerance",
    "croutons": "Gluten Allergy",
    "tahini": "Seed Allergy",
    "almond milk": "Nut Allergy",
    "chickpeas": "Legume Allergy",
    "lentils": "Legume Allergy",
    "tofu": "Soy Allergy",
    "miso": "Soy Allergy",
    "peanut butter": "Peanut Allergy",
    "prawns": "Shellfish Allergy",
    "lobster bisque": "Shellfish Allergy",
    "parmesan": "Milk Allergy / Lactose Intolerance",
    "basil": "Hypersensitivity",
    "cream cheese": "Milk Allergy / Lactose Intolerance",
    "buttermilk": "Milk Allergy / Lactose Intolerance",
    "paneer": "Milk Allergy / Lactose Intolerance",
    "sour cream": "Milk Allergy / Lactose Intolerance",
    "margarine": "Milk Allergy / Lactose Intolerance",
    "soy milk": "Soy Allergy",
    "oats": "Gluten Allergy",
    "buckwheat": "Gluten Allergy",
    "clam": "Shellfish Allergy",
    "lobster": "Shellfish Allergy",
}

# Preprocess the data: Vectorize the Ingredients column
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB(alpha=2.5)  # Adjust alpha for smoothing
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Allergen Detection System: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Function to predict allergy status and type
def predict_allergy_with_type(ingredients):
    """
    Predict whether a recipe contains allergens and identify the type.
    """
    ingredients_vectorized = vectorizer.transform([ingredients])
    prediction = model.predict(ingredients_vectorized)[0]

    if prediction == 'Yes':
        # Match ingredients to allergens in the mapping
        detected_allergens = [
            allergen for allergen in allergen_type_mapping.keys()
            if allergen in ingredients.lower()
        ]
        allergy_types = [allergen_type_mapping[allergen] for allergen in detected_allergens]
        return prediction, ', '.join(detected_allergens), ', '.join(allergy_types)
    else:
        return prediction, None, None

# Example usage
new_ingredients = "chicken, butter, soy sauce"
allergy_status, potential_allergens, allergy_type = predict_allergy_with_type(new_ingredients)

print(f"\nPrediction for '{new_ingredients}':")
print(f"Allergy?: {allergy_status}")
print(f"Potential Allergy: {potential_allergens}")
print(f"Type: {allergy_type}")