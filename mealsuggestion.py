import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Ensure required NLTK data is downloaded
nltk.download('wordnet')
nltk.download('stopwords')

# Load the dataset
dataset_path = 'dataset/final_allergen_detection_dataset.csv'
df = pd.read_csv(dataset_path)

# Clean and normalize text data
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = text.lower()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Apply text cleaning to the dataset
df['Cleaned_Ingredients'] = df['Ingredients'].apply(clean_text)

# Prepare data for KNN classification
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Cleaned_Ingredients'])
y = df['Allergy?']  # Binary labels: 'Yes' or 'No'

# Train KNN Classifier for classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Classification Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(report)

# Call the classification function to evaluate the model

# Train NearestNeighbors for meal suggestion (unsupervised)
meal_knn = NearestNeighbors(n_neighbors=5, metric='cosine')
meal_knn.fit(X)

def suggest_meal(ingredients, known_allergen):
    """
    Suggest meals by finding the nearest recipes to the input ingredients.
    Excludes recipes that contain the known allergen.
    """
    # Clean the input ingredients
    cleaned_ingredients = clean_text(ingredients)
    ingredients_vectorized = vectorizer.transform([cleaned_ingredients])

    # Find nearest recipes
    distances, indices = meal_knn.kneighbors(ingredients_vectorized)

    suggestions = []
    for idx in indices[0]:
        recipe_name = df.loc[idx, 'Recipe Name']
        recipe_ingredients = df.loc[idx, 'Ingredients'].lower()

        # Exclude recipes containing the known allergen
        if known_allergen in recipe_ingredients:
            continue  # Skip recipes with the allergen

        # Replace known allergen in suggested recipes (just in case)
        safe_ingredients = recipe_ingredients.replace(known_allergen, "[Safe Alternative]")
        
        suggestions.append({
            "Recipe": recipe_name,
            "Ingredients": safe_ingredients
        })

    return suggestions

# Example usage
print("\nMeal Suggestions:")
known_allergen = "peanut"
ingredients_input = "chicken, oil, soy sauce, peanut"
meal_suggestions = suggest_meal(ingredients_input, known_allergen)

for suggestion in meal_suggestions:
    print(f"Recipe: {suggestion['Recipe']}")
    print(f"Ingredients: {suggestion['Ingredients']}")
    print("-" * 40)
