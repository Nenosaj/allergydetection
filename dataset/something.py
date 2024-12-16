import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data
nltk.download('wordnet')
nltk.download('stopwords')

# Load the dataset
dataset_path = 'dataset/final_allergen_detection_dataset.csv'
df = pd.read_csv(dataset_path)

# Initialize tools for cleaning
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define synonym replacements
synonym_map = {
    "soya sauce": "soy sauce",
    "unsalted butter": "butter",
    "groundnut oil": "peanut oil",
    # Add more as needed
}

# Cleaning function
def clean_ingredients(ingredient_text):
    # 1. Convert to lowercase
    text = ingredient_text.lower()
    # 2. Replace synonyms
    for synonym, replacement in synonym_map.items():
        text = text.replace(synonym, replacement)
    # 3. Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # 4. Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # 5. Lemmatize words
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Apply cleaning to the Ingredients column
df['Cleaned_Ingredients'] = df['Ingredients'].apply(clean_ingredients)

# Save the cleaned dataset
output_path = 'dataset/cleaned_final_dataset.csv'
df.to_csv(output_path, index=False)

print(f"Cleaned dataset saved to {output_path}")
