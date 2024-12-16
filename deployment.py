import streamlit as st
from allergendetection import predict_allergy_with_type  # Import the prediction function from naive_bayes.py
from mealsuggestion import suggest_meal
import pandas as pd

# Title of the Streamlit App
st.title("Allergy Detection and Meal Suggestion System")

# Section 1: Display Dataset
st.subheader("Dataset Viewer")
dataset_path = 'dataset/final_allergen_detection_dataset.csv'  # Path to your dataset
df = pd.read_csv(dataset_path)

if st.checkbox("Show Dataset"):
    st.dataframe(df)

# User input: Do you have a known allergy?
has_known_allergy = st.radio("Do you have a known allergy?", ["Yes", "No"])

# Logic for users with known allergies
if has_known_allergy == "Yes":
    known_allergen = st.text_input("Enter your known allergen (e.g., peanuts, milk):").lower().strip()
    ingredients = st.text_area("Enter the recipe ingredients (comma-separated):").lower()

    if st.button("Check Recipe"):
        if ingredients:
            allergy_status, potential_allergens, allergy_types = predict_allergy_with_type(ingredients)

            if known_allergen in (potential_allergens or ""):
                st.error(f"Warning: The recipe contains your known allergen: {known_allergen}.")
                
                # Trigger meal suggestion
                st.subheader("Meal Suggestions (Allergen-Free Alternatives):")
                suggestions = suggest_meal(ingredients, known_allergen)
                for suggestion in suggestions:
                    st.write(f"**Recipe**: {suggestion['Recipe']}")
                    st.write(f"**Ingredients**: {suggestion['Ingredients']}")
                    st.write("---")
            else:
                st.success(f"The recipe does not contain your known allergen: {known_allergen}.")
                if potential_allergens:
                    st.warning(f"Other detected allergens: {potential_allergens}")
                    st.info(f"Allergy Types: {allergy_types}")
                else:
                    st.success("No other allergens detected in the recipe.")
        else:
            st.warning("Please enter the recipe ingredients.")

# Logic for users without known allergies
elif has_known_allergy == "No":
    st.info("We recommend consulting a healthcare professional for personalized advice.")
    ingredients = st.text_area("Enter the recipe ingredients (comma-separated):").lower()

    if st.button("Check Recipe"):
        if ingredients:
            allergy_status, potential_allergens, allergy_types = predict_allergy_with_type(ingredients)

            st.write(f"Allergy?: {allergy_status}")
            if potential_allergens:
                st.warning(f"Detected allergens: {potential_allergens}")
                st.info(f"Allergy Types: {allergy_types}")
            else:
                st.success("No allergens detected in the recipe.")
        else:
            st.warning("Please enter the recipe ingredients.")