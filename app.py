import streamlit as st
import joblib
import pandas as pd
import re
import nltk
import numpy as np
import altair as alt
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from typing import Optional, Dict

# --- Configuration and Resource Setup ---

# The filename MUST match your uploaded file
MODEL_FILENAME = 'mental_health_predictor.joblib' 

# Define your keyword lists exactly as they were in your notebook!
RISK_KEYWORDS = [
    'sad', 'alone', 'cry', 'hopeless', 'suicide', 'depressed', 
    'anxiety', 'fear', 'lost', 'end it', 'die', 'worthless'
]
POSITIVE_KEYWORDS = [
    'happy', 'joy', 'love', 'hope', 'great', 'fun', 
    'excited', 'blessed', 'thankful', 'strong', 'better'
]
# **********************************************

# --- Global Labels Dictionary (Used for Confidence Mapping Only) ---
PREDICTION_LABELS = {
    0: "LOW_RISK",
    1: "HIGH_RISK"
}
# --- End Global Labels Dictionary ---

# --- Heuristic Category Definitions ---
CATEGORY_KEYWORDS = {
    "Suicidal": ['suicide', 'kill myself', 'end it', 'not want to live', 'die', 'jump off', 'hang myself'],
    "Depression": ['depressed', 'sad', 'empty', 'lonely', 'hopeless', 'worthless', 'mood swing', 'sleep too much', 'lack of energy'],
    "Anxiety": ['anxiety', 'nervous', 'scared', 'fear', 'panic', 'worry', 'trembling', 'heart racing', 'overthinking'],
    "Stress": ['stress', 'overwhelmed', 'pressure', 'deadline', 'workload', 'too much', 'burnt out', 'tired', 'fatigue'],
    "Bi-Polar": ['manic', 'bipolar', 'racing thoughts', 'grandeur', 'euphoria', 'impulsive', 'crash'],
    "Personality Disorder": ['unstable relationships', 'borderline', 'manipulative', 'split', 'self-harm', 'empty feeling'],
}

CATEGORY_WEIGHTS = {
    "Suicidal": 3.0,
    "Depression": 2.5,
    "Anxiety": 2.0,
    "Stress": 1.5,
    "Bi-Polar": 1.0,
    "Personality Disorder": 1.0,
}
# --- End Heuristic Category Definitions ---

# --- NLTK Resource Initialization (THE FIX) ---
@st.cache_resource
def initialize_nltk_resources():
    """
    Downloads NLTK resources and returns the necessary objects (VADER and stopwords).
    Uses Streamlit's cache to ensure this runs only once across deployments/sessions.
    """
    try:
        # Download resources required by the application
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Initialize and return objects that rely on the downloaded data
        vader_analyzer = SentimentIntensityAnalyzer()
        english_stopwords_set = set(stopwords.words('english'))
        
        return vader_analyzer, english_stopwords_set
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {e}")
        st.stop()

# Initialize VADER and STOPWORDS globally using the cached function
vader, english_stopwords = initialize_nltk_resources()


# --- Feature Extraction Functions (Replicating Notebook Logic) ---

def clean_text(text: str) -> str:
    """Replicates the text cleaning step used before training."""
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text

def count_keywords(text: str, keywords: list) -> int:
    """Counts occurrences of specific keywords in the text."""
    count = 0
    words = text.split()
    for word in words:
        if word in keywords:
            count += 1
    return count

def extract_features(raw_text: str) -> Optional[pd.DataFrame]:
    """
    Generates the 11 features (1 text + 10 numerical) required by the pipeline and heuristics.
    (Note: The model only uses 10 features, but the heuristic needs the new 'vader_neg' feature).
    """
    if not raw_text.strip():
        return None
    
    text_processed = clean_text(raw_text)
    sentiment_scores = vader.polarity_scores(raw_text) # Get all VADER scores
    blob = TextBlob(raw_text)
    
    text_length = len(raw_text)
    word_count = len(raw_text.split())
    
    features_dict = {
        'cleaned_text': text_processed,
        'text_length': text_length,
        'word_count': word_count,
        'exclamation_count': raw_text.count('!'),
        'question_count': raw_text.count('?'),
        'caps_ratio': sum(1 for c in raw_text if c.isupper()) / text_length if text_length > 0 else 0,
        'risk_keywords_count': count_keywords(text_processed, RISK_KEYWORDS),
        'positive_keywords_count': count_keywords(text_processed, POSITIVE_KEYWORDS),
        'textblob_polarity': blob.sentiment.polarity,
        'vader_compound': sentiment_scores['compound'],
        'vader_neg': sentiment_scores['neg'] # ADDED: VADER negative score (0 to 1) for heuristics
    }
    
    return pd.DataFrame([features_dict])


# --- Heuristic Scoring Function (UPDATED) ---

def calculate_heuristic_scores(text: str, vader_neg_score: float) -> Dict[str, float]:
    """
    Calculates heuristic percentage scores for detailed categories.
    UPDATED: Uses vader_neg_score for influence, ensuring negative keywords 
    always contribute significantly.
    """
    text_processed = clean_text(text)
    raw_scores = {}
    
    # 1. Calculate raw score for each pathological category
    for category, keywords in CATEGORY_KEYWORDS.items():
        keyword_count = count_keywords(text_processed, keywords)
        weight = CATEGORY_WEIGHTS[category]
        
        # Scoring Logic:
        # Base Score is driven by keyword count and category weight.
        # It is multiplied by (1 + vader_neg_score) to boost the score if the overall 
        # VADER assessment is also negative (high vader_neg_score).
        # This prevents the score from being zeroed out by a neutral/positive VADER compound score.
        score = keyword_count * weight * (1.0 + vader_neg_score) 
        
        raw_scores[category] = score

    # 2. Normalize and calculate 'Normal' percentage
    total_pathological_score = sum(raw_scores.values())
    
    if total_pathological_score > 0:
        # Define a factor to determine how much the raw score impacts the final percentage
        # Use tanh to smooth the total score impact on the 'Normal' percentage
        pathology_factor = np.tanh(total_pathological_score / 3) 
        
        normal_percentage = max(0.0, 1.0 - pathology_factor)
        remaining_percentage = 1.0 - normal_percentage
        
        final_scores = {"Normal": normal_percentage}
        
        # Allocate remaining percentage (pathology_factor) based on normalized raw scores
        if total_pathological_score > 0:
            for category, score in raw_scores.items():
                final_scores[category] = (score / total_pathological_score) * remaining_percentage
        else:
            for category in CATEGORY_KEYWORDS:
                final_scores[category] = 0.0
    else:
        # If no keywords are found, assume 100% normal
        final_scores = {"Normal": 1.0}
        for category in CATEGORY_KEYWORDS:
            final_scores[category] = 0.0

    # Convert to percentage (0.0 to 100.0) and reorder
    ordered_results = {
        "Normal": final_scores.get("Normal", 0.0) * 100,
        "Depression": final_scores.get("Depression", 0.0) * 100,
        "Suicidal": final_scores.get("Suicidal", 0.0) * 100,
        "Anxiety": final_scores.get("Anxiety", 0.0) * 100,
        "Stress": final_scores.get("Stress", 0.0) * 100,
        "Bi-Polar": final_scores.get("Bi-Polar", 0.0) * 100,
        "Personality Disorder": final_scores.get("Personality Disorder", 0.0) * 100,
    }

    # Final adjustment to ensure sum is exactly 100.0% (due to floating point arithmetic)
    current_sum = sum(ordered_results.values())
    if current_sum > 0:
        adjustment_factor = 100.0 / current_sum
        for k in ordered_results:
            ordered_results[k] *= adjustment_factor

    return ordered_results


# --- Model Loading and Prediction ---

@st.cache_resource 
def load_predictor():
    """Loads the model pipeline from the joblib file."""
    try:
        # NOTE: Using the correct filename 'mental_health_predictor (1).joblib'
        model = joblib.load(MODEL_FILENAME) 
        st.success("Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_FILENAME}' not found. Please ensure it is in your repository.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

predictor_pipeline = load_predictor()

# --- STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(page_title="Mental Health Risk Predictor", page_icon="🧠", layout="centered")
    st.title('🧠 Social Media Mental Health Risk Predictor')
    
    # Explanation for the user about the two different outputs
    st.markdown("""
        <p style='color:red;'>
        <strong>IMPORTANT NOTE:</strong> The underlying Machine Learning model is a <strong>binary classifier</strong> (High/Low Risk). 
        The primary label and the percentages shown below are a <strong>heuristic interpretation</strong> based on keywords and sentiment to provide detailed categories, as requested.
        </p>
    """, unsafe_allow_html=True)
    st.markdown('---')
    
    # Ensure model is loaded before proceeding
    if predictor_pipeline is None:
        st.stop()

    st.markdown('### Enter a Social Media Post for Analysis')
    
    user_input = st.text_area(
        "Paste the text here:", 
        height=180, 
        placeholder="e.g., I feel so alone and worthless lately, I don't know how to keep going."
    )

    if st.button('Analyze Risk Level', type="primary"):
        if not user_input.strip():
            st.warning('Please enter some text to analyze.')
            return

        with st.spinner('Analyzing features and running prediction...'):
            input_df = extract_features(user_input)
            
            if input_df is not None:
                try:
                    # 1. PRIMARY MODEL PREDICTION (Binary) - Still calculated for reference/confidence
                    # We must drop 'vader_neg' before passing to the model pipeline, as the model was trained without it
                    model_input_df = input_df.drop(columns=['vader_neg']) 
                    
                    prediction_numerical = predictor_pipeline.predict(model_input_df)[0]
                    proba = predictor_pipeline.predict_proba(model_input_df)[0]
                    risk_proba = proba[1] 
                    
                    # 2. HEURISTIC CATEGORY SCORING
                    # CRUCIAL CHANGE: Pass the new 'vader_neg' feature to the heuristic function
                    vader_neg_score = input_df['vader_neg'].iloc[0]
                    heuristic_scores = calculate_heuristic_scores(user_input, vader_neg_score)
                    
                    # 3. DETERMINE THE PRIMARY LABEL FROM HEURISTIC SCORES
                    top_category = max(heuristic_scores, key=heuristic_scores.get)
                    top_percentage = heuristic_scores[top_category]
                    
                    st.markdown('## Analysis Results')
                    
                    # --- DISPLAY: Primary Label (Your requested output) ---
                    st.subheader(f'🎯 Primary Risk Indication: {top_category.upper()}')
                    
                    # Determine color/icon based on the top category (Normal vs. Pathological)
                    if top_category.upper() == "NORMAL" or top_percentage < 30.0:
                        st.success(f'### 🟢 {top_category.upper()} ({top_percentage:.2f}%)')
                        st.info("The content primarily suggests a normal emotional state, though specific keywords may be present.")
                    else:
                        st.error(f'### 🔴 {top_category.upper()} ({top_percentage:.2f}%)')
                        st.warning(f"This content strongly suggests a primary focus on **{top_category.upper()}**. This heuristic assessment, combined with the underlying model's binary risk assessment, suggests seeking professional advice.")

                    st.markdown("---")
                    
                    # --- DISPLAY: Binary Model Confidence (For Context) ---
                    st.subheader('Binary Risk Model Assessment')
                    binary_label = PREDICTION_LABELS.get(prediction_numerical, "UNKNOWN_RISK")
                    if prediction_numerical == 1:
                        st.text(f"Model classified as: {binary_label} (Confidence: {risk_proba:.2%})")
                    else:
                        st.text(f"Model classified as: {binary_label} (Confidence: {1 - risk_proba:.2%})")
                        
                    st.markdown("---")
                    
                    # --- DISPLAY: Detailed Category Percentages (Heuristic Chart) ---
                    st.subheader('Detailed Category Breakdown (Percentages)')
                    
                    # Convert heuristic results to DataFrame for charting
                    chart_data = pd.DataFrame(
                        heuristic_scores.items(), 
                        columns=['Category', 'Percentage']
                    )
                    # Sort by percentage descending, put Normal last
                    chart_data['SortKey'] = chart_data.apply(
                        lambda row: -row['Percentage'] if row['Category'] != 'Normal' else row['Percentage'], axis=1
                    )
                    chart_data = chart_data.sort_values('SortKey', ascending=True)

                    # Create Altair bar chart
                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X('Percentage', title='Percentage of Contribution (%)'),
                        y=alt.Y('Category', sort='-x', title='Mental Health Category'),
                        tooltip=['Category', alt.Tooltip('Percentage', format='.2f')],
                        color=alt.Color('Category', scale=alt.Scale(
                            domain=list(heuristic_scores.keys()),
                            range=['#368038', '#EF4444', '#B91C1C', '#F59E0B', '#1D4ED8', '#6D28D9', '#7C3AED'] # Green for Normal, Red for high risk, etc.
                        ))
                    ).properties(
                        height=400
                    ).interactive() 
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    

                    # Display the final results as a table
                    final_table = chart_data[['Category', 'Percentage']].rename(columns={'Percentage': 'Contribution (%)'}).set_index('Category')
                    # Use explicit dictionary formatting for robustness
                    st.dataframe(final_table.style.format({'Contribution (%)': "{:.2f}%"}), use_container_width=True)

                    st.markdown('---')
                    st.markdown('#### Extracted Features (Including VADER Negative Score)')
                    
                    # Display all extracted features for debugging/transparency
                    # We must exclude the 'vader_compound' column from the list of columns to be formatted as float
                    styled_df = input_df.style.format(
                        '{:.4f}', 
                        subset=pd.IndexSlice[:, input_df.columns.difference(['clean_text'])] 
                    )
                    st.dataframe(styled_df, use_container_width=True)


                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    st.warning("Please verify that your feature extraction logic and the column names exactly match what the loaded pipeline expects.")


if __name__ == '__main__':
    main()