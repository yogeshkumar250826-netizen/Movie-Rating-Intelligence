import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Rating Intelligence",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM UI
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.7rem !important;
        font-weight: 800 !important;
    }

    h2, h3 {
        font-weight: 700 !important;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        padding: 18px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
    }

    .info-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 15px;
    }

    .app-subtitle {
        font-size: 1.1rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    .footer {
        text-align: center;
        opacity: 0.55;
        padding-top: 30px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD MODEL AND PREPROCESSING ARTIFACTS
# --------------------------------------------------

MODEL_PATH = os.path.join(
    "models",
    "movie_rating_model.pkl"
)

ARTIFACT_PATH = os.path.join(
    "models",
    "preprocessing_artifacts.pkl"
)

model = joblib.load(MODEL_PATH)

artifacts = joblib.load(
    ARTIFACT_PATH
)

duration_median = artifacts["duration_median"]
frequency_maps = artifacts["frequency_maps"]
training_genres = artifacts["training_genres"]
feature_columns = artifacts["feature_columns"]

# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("🎬 Movie Rating Intelligence")

st.markdown(
    """
    <div class="app-subtitle">
        Explainable movie rating prediction powered by machine learning.
        Enter a movie concept, estimate its IMDb-style rating, and discover
        which characteristics influenced the prediction.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        <b>Prediction + Explanation</b><br>
        The system combines movie metadata with a Gradient Boosting model
        and SHAP explanations to provide more than just a rating estimate.
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# MOVIE INPUTS
# --------------------------------------------------

st.header("Movie Information")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input(
        "Release Year",
        min_value=1917,
        max_value=2030,
        value=2020,
        step=1
    )

    duration = st.number_input(
        "Duration (minutes)",
        min_value=1,
        max_value=400,
        value=135,
        step=1
    )

    votes = st.number_input(
        "IMDb Vote Count",
        min_value=0,
        value=1000,
        step=100
    )

with col2:
    selected_genres = st.multiselect(
        "Genre",
        options=training_genres,
        default=["Drama"] if "Drama" in training_genres else []
    )

    director = st.text_input(
        "Director",
        placeholder="Enter director name"
    )

    actor_1 = st.text_input(
        "Actor 1",
        placeholder="Enter first actor"
    )

    actor_2 = st.text_input(
        "Actor 2",
        placeholder="Enter second actor"
    )

    actor_3 = st.text_input(
        "Actor 3",
        placeholder="Enter third actor"
    )

st.divider()

predict_button = st.button(
    "Predict Rating",
    type="primary",
    use_container_width=True
)

# --------------------------------------------------
# PREPROCESS USER INPUT
# --------------------------------------------------

def prepare_movie_input(
    year,
    duration,
    votes,
    genres,
    director,
    actor_1,
    actor_2,
    actor_3
):
    movie = {
        feature: 0.0
        for feature in feature_columns
    }

    # Numerical features
    movie["Year"] = year
    movie["Duration"] = (
        duration if duration > 0
        else duration_median
    )
    movie["Log_Votes"] = np.log1p(votes)

    # Frequency encoded features
    people = {
        "Director": director,
        "Actor 1": actor_1,
        "Actor 2": actor_2,
        "Actor 3": actor_3
    }

    for column, value in people.items():
        movie[column + "_Frequency"] = (
            frequency_maps[column].get(
                value.strip(),
                0
            )
        )

    # Genre features
    for genre in genres:
        genre_column = (
            "Genre_" +
            genre.replace(" ", "_")
        )

        if genre_column in movie:
            movie[genre_column] = 1

    # If no genre is selected
    if (
        len(genres) == 0
        and "Genre_Unknown" in movie
    ):
        movie["Genre_Unknown"] = 1

    movie_df = pd.DataFrame(
        [movie],
        columns=feature_columns
    )

    return movie_df

# --------------------------------------------------
# MAKE PREDICTION
# --------------------------------------------------

# --------------------------------------------------
# DISPLAY NAMES FOR EXPLANATIONS
# --------------------------------------------------

feature_display_names = {
    "Year": "Release Year",
    "Duration": "Movie Duration",
    "Log_Votes": "Audience Vote Count",
    "Director_Frequency": "Director Experience",
    "Actor 1_Frequency": "Lead Actor Experience",
    "Actor 2_Frequency": "Second Actor Experience",
    "Actor 3_Frequency": "Third Actor Experience"
}


def get_display_name(feature):
    if feature.startswith("Genre_"):
        return (
            feature
            .replace("Genre_", "Genre: ")
            .replace("_", " ")
        )

    return feature_display_names.get(
        feature,
        feature.replace("_", " ")
    )


# --------------------------------------------------
# MAKE PREDICTION
# --------------------------------------------------

if predict_button:

    movie_input = prepare_movie_input(
        year,
        duration,
        votes,
        selected_genres,
        director,
        actor_1,
        actor_2,
        actor_3
    )

    predicted_rating = model.predict(
        movie_input
    )[0]

    # Keep displayed estimate within IMDb rating scale
    displayed_rating = np.clip(
        predicted_rating,
        1,
        10
    )

    st.header("Movie Rating Estimate")

    # Rating category
    if displayed_rating >= 8:
        rating_label = "Excellent"
    elif displayed_rating >= 7:
        rating_label = "Strong"
    elif displayed_rating >= 6:
        rating_label = "Good"
    elif displayed_rating >= 5:
        rating_label = "Average"
    else:
        rating_label = "Below Average"

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric(
            label="Estimated Rating",
            value=f"{displayed_rating:.1f} / 10"
        )

    with result_col2:
        st.metric(
            label="Rating Category",
            value=rating_label
        )

    with result_col3:
        st.metric(
            label="Model Used",
            value="Gradient Boosting"
        )

    st.progress(
        int(displayed_rating * 10)
    )

    st.caption(
        "The rating is a machine-learning estimate based on patterns "
        "learned from historical movie data and is not an actual IMDb rating."
    )

    st.markdown("### Prediction Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.markdown(
            f"""
            <div class="info-card">
                <b>Movie Profile</b><br><br>
                Release Year: {year}<br>
                Duration: {duration} minutes<br>
                Votes: {votes:,}
            </div>
            """,
            unsafe_allow_html=True
        )

    with summary_col2:
        genre_text = (
            ", ".join(selected_genres)
            if selected_genres
            else "Not specified"
        )

        st.markdown(
            f"""
            <div class="info-card">
                <b>Creative Profile</b><br><br>
                Genre: {genre_text}<br>
                Director: {director if director else "Not specified"}<br>
                Lead Actor: {actor_1 if actor_1 else "Not specified"}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()

with st.expander("About the Prediction Model"):

    st.markdown(
        """
        ### Model Performance

        Three regression models were evaluated on unseen test data.
        Gradient Boosting produced the strongest overall performance
        and was selected as the final prediction model.
        """
    )

    performance_data = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "Gradient Boosting"
        ],
        "MAE": [
            0.9205,
            0.8096,
            0.7894
        ],
        "RMSE": [
            1.1816,
            1.0679,
            1.0369
        ],
        "R²": [
            0.2490,
            0.3868,
            0.4212
        ]
    })

    st.dataframe(
        performance_data,
        hide_index=True,
        use_container_width=True
    )

    st.markdown(
        """
        **Selected Model:** Gradient Boosting

        **MAE:** 0.7894 — predictions differ from actual ratings
        by about 0.79 rating points on average.

        **R²:** 0.4212 — the model explains part of the variation
        in movie ratings, while also showing that movie ratings
        depend on factors not captured by the available metadata.

        The application should therefore be treated as an
        explainable rating estimation system rather than a
        guaranteed rating predictor.
        """
    )

    # --------------------------------------------------
    # EXPLAIN PREDICTION WITH SHAP
    # --------------------------------------------------

    st.subheader(
        "Why did the model predict this rating?"
    )

    explainer = shap.Explainer(model)

    movie_shap = explainer(
        movie_input
    )

    shap_contributions = pd.DataFrame({
        "Feature": feature_columns,
        "Contribution": movie_shap.values[0]
    })

    shap_contributions["Absolute_Contribution"] = (
        shap_contributions["Contribution"].abs()
    )

    shap_contributions = (
        shap_contributions
        .sort_values(
            "Absolute_Contribution",
            ascending=False
        )
        .head(8)
    )

    positive_features = shap_contributions[
        shap_contributions["Contribution"] > 0
    ]

    negative_features = shap_contributions[
        shap_contributions["Contribution"] < 0
    ]

    col_positive, col_negative = st.columns(2)

    # Positive Contributions
    with col_positive:

        st.markdown(
            "#### ↑ Pushing the estimate higher"
        )

        if len(positive_features) == 0:
            st.write(
                "No strong positive contributions."
            )

        for _, row in positive_features.iterrows():

            display_name = get_display_name(
                row["Feature"]
            )

            st.write(
                f"**{display_name}** "
                f"(+{row['Contribution']:.3f})"
            )

    # Negative Contributions
    with col_negative:

        st.markdown(
            "#### ↓ Pushing the estimate lower"
        )

        if len(negative_features) == 0:
            st.write(
                "No strong negative contributions."
            )

        for _, row in negative_features.iterrows():

            display_name = get_display_name(
                row["Feature"]
            )

            st.write(
                f"**{display_name}** "
                f"({row['Contribution']:.3f})"
            )

    st.caption(
        "SHAP contributions explain how features influenced "
        "the model's estimate relative to its baseline prediction. "
        "They do not represent causal effects on IMDb ratings."
    )