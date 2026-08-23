import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download


st.set_page_config(
    page_title="Luxury Watch Price Prediction",
    page_icon="⌚",
    layout="wide"
)

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #a7a9b4;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 5px;
}

.section-caption {
    color: #8f929e;
    font-size: 14px;
    margin-bottom: 20px;
}

.result-label {
    font-size: 18px;
    font-weight: 600;
    margin-top: 18px;
    margin-bottom: 8px;
}

.category-card {
    padding: 18px 22px;
    border-radius: 10px;
    background: #123d2a;
    border: 1px solid #1f6b48;
    font-size: 25px;
    font-weight: 700;
}

.price-card {
    padding: 20px 22px;
    border-radius: 10px;
    background: #123d2a;
    border: 1px solid #1f6b48;
}

.price-usd {
    font-size: 30px;
    font-weight: 800;
    margin-bottom: 5px;
}

.price-inr {
    font-size: 20px;
    font-weight: 600;
    color: #a7f3c5;
}

.footer {
    text-align: center;
    color: #777b87;
    font-size: 13px;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

HF_REPO = "TASMVIDIT07/watch-price-prediction-model"


@st.cache_resource
def load_models():

    price_model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="random_forest_model.pkl"
    )

    category_model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="watch_category_classifier.pkl"
    )

    price_encoders_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="label_encoders.pkl"
    )

    category_encoders_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="classification_label_encoders.pkl"
    )

    price_model = joblib.load(price_model_path)
    category_model = joblib.load(category_model_path)

    price_encoders = joblib.load(price_encoders_path)
    category_encoders = joblib.load(category_encoders_path)

    return (
        price_model,
        category_model,
        price_encoders,
        category_encoders
    )


@st.cache_data
def load_data():

    watches_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="watches_cleaned.csv"
    )

    return pd.read_csv(watches_path)


price_model, category_model, price_encoders, category_encoders = load_models()

df = load_data()



df = pd.read_csv("watches_cleaned.csv")

st.markdown(
    '<div class="main-title">⌚ Luxury Watch Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered category and market price prediction for luxury watches.</div>',
    unsafe_allow_html=True
)

st.divider()

st.info(
    "⚠️ Predictions are generated using machine learning models trained on historical luxury watch listings. Results are estimates and may vary for rare or limited-edition watches."
)
#-------------------------------------------------------------
                #UI STARTS HERE
#-------------------------------------------------------------
st.sidebar.title("📌 Project Information")

st.sidebar.markdown("""
### 🤖 Models Used

- Random Forest Regressor
- Random Forest Classifier
""")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 Dataset")

st.sidebar.link_button(
    "🔗 View Dataset",
    "https://www.kaggle.com/datasets/philmorekoung11/luxury-watch-listings"
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 👨‍💻 Developed By

**Vidit Vohra**
""")

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:

    st.markdown(
    '<div class="section-title">⌚ Watch Details</div>',
    unsafe_allow_html=True
)

    st.markdown(
    '<div class="section-caption">Enter the specifications of the watch</div>',
    unsafe_allow_html=True
)

    st.markdown("---")

    brand = st.selectbox(
        "Brand",
        sorted(df["brand"].dropna().unique())
    )

    brand_df = df[df["brand"] == brand]

    model = st.selectbox(
        "Model",
        sorted(brand_df["model"].dropna().unique())
    )

    model_df = brand_df[brand_df["model"] == model]

    reference = st.selectbox(
        "Reference",
        sorted(model_df["ref"].dropna().unique())
    )

    condition = st.selectbox(
        "Condition",
        sorted(df["cond"].dropna().unique())
    )

    gender = st.selectbox(
        "Gender",
        sorted(df["sex"].dropna().unique())
    )

    year = st.number_input(
        "Year of Production",
        min_value=1900,
        max_value=2030,
        value=2020
    )

    size = st.number_input(
        "Watch Size (mm)",
        min_value=20,
        max_value=60,
        value=40
    )

    st.markdown("<br>", unsafe_allow_html=True)

    predict = st.button(
        "🔮 Predict Watch",
        use_container_width=True
    )


with right_col:

        st.markdown(
    '<div class="section-title">📈 Prediction</div>',
    unsafe_allow_html=True
)

        st.markdown(
    '<div class="section-caption">Machine Learning estimated results</div>',
    unsafe_allow_html=True
)

        st.markdown("---")

        if not predict:

            st.info(
            "👈 Enter the watch details and click "
            "**Predict Watch** to generate the prediction."
        )

        else:
         try:

            input_data = pd.DataFrame({
                "brand": [brand],
                "model": [model],
                "ref": [reference],
                "yop": [year],
                "cond": [condition],
                "sex": [gender],
                "size": [size]
            })

            price_input = input_data.copy()

            for col in ["brand", "model", "ref", "cond", "sex"]:
                price_input[col] = price_encoders[col].transform(
                    price_input[col]
                )

            predicted_price = price_model.predict(price_input)[0]

            if predicted_price <= 3209:
                predicted_category = "Basic"
            elif predicted_price <= 15090:
                predicted_category = "Mid"
            elif predicted_price <= 62040:
                predicted_category = "Premium"
            else:
                predicted_category = "Luxury"

            usd_to_inr = 95.21
            price_inr = predicted_price * usd_to_inr

            if price_inr >= 10000000:
                price_display = f"₹{price_inr / 10000000:.2f} Cr"
            elif price_inr >= 100000:
                price_display = f"₹{price_inr / 100000:.2f} Lakh"
            else:
                price_display = f"₹{price_inr:,.0f}"

            st.markdown(
                '<div class="result-label">🏷️ Predicted Category</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="category-card">{predicted_category}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="result-label">💰 Estimated Market Price</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="price-card">
                    <div class="price-usd">$ {predicted_price:,.2f}</div>
                    <div class="price-inr">≈ {price_display}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("---")

            st.caption(
            "The displayed price is an ML-based estimate and may vary "
            "from the actual market value.")

        

         except Exception:
            st.warning(
                "⚠️ Unable to generate prediction for these watch details. "
                "Please try another combination."
            )
st.markdown(
    """
    <div class="footer">
        Developed by Vidit Vohra • Powered by Machine Learning & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
