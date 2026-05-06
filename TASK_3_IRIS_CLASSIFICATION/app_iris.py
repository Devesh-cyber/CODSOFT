import numpy as np
import joblib
import streamlit as st

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')


species_map = {0: 'Iris Setosa 🌸', 
               1: 'Iris Versicolor 🌿', 
               2: 'Iris Virginica 🌺'}

st.set_page_config(
    page_title = "Iris Flower Classification Predictor",
    page_icon = '🌺',
    layout = 'centered'
)

st.title("🌺 Iris Flowe Classification Predictor")
st.markdown('Fill in the details below to get an estimated Flower')
st.divider()

st.subheader("📋 Flower Measurments")

col1, col2 = st.columns(2)

with col1: 
    sepal_length = st.number_input("Sepal Length (cm) : ", 4.3,  7.9, 5.1)
    sepal_width = st.number_input("Sepal Width (cm) : ", 2.0, 4.4, 3.5)

with col2:
    petal_length = st.number_input("Petal Length (cm): ", 1.0, 6.9, 1.4)
    petal_width = st.number_input("Petal Width (cm) : ", 0.1, 2.5, 0.2)

st.divider()

if st.button("🔍 Predict Species", use_container_width=True):
    features = np.array([[
        sepal_length, sepal_width,
        petal_length, petal_width
    ]])

    features_scaler = scaler.transform(features)
    prediction = model.predict(features_scaler)[0]
    flower_name = species_map[prediction]

    st.success(f'### 🌺 Predicted Species - {flower_name}')

    st.markdown("#### 📊 Your Inputs")
    st.dataframe({
        "Feature" : [
            "Sepal Length", "Sepal Width",
            "Petal Lenghth", "Petal Width"
        ],
        "Value" : [
            sepal_length, sepal_width,
            petal_length, petal_width
        ]
    }, use_container_width=True)

st.markdown("---")
st.caption("Built with Scikit-learn + Streamlit · California Housing Dataset")