import numpy as np
import joblib
import streamlit as st

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

survival = {
    1 : 'Survived',
    0 : 'Not Survived'
}

st.set_page_config(
    page_title= 'Titanic Survival Prediction',
    page_icon='🚢',
    layout='centered'
)

st.title('🚢 Titanic Survival Prediction')
st.markdown('Enter passenger details to predict survival chances.')
st.divider()

st.subheader('📋 Titanic Details')
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

    sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.slider(
        "Age",
        1, 80, 25
    )

    sibsp = st.slider(
        "Siblings / Spouses Aboard",
        0, 8, 0
    )

with col2:
    parch = st.slider(
        "Parents / Children Aboard",
        0, 6, 0
    )

    fare = st.slider(
        "Ticket Fare ($)",
        0.0, 600.0, 50.0, 1.0
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["C", "Q", "S"]
    )

sex = 1 if sex == "Male" else 0

embarked_map = {
    "C": 0,
    "Q": 1,
    "S": 2
}

embarked = embarked_map[embarked]
st.divider()

if st.button("🔍 Predict Survival", use_container_width=True):

    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    features = np.array([[
        pclass,
        sex,
        age,
        sibsp,
        parch,
        fare,
        embarked,
        family_size,
        is_alone
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    survived = survival[prediction]

    if prediction == 1:
        st.success(f"🎉 Prediction: {survived}")
    else:
        st.error(f"⚠️ Prediction: {survived}")

    st.markdown("#### 📊 Passenger Details Summary")

    st.dataframe({
        "Feature": [
            "Passenger Class",
            "Gender",
            "Age",
            "Siblings/Spouses",
            "Parents/Children",
            "Fare",
            "Embarked",
            "Family Size",
            "Is Alone"
        ],
        "Value": [
            pclass,
            "Male" if sex == 1 else "Female",
            age,
            sibsp,
            parch,
            fare,
            list(embarked_map.keys())[list(embarked_map.values()).index(embarked)],
            family_size,
            is_alone
        ]
    }, use_container_width=True)

st.markdown("---")
st.caption("Built with Scikit-learn + Streamlit · Titanic Dataset")