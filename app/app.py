import streamlit as st
import joblib
import numpy as np

st.title("AI Enabled Visa Processing Time Estimator")

# -----------------------------
# Load model and encoders
# -----------------------------
model = joblib.load("visa_model.pkl")
center_encoder = joblib.load("center_encoder.pkl")
status_encoder = joblib.load("status_encoder.pkl")

# -----------------------------
# US State Full Name Mapping
# -----------------------------
state_full_names = {
    "AK": "Alaska",
    "AL": "Alabama",
    "AR": "Arkansas",
    "AS": "American Samoa",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DC": "District of Columbia",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "PR": "Puerto Rico",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming"
}

# -----------------------------
# Prepare dropdown options
# -----------------------------
center_codes = list(center_encoder.classes_)
statuses = list(status_encoder.classes_)

# Create display labels
display_centers = []
code_mapping = {}

for code in center_codes:
    full_name = state_full_names.get(code, code)
    label = f"{full_name} ({code})"
    display_centers.append(label)
    code_mapping[label] = code

# -----------------------------
# Dropdown UI
# -----------------------------
selected_center_label = st.selectbox("Select Processing Center", display_centers)
selected_status = st.selectbox("Select Case Status", statuses)

if st.button("Estimate Processing Time"):

    # Get actual state code from label
    selected_center_code = code_mapping[selected_center_label]

    # Encode values
    center_encoded = center_encoder.transform([selected_center_code])[0]
    status_encoded = status_encoder.transform([selected_status])[0]

    input_data = np.array([[center_encoded, status_encoded]])

    prediction = model.predict(input_data)

    st.success(f"Estimated Processing Time: {int(prediction[0])} days")
