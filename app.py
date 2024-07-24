import streamlit as st
import joblib
import pandas as pd

def load_model():
    """
    Load the trained Random Forest Regressor model from a file.

    Returns:
        model: The loaded machine learning model.
    """
    return joblib.load("model_a.pkl")

def get_mappings():
    """
    Define and return the mappings for categorical variables.

    Returns:
        dict: A dictionary containing mappings for various categorical variables.
    """
    return {
        'crop': {'Rice': 0, 'Maize': 1},
        'soil': {'clay': 0, 'loamy': 1, 'sandy': 2},
        'pesticides': {'Gammalin 20': 0, 'Nogos 50': 1, 'Vetox 85': 2},
        'herbicides': {
            'PARAFORCE (PARAQUAT)': 1, 'FORCE UP (GLYPHOSATE)': 2, 'RiceForce (oxadiazon)': 3, '2,4-D': 4,
            'ATRAFORCE (ATRAZINE)': 5, 'BUTACHLOR': 6, '3MAIZEFORCE': 7, 'Propalnil + 2.4 D': 8, 'Bispyribac Sodium': 9,
            'POWERFORCE (Atrazine + Paraquat)': 10, 'BentaForce': 11, 'GuardForce': 12, 'ATALAR': 13, 'METOLACHLOR': 14
        },
        'seed': {
            'Faro-55': 1, 'Faro-54': 2, 'Faro-44': 3, 'Gawal-R1': 4, 'Faro-24': 5, 'Faro-62': 6,
            'SC-419': 7, 'DK-649': 8, 'DK-234': 9, 'DK-777': 10, 'SC-612': 11, 'Oba-98': 12
        },
        'season': {'Rainy': 0, 'Dry': 1},
        'fertilizer': {
            'NPK 15:15:15': 0, 'NPK 12:12:17': 1, 'NPK 20:10:10': 2, 'NPK 2-1-1': 3, 'NPK 12-11-12': 4, 'NPK 27:13:13': 5
        }
    }

def get_seed_options(crop):
    """
    Get seed options based on the selected crop type.

    Args:
        crop (str): The selected crop type.

    Returns:
        list: A list of seed options corresponding to the selected crop type.
    """
    if crop == "Rice":
        return ['Faro-55', 'Faro-54', 'Faro-44', 'Gawal-R1', 'Faro-24', 'Faro-62']
    else:
        return ['SC-419', 'DK-649', 'DK-234', 'DK-777', 'SC-612', 'Oba-98']

def validate_inputs(inputs):
    """
    Validate that all input fields have been filled.

    Args:
        inputs (dict): A dictionary containing user inputs.

    Returns:
        bool: True if all inputs are valid, False otherwise.
    """
    for key, value in inputs.items():
        if value is None or value == 0:
            return False
    return True

def map_inputs(inputs, mappings):
    """
    Map input values to their numerical representations.

    Args:
        inputs (dict): A dictionary containing user inputs.
        mappings (dict): A dictionary containing mappings for categorical variables.

    Returns:
        dict: A dictionary containing mapped inputs.
    """
    return {
        'Crop': [mappings['crop'][inputs['crop']]],
        'Avg_temp (C?)': [inputs['avg_temp']],
        'Avg_Humidity': [inputs['avg_humidity']],
        'Avg_Rainfall(mm)': [inputs['avg_rainfall']],
        'Season': [mappings['season'][inputs['season']]],
        'Fertilizer': [mappings['fertilizer'][inputs['fertilizer']]],
        'Nitrogen (Kg/ha': [inputs['nitrogen']],
        'Phosphorus (Kg/ha)': [inputs['phosphorus']],
        'Potassium(Kg/ha)': [inputs['potassium']],
        'pH': [inputs['ph']],
        'Soil Type': [mappings['soil'][inputs['soil_type']]],
        'Seed Rate (kg/ha)': [inputs['seed_rate']],
        'Area (ha)': [inputs['area']],
        'Pesticides': [mappings['pesticides'][inputs['pesticides']]],
        'Pesticides Rate (ml/ha)': [inputs['pesticides_rate']],
        'Herbicides': [mappings['herbicides'][inputs['herbicides']]],
        'Herbicides Rate (ml/ha)': [inputs['herbicides_rate']],
        'seed_type': [mappings['seed'][inputs['seed_type']]]
    }

def main():
    """
    Main function to run the Streamlit app for crop yield prediction.
    """
    # Load model and mappings
    model = load_model()
    mappings = get_mappings()

    # Title of the app
    st.title("Crop Yield Prediction")

    # Input fields for the features
    crop = st.selectbox("Crop", ["Rice", "Maize"])
    avg_temp = st.number_input("Average Temperature (C)", step=0.1)
    avg_humidity = st.number_input("Average Humidity (%)", step=0.1)
    avg_rainfall = st.number_input("Average Rainfall (mm)", step=0.1)
    season = st.selectbox("Season", ["Rainy", "Dry"])
    fertilizer = st.selectbox("Fertilizer", ['NPK 15:15:15', 'NPK 12:12:17', 'NPK 20:10:10', 'NPK 2-1-1', 'NPK 12-11-12', 'NPK 27:13:13'])
    nitrogen = st.number_input("Nitrogen (Kg/ha)", step=0.1)
    phosphorus = st.number_input("Phosphorus (Kg/ha)", step=0.1)
    potassium = st.number_input("Potassium (Kg/ha)", step=0.1)
    ph = st.number_input("pH", step=0.1)
    soil_type = st.selectbox("Soil Type", ['clay', 'loamy', 'sandy'])
    seed_rate = st.number_input("Seed Rate (kg/ha)", step=0.1)
    area = st.number_input("Area (ha)", step=0.1)
    pesticides = st.selectbox("Pesticides", ['Gammalin 20', 'Nogos 50', 'Vetox 85'])
    pesticides_rate = st.number_input("Pesticides Rate (ml/ha)", step=0.1)
    herbicides = st.selectbox("Herbicides", ['PARAFORCE (PARAQUAT)', 'FORCE UP (GLYPHOSATE)', 'RiceForce (oxadiazon)', '2,4-D', 'ATRAFORCE (ATRAZINE)', 'BUTACHLOR', '3MAIZEFORCE', 'Propalnil + 2.4 D', 'Bispyribac Sodium', 'POWERFORCE (Atrazine + Paraquat)', 'BentaForce', 'GuardForce', 'ATALAR', 'METOLACHLOR'])
    herbicides_rate = st.number_input("Herbicides Rate (ml/ha)", step=0.1)
    seed_type = st.selectbox("Seed Type", get_seed_options(crop))

    # Collect inputs
    inputs = {
        'crop': crop, 'avg_temp': avg_temp, 'avg_humidity': avg_humidity, 'avg_rainfall': avg_rainfall,
        'season': season, 'fertilizer': fertilizer, 'nitrogen': nitrogen, 'phosphorus': phosphorus,
        'potassium': potassium, 'ph': ph, 'soil_type': soil_type, 'seed_rate': seed_rate, 'area': area,
        'pesticides': pesticides, 'pesticides_rate': pesticides_rate, 'herbicides': herbicides,
        'herbicides_rate': herbicides_rate, 'seed_type': seed_type
    }

    # Predict crop yield
    if st.button("Predict Crop Yield"):
        if validate_inputs(inputs):
            input_data = pd.DataFrame(map_inputs(inputs, mappings))
            prediction = model.predict(input_data)
            st.write(f"Predicted Crop Yield: {prediction[0]}")
        else:
            st.warning("Please input values for all fields.")

if __name__ == "__main__":
    main()
