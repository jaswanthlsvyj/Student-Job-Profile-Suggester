import streamlit as st
import numpy as np
import joblib  # Using joblib for loading the saved model

# Assuming the model file is in the same directory as your script (app.py)
model_path = "Models/best_models.joblib"

# Initialize model components
best_models, skill_encoder, profile_encoder = None, None, None

# Try to load the model and handle errors
try:
    best_models, skill_encoder, profile_encoder = joblib.load(model_path)
except FileNotFoundError:
    st.error("Error: Model file not found!")
    st.stop()  # Stop further execution if the model is not loaded

# Function to get user input
def get_user_input():
    # Collecting user inputs for each feature using sliders
    dsa = st.slider('DSA score (0-100)', min_value=0, max_value=100, value=72)
    dbms = st.slider('DBMS score (0-100)', min_value=0, max_value=100, value=74)
    os = st.slider('Operating Systems score (0-100)', min_value=0, max_value=100, value=73)
    cn = st.slider('Computer Networks score (0-100)', min_value=0, max_value=100, value=60)
    mathematics = st.slider('Mathematics score (0-100)', min_value=0, max_value=100, value=87)
    aptitude = st.slider('Aptitude score (0-100)', min_value=0, max_value=100, value=82)
    communication = st.slider('Communication score (0-100)', min_value=0, max_value=100, value=64)
    problem_solving = st.slider('Problem Solving score (0-10)', min_value=0, max_value=10, value=7)
    creativity = st.slider('Creativity score (0-10)', min_value=0, max_value=10, value=6)
    hackathons = st.slider('Number of Hackathons', min_value=0, max_value=10, value=4)

    skillsList = ('Angular', 'Ansible', 'BASH/SHELL', 'C/C++', 'Cisco Packet tracer', 
                  'Deep Learning', 'Figma', 'GitHub', 'HTML/CSS', 'Java', 
                  'Javascript', 'Linux', 'MYSQL', 'Machine Learning', 
                  'Node.js', 'Oracle', 'Photoshop', 'Python', 
                  'Pytorch', 'R', 'React', 'Tensorflow', 'Wire Shark')
    
    # Drop-down for skill selections
    skill_1 = st.selectbox('Skill 1', skillsList, index=21)
    skill_2 = st.selectbox('Skill 2', skillsList, index=19)

    # Encode the skills
    skill_1_encoded = skill_encoder.transform([skill_1])[0]  
    skill_2_encoded = skill_encoder.transform([skill_2])[0] 

    # Create a list of numerical features i.e. user_input
    user_input = [dsa, dbms, os, cn, mathematics, aptitude, communication, problem_solving, creativity, hackathons, skill_1_encoded, skill_2_encoded]

    # Concatenate the numerical features and skills i.e. user_input
    user_input = np.array(user_input).reshape(1, -1)

    return user_input

# Main function to build the Streamlit app
def main():
    st.title("Job Profile Prediction")
    st.write("Input your skills and scores to predict the most suitable job profile.")

    # Get user input
    user_input_original = get_user_input()

    # When the user clicks the "Predict" button
    if st.button('Predict Job Profile'):
        # Make a prediction based on the user input
        user_input = user_input_original.copy()
        # Predict using the model
        predicted_profile = best_models['XGB'].predict(user_input)
        predicted_job = profile_encoder.inverse_transform(predicted_profile)
        
        # Display the prediction result
        st.success(f'Predicted Job Profile : {predicted_job[0]}')  


# Run the app
if __name__ == '__main__':
    main()
