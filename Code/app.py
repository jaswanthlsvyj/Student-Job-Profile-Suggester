import streamlit as st
import numpy as np
import joblib  # Using joblib for loading the saved model
import os

# Assuming the model file is in the same directory as your script (app.py)
model_path = "Models/SVC_accuracy_99.joblib"

try:
    model, skill_encoder, profile_encoder = joblib.load(model_path)
except FileNotFoundError:
    print("Error: Model file not found!")

# Function to get user input
def get_user_input():
    # Collecting user inputs for each feature using sliders
    dsa = st.slider('DSA score (0-100)', min_value=0, max_value=100, value=50)
    dbms = st.slider('DBMS score (0-100)', min_value=0, max_value=100, value=50)
    os = st.slider('Operating Systems score (0-100)', min_value=0, max_value=100, value=50)
    cn = st.slider('Computer Networks score (0-100)', min_value=0, max_value=100, value=50)
    mathematics = st.slider('Mathematics score (0-100)', min_value=0, max_value=100, value=50)
    aptitude = st.slider('Aptitude score (0-100)', min_value=0, max_value=100, value=50)
    communication = st.slider('Communication score (0-100)', min_value=0, max_value=100, value=50)
    problem_solving = st.slider('Problem Solving score (0-10)', min_value=0, max_value=10, value=5)
    creativity = st.slider('Creativity score (0-10)', min_value=0, max_value=10, value=5)
    hackathons = st.slider('Number of Hackathons', min_value=0, max_value=10, value=1)

    skillsList = ('Angular', 'Ansible', 'BASH/SHELL', 'C/C++', 'Cisco Packet tracer', 
                  'Deep Learning', 'Figma', 'GitHub', 'HTML/CSS', 'Java', 
                  'Javascript', 'Linux', 'MYSQL', 'Machine Learning', 
                  'Node.js', 'Oracle', 'Photoshop', 'Python', 
                  'Pytorch', 'R', 'React', 'Tensorflow', 'Wire Shark')
    
    # Drop-down for skill selections
    skill_1 = st.selectbox('Skill 1', skillsList)
    skill_2 = st.selectbox('Skill 2', skillsList)

    # Create a 2D array for numerical features
    numerical_array = np.array([[dsa, dbms, os, cn, mathematics, aptitude, communication, problem_solving, creativity, hackathons]])

    skill_1_encoded = skill_encoder.transform([skill_1]).reshape(1, -1)  # Reshape to (1, n_features)
    skill_2_encoded = skill_encoder.transform([skill_2]).reshape(1, -1)  # Reshape to (1, n_features)

    # Concatenate the numerical features and skills
    user_input = np.concatenate((numerical_array, skill_1_encoded, skill_2_encoded), axis=1)

    return user_input

# Main function to build the Streamlit app
def main():
    st.title("Job Profile Prediction")
    st.write("Input your skills and scores to predict the most suitable job profile.")

    # Get user input
    user_input = get_user_input()

    # When the user clicks the "Predict" button
    if st.button('Predict Job Profile'):
        # Make a prediction based on the user input
        prediction = model.predict(user_input)
        
        predicted_profile = profile_encoder.inverse_transform(prediction)
        
        # Display the prediction result
        st.success(f'Predicted Job Profile: {predicted_profile[0]}')  # Changed to [0] to get the string result

# Run the app
if __name__ == '__main__':
    main()
