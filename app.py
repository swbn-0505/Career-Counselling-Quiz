from flask import Flask, request, send_file, Response
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize the Flask application
app = Flask(__name__)

# Home page route
@app.route('/career-counselling-quiz')
def home():
    # Serve the index.html file from the parent directory
    return send_file('index.html')

# Result page route
@app.route('/result', methods=['POST'])
def result():
    # Get user input from the form
    maths = int(request.form['maths'])
    phy_chem = int(request.form['phy_chem'])
    computer = int(request.form['computer'])
    medical = int(request.form['medical'])
    literature = int(request.form['literature'])
    humanities = int(request.form['humanities'])
    personality = int(request.form['personality'])
    hobbies = int(request.form['hobbies'])

    # Create an input array for prediction
    user_input = np.array([[maths, phy_chem, computer, medical, literature, humanities, personality, hobbies]])

    # Predict the career based on user input
    predicted_career = model.predict(user_input)[0]

    # Read the result.html file and inject the career prediction dynamically
    with open('result.html', 'r') as file:
        result_html = file.read()

    # Replace a placeholder in result.html with the actual career prediction
    result_html = result_html.replace('{{ career }}', predicted_career)

    # Return the modified HTML as a response
    return Response(result_html, content_type='text/html')

if __name__ == '__main__':
    app.run(debug=True, port=5500)



