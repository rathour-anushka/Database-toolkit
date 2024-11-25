import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Read the dataset
        df = pd.read_csv(file)

    except Exception as e:
        return jsonify({"error": f"Error reading file: {e}"}), 500

    # Get the user prompt
    user_prompt = request.form.get('prompt')
    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Prepare the data and prompt for the model
    prompt = f"Here is the dataset: {df.head().to_dict()}. Based on this data, {user_prompt}"

    # Interact with the Gemini API
    try:
        chat_session = model.start_chat()
        response = chat_session.send_message(prompt)
        suggestions = response.text
    except Exception as e:
        return jsonify({"error": f"Error during chat: {e}"}), 500

    if "drop_na" in suggestions.lower():
        df = df.dropna()
    
    # to develop ---------------if the response suggests creating a histogram
    if "create histogram" in suggestions.lower():
        plt.figure(figsize=(10, 6))
        df[df.columns[0]].hist(bins=30)
        plt.title('Histogram of ' + df.columns[0])
        plt.xlabel(df.columns[0])
        plt.ylabel('Frequency')
        plt.savefig('static/histogram.png')
        plt.close()
        visualization_url = "static/histogram.png"
    else:
        visualization_url = None  # No visualization generated

    return jsonify({
        "message": "File processed successfully.",
        "suggestions": suggestions,
        "visualization_url": visualization_url
    })

if __name__ == '__main__':
    app.run(debug=True)