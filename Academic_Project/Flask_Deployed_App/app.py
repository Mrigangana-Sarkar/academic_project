# app.py
import os
import warnings
import pickle
import numpy as np
import pandas as pd
import torch
import requests
from flask import Flask, render_template, request, jsonify, session
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import datetime

# -------------------------
# Ignore warnings
warnings.filterwarnings("ignore")

# -------------------------
# Load disease and supplement data
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# -------------------------
# Load plant disease prediction model
model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    """Predict plant disease from uploaded image"""
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0)
    output = model(input_data).detach().numpy()
    index = np.argmax(output)
    return index

# -------------------------
# Load crop recommendation models
crop_model = pickle.load(open('model.pkl', 'rb'))
stand_scaler = pickle.load(open('standscaler.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))

crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# -------------------------
# Flask app initialization
app = Flask(__name__)
app.secret_key = "agrocare_super_secret_key_123"
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyCKNDZyjvsEBmzjJUjva1c4iUvf6pyp0rU"
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0.3,        
        "top_p": 0.9,
        "max_output_tokens": 1200
    }
)
# -------------------------
# Static pages
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/crop-recommendation')
def crop_recommendation_page():
    return render_template("Crop Recommendation.html")

# -------------------------
# 🔹 NEW: Chatbot pages
@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

SYSTEM_PROMPT = """
You are AgroCare AI, an expert agricultural assistant.

IMPORTANT BEHAVIOR RULES (HIGHEST PRIORITY):

1. Greeting Handling:
- If the user input is a greeting (e.g., hello, hi, hey, good morning, good evening):
- Respond ONLY using the format below and NOTHING else:

# Welcome:
- Hello! I am AgroCare AI, your agricultural assistant.
- Please ask a question related to crops, plant diseases, or farming practices.

2. If the input is NOT a greeting, strictly follow the output rules below.

--------------------------------------------------

OUTPUT FORMAT (STRICT - DO NOT VIOLATE):

1. Use ONLY this structure:

# Section Name:
- Bullet point
- Bullet point

2. Mandatory rules:
- Every section heading MUST start with '# ' and end with a colon (:)
- Section headings MUST be on their own line
- NEVER use '##', '###', '*', or bullets for headings
- Bullet points MUST start with '- '
- Headings must NEVER appear inside bullet points
- Do NOT write paragraphs
- Each bullet should be 1-2 lines only
- Use agriculture-focused academic language
- No emojis
- No filler or conversational text


If the format is violated, rewrite internally and output again.

IMPORTANT:
- If the response becomes long, LIMIT to the most important sections only.
- NEVER cut a section mid-bullet.
- Always finish the current section before stopping.
- Prefer fewer complete sections over incomplete output.

"""



# -------------------------
# Chatbot API with MEMORY
@app.route('/chatbot-api', methods=['POST'])
def chatbot_api():
    try:
        data = request.get_json(force=True)

        user_message = (data.get("message") or "").strip()
        message_lower = user_message.lower()

        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        # -------------------------------------------------
        # Initialize session memory
        # -------------------------------------------------
        if "chat_history" not in session:
            session["chat_history"] = []
            session["last_topic"] = None

        # -------------------------------------------------
        # Topic / crop detection
        # -------------------------------------------------
        crop_keywords = [
            "rice", "coffee", "wheat", "maize", "cotton",
            "banana", "mango", "apple", "grapes", "tea", "papaya"
        ]

        for crop in crop_keywords:
            if crop in message_lower:
                session["last_topic"] = crop
                break

        # -------------------------------------------------
        # Follow-up detection
        # -------------------------------------------------
        is_follow_up = (
            session.get("last_topic") is not None
            and len(session["chat_history"]) > 0
            and len(message_lower.split()) <= 8
        )



        final_user_message = user_message
        if is_follow_up:
            final_user_message = (
                f"Previous topic: {session['last_topic']} related plant disease.\n"
                f"User follow-up question: {user_message}\n"
                f"Respond assuming continuity from the previous explanation."
            )



        # -------------------------------------------------
        # Detect table intent (dynamic style override)
        # -------------------------------------------------
        table_request = any(
            word in message_lower
            for word in [
                "table", "tabular", "tabular form",
                "difference", "difference between",
                "compare", "comparison", "vs", "versus"
            ]
        )


        if table_request:
            effective_prompt = """
You are AgroCare AI, an expert agricultural assistant.

Override rules:
- The user explicitly requested a TABLE.
- Respond using a clear comparison table.
- Ignore bullet-only restrictions for this response.
- Keep the table concise and agriculture-focused.
"""
        else:
            effective_prompt = SYSTEM_PROMPT

        # -------------------------------------------------
        # Build FULL conversation with memory
        # -------------------------------------------------
        conversation = effective_prompt.strip() + "\n\n"
        MAX_HISTORY = 6
        for msg in session["chat_history"][-MAX_HISTORY:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation += f"{role}: {msg['text']}\n"


        conversation += f"User: {final_user_message}\nAssistant:"

        # -------------------------------------------------
        # Gemini call (single-shot, memory-safe)
        # -------------------------------------------------
        response = gemini_model.generate_content(conversation)
        bot_reply = response.text.strip()

        # -------------------------------------------------
        # Save memory
        # -------------------------------------------------
        session["chat_history"].append({
            "role": "user",
            "text": user_message
        })
        session["chat_history"].append({
            "role": "model",
            "text": bot_reply
        })

        session.modified = True

        return jsonify({"reply": bot_reply})

    except Exception as e:
        print("CHATBOT ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# -------------------------
# Plant disease detection route
@app.route('/submit', methods=['POST'])
def submit():
    image = request.files['image']
    filename = image.filename
    file_path = os.path.join('static/uploads', filename)
    image.save(file_path)

    pred = prediction(file_path)
    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]
    supplement_name = supplement_info['supplement name'][pred]
    supplement_image_url = supplement_info['supplement image'][pred]
    supplement_buy_link = supplement_info['buy link'][pred]

    return render_template(
        'submit.html',
        title=title,
        desc=description,
        prevent=prevent,
        image_url=image_url,
        pred=pred,
        sname=supplement_name,
        simage=supplement_image_url,
        buy_link=supplement_buy_link
    )

# -------------------------
# Crop recommendation route
@app.route("/predict", methods=['POST'])
def crop_predict():
    # Existing fields
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    # New fields from frontend
    soil_temp = request.form.get('SoilTemp')       # Soil Temperature (°C)
    soil_moisture = request.form.get('SoilMoisture')  # Soil Moisture (%)

    # Convert to floats (more robust for scaler input)
    try:
        feature_list = [
            float(N), float(P), float(K),
            float(temp), float(humidity), float(ph),
            float(rainfall),
            float(soil_temp) if soil_temp is not None else 0.0,
            float(soil_moisture) if soil_moisture is not None else 0.0
        ]
    except Exception as e:
        # If conversion fails, fall back to original string approach (keeps behavior safe)
        feature_list = [N, P, K, temp, humidity, ph, rainfall, soil_temp, soil_moisture]

    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = minmax_scaler.transform(single_pred)
    final_features = stand_scaler.transform(scaled_features)
    prediction_val = crop_model.predict(final_features)

    crop_image_dict = {
        "Rice": "rice.jpg",
        "Maize": "maize.jpg",
        "Jute": "jute.jpg",
        "Cotton": "cotton.jpg",
        "Coconut": "coconut.jpg",
        "Papaya": "papaya.jpg",
        "Orange": "orange.jpg",
        "Apple": "apple.jpg",
        "Muskmelon": "muskmelon.jpg",
        "Watermelon": "watermelon.jpg",
        "Grapes": "grapes.jpg",
        "Mango": "mango.jpg",
        "Banana": "banana.jpg",
        "Pomegranate": "pomegranate.jpg",
        "Lentil": "lentil.jpg",
        "Blackgram": "blackgram.jpg",
        "Mungbean": "mungbean.jpg",
        "Mothbeans": "mothbeans.jpg",
        "Pigeonpeas": "pigeonpeas.jpg",
        "Kidneybeans": "kidneybeans.jpg",
        "Chickpea": "chickpea.jpg",
        "Coffee": "coffee.jpg"
    }

    if prediction_val[0] in crop_dict:
        crop = crop_dict[prediction_val[0]]
        image_file = crop_image_dict.get(crop, "default.jpg")
        image_path = f"/static/crops/{image_file}"

        result = f"{crop} is the best crop to be cultivated here."
    else:
        crop = None
        image_path = None
        result = "Sorry, we could not determine the best crop with the provided data."

    return render_template(
        'crop recommendation.html',
        result=result,
        crop=crop,
        crop_image=image_path
    )

# -------------------------
# Weather data fetch – current temperature + avg daily humidity + 6-month avg rainfall
@app.route("/fetch-weather")
def fetch_weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "Latitude and Longitude required"}), 400

    try:
        # 1️⃣ Fetch current temperature and full-day hourly humidity
        current_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,relative_humidity_2m"
            "&current_weather=true&timezone=auto"
        )
        current_data = requests.get(current_url, timeout=10).json()

        temperature = current_data.get("current_weather", {}).get("temperature", 0.0)

        # Average humidity of current day
        times = current_data.get("hourly", {}).get("time", [])
        humidity_list = current_data.get("hourly", {}).get("relative_humidity_2m", [])
        avg_humidity = None
        if times and humidity_list:
            today_str = datetime.date.today().isoformat()
            today_values = [
                h for t, h in zip(times, humidity_list)
                if t.startswith(today_str)
            ]
            if today_values:
                avg_humidity = round(sum(today_values) / len(today_values), 2)

        # 2️⃣ Historical rainfall for past 1 week
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=7)

        rain_url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            "&daily=precipitation_sum&timezone=auto"
        )

        rain_data = requests.get(rain_url, timeout=15).json()
        rain_values = rain_data.get("daily", {}).get("precipitation_sum", [])
        avg_rainfall = round(sum(rain_values) / len(rain_values), 2) if rain_values else 0.0

        return jsonify({
            "Temperature": temperature,
            "Humidity": avg_humidity,
            "Rainfall": avg_rainfall
        })

    except Exception as e:
        return jsonify({"error": f"Weather fetch failed: {str(e)}"}), 500

# -------------------------
# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
