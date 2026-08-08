# 🌱 AgroCare: Plant Disease Detection & Crop Recommendation System

AgroCare is an AI-driven smart agriculture assistance system that
combines **Internet of Things (IoT), Machine Learning, Deep Learning,
and Natural Language Processing** to provide intelligent agricultural
support.

The system assists farmers with **crop recommendation, soil monitoring,
environmental analysis, plant disease detection, and agriculture-related
guidance through an AI-powered chatbot**.

## 📌 Table of Contents

-   [About the Project](#-about-the-project)
-   [Key Features](#-key-features)
-   [System Architecture](#-system-architecture)
-   [Project Workflow](#-project-workflow)
-   [Machine Learning Module](#-machine-learning-module)
-   [Plant Disease Detection](#-plant-disease-detection)
-   [IoT and Hardware Module](#-iot-and-hardware-module)
-   [Chatbot Module](#-chatbot-module)
-   [Technologies Used](#-technologies-used)
-   [Input Parameters](#-input-parameters)
-   [Hardware Components](#-hardware-components)
-   [Project Structure](#-project-structure)
-   [Installation](#-installation)
-   [Configuration](#-configuration)
-   [Running the Project](#-running-the-project)
-   [How the System Works](#-how-the-system-works)
-   [Results](#-results)
-   [Future Scope](#-future-scope)
-   [Team Members](#-team-members)
-   [Supervisor](#-supervisor)
-   [References](#-references)

## 🌾 About the Project

Agriculture plays a major role in food security and economic
development. Selecting an appropriate crop based on soil and
environmental conditions is important for improving productivity and
maintaining soil health.

AgroCare addresses this problem by integrating:

-   🌱 Machine Learning
-   🤖 Artificial Intelligence
-   📡 IoT-based soil monitoring
-   🌦️ Weather API integration
-   💬 AI-powered agricultural chatbot
-   🧠 Deep Learning for plant disease detection
-   🌐 Web-based user interface

The Crop Recommendation module collects soil parameters such as
**Nitrogen (N), Phosphorus (P), Potassium (K), soil pH, soil moisture,
and soil temperature**, while environmental parameters such as
**temperature, humidity, and rainfall** are obtained through an external
weather API. These inputs are processed by a machine learning
classification model to recommend a suitable crop.

The system also includes a **Gemini API-powered chatbot** that allows
users to ask natural-language questions about crops, soil conditions,
fertilizers, diseases, and farming practices.

## 🚀 Key Features

### 🌱 Crop Recommendation

The system predicts a suitable crop using nine numerical parameters:

-   Nitrogen (N)
-   Phosphorus (P)
-   Potassium (K)
-   Soil pH
-   Soil Moisture
-   Soil Temperature
-   Temperature
-   Humidity
-   Rainfall

### 🔬 Plant Disease Detection

The Plant Disease Detection module uses **Convolutional Neural Networks
(CNNs)** to classify plant diseases from leaf images using publicly
available datasets.

### 📡 IoT-Based Soil Monitoring

An **Arduino Nano** is used with soil sensors to collect:

-   Soil moisture
-   Soil temperature
-   Soil pH
-   Nitrogen
-   Phosphorus
-   Potassium

### 🌦️ Weather Data Integration

Environmental information is obtained using the **OpenWeatherMap API**,
including:

-   Temperature
-   Humidity
-   Rainfall

### 💬 AI Agricultural Chatbot

The Gemini API-powered chatbot supports natural-language questions
related to:

-   Crop recommendations
-   Soil conditions
-   Fertilizers
-   Plant diseases
-   Crop treatment
-   Farming practices
-   Suitable crops for particular soil conditions

## 🏗️ System Architecture

``` text
                    ┌─────────────────────┐
                    │       USER          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Web Frontend      │
                    │   HTML/CSS/JS       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌───────────────┐
      │ Arduino Nano │ │ Weather API  │ │ Gemini Chatbot │
      └──────┬───────┘ └──────┬───────┘ └───────┬───────┘
             │                │                 │
             ▼                ▼                 ▼
      ┌──────────────┐ ┌──────────────┐ ┌────────────────┐
      │ Soil Sensors │ │ Environmental│ │ Natural        │
      │              │ │ Data         │ │ Language Query │
      └──────┬───────┘ └──────┬───────┘ └───────┬────────┘
             │                │                 │
             └────────────────┼─────────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ Data Processing &   │
                    │ Validation          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Machine Learning    │
                    │ Classification      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Crop Recommendation │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Result Display      │
                    └─────────────────────┘
```

## 🔄 Project Workflow

``` text
Soil Sensors
     │
     ▼
Arduino Nano
     │
     ▼
Sensor Data
     │
     ├──────────────► Soil Moisture
     ├──────────────► Soil Temperature
     ├──────────────► Soil pH
     └──────────────► NPK
                         │
                         ▼
                  Frontend Interface
                         │
                         ▼
                 Weather API Data
                         │
                         ├── Temperature
                         ├── Humidity
                         └── Rainfall
                         │
                         ▼
                 Data Validation
                         │
                         ▼
                 Data Preprocessing
                         │
                         ▼
               Machine Learning Model
                         │
                         ▼
                Crop Prediction
                         │
                         ▼
                 Frontend Output
```

## 🤖 Machine Learning Module

The Crop Recommendation System is formulated as a **supervised
classification problem**.

### Algorithms Used

-   Decision Tree
-   Random Forest
-   K-Nearest Neighbors (KNN)
-   XGBoost

### Model Input

  -----------------------------------------------------------------------
  Feature                             Description
  ----------------------------------- -----------------------------------
  Nitrogen (N)                        Essential nutrient for plant growth

  Phosphorus (P)                      Supports root development and
                                      flowering

  Potassium (K)                       Supports plant health and disease
                                      resistance

  Temperature                         Influences growth and crop yield

  Humidity                            Influences transpiration and water
                                      availability

  pH                                  Represents soil acidity or
                                      alkalinity

  Rainfall                            Represents water availability

  Soil Temperature                    Influences microbial activity and
                                      nutrient availability

  Soil Moisture                       Represents soil water content
  -----------------------------------------------------------------------

### Model Output

The model produces a predicted crop label, such as:

-   Rice
-   Wheat
-   Maize
-   Muskmelon
-   Other crops included in the dataset

## 🦠 Plant Disease Detection

The Plant Disease Detection component uses **Convolutional Neural
Networks (CNNs)** to classify plant diseases from leaf images.

The module was developed using publicly available datasets. The project
report describes the earlier implementation as capable of classifying
plant diseases with reasonable accuracy while identifying model
optimization, scalability, and dataset diversity as areas for
improvement.

## 📡 IoT and Hardware Module

The hardware implementation is based on an **Arduino Nano**.

### Sensors Used

#### Soil Moisture Sensor

Measures the water content present in the soil.

#### DS18B20 Soil Temperature Sensor

Measures soil temperature.

#### Soil pH Sensor

Measures soil acidity or alkalinity on a scale of 0--14.

#### NPK Sensor

Measures:

-   Nitrogen (N)
-   Phosphorus (P)
-   Potassium (K)

The NPK sensor communicates with the Arduino Nano through an **RS485
Modbus interface**.

### Hardware Communication

``` text
Soil Moisture Sensor ──────┐
                           │
DS18B20 Temperature Sensor ├──► Arduino Nano
                           │
Soil pH Sensor ────────────┤
                           │
NPK Sensor ─► RS485 ───────┘
```

The hardware power system uses a 12V supply and an **LM7805 voltage
regulator** to provide a 5V supply for the required components.

## 💬 Chatbot Module

The AgroCare chatbot is integrated using the **Gemini API**.

### Chatbot Workflow

``` text
User Query
     │
     ▼
Frontend Chat Interface
     │
     ▼
Input Validation
     │
     ▼
Text Preprocessing
     │
     ├── Lowercase Conversion
     ├── Keyword Detection
     └── Topic Identification
     │
     ▼
Context & Chat History
     │
     ▼
System Prompt
     │
     ▼
Gemini LLM
     │
     ├── Tokenization
     ├── Embeddings
     ├── Transformer Processing
     └── Probability / Softmax
     │
     ▼
Generated Response
     │
     ▼
User
```

The chatbot is designed to:

-   Act as an Agricultural Expert Assistant
-   Provide short structured bullet points
-   Give step-by-step practical solutions
-   Focus on crop and farming advice

## 🧠 LLM Processing

The chatbot workflow includes:

1.  **Tokenization** -- Divides the user's text into tokens.
2.  **Embedding Creation** -- Converts tokens into numerical vector
    representations.
3.  **Transformer Processing** -- Processes relationships between
    tokens.
4.  **Softmax** -- Calculates probabilities for possible next tokens.
5.  **Output Generation** -- Converts generated tokens into readable
    text.

## 🛠️ Technologies Used

### Software

-   Python
-   Flask
-   HTML
-   CSS
-   JavaScript
-   Machine Learning
-   Deep Learning
-   Gemini API
-   OpenWeatherMap API

### Machine Learning

-   Decision Tree
-   Random Forest
-   KNN
-   XGBoost
-   CNN

### Hardware

-   Arduino Nano
-   Soil Moisture Sensor
-   DS18B20 Temperature Sensor
-   Soil pH Sensor
-   JXBS-3001 NPK Sensor
-   RS485 Module
-   LM7805 Voltage Regulator
-   Capacitors
-   PCB

## 📥 Input Parameters

``` text
1. Nitrogen (N)
2. Phosphorus (P)
3. Potassium (K)
4. Soil pH
5. Soil Moisture
6. Soil Temperature
7. Temperature
8. Humidity
9. Rainfall
```

## 📁 Project Structure

``` text
AgroCare/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── crop_model.pkl
│   └── plant_disease_model/
│
├── dataset/
│   └── crop_recommendation.csv
│
├── templates/
│   ├── index.html
│   └── chatbot.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── images/
│
├── hardware/
│   ├── arduino/
│   │   └── sensor_code.ino
│   ├── circuit/
│   └── pcb/
│
└── plant_disease/
    ├── dataset/
    ├── model/
    └── prediction.py
```

> **Note:** Update the structure above if the actual filenames and
> folders in the repository are different.

## ⚙️ Installation

### 1. Clone the Repository

``` bash
git clone https://github.com/<your-username>/AgroCare.git
cd AgroCare
```

### 2. Create a Virtual Environment

``` bash
python -m venv venv
```

#### Windows

``` bash
venv\Scripts\activate
```

#### Linux / macOS

``` bash
source venv/bin/activate
```

### 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

## 🔑 Configuration

Create a `.env` file for the required API credentials:

``` env
GEMINI_API_KEY=your_gemini_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
```

**Important:** Never upload API keys or other secret credentials to
GitHub.

Add the following to `.gitignore`:

``` text
.env
venv/
__pycache__/
*.pyc
```

## ▶️ Running the Project

After installing the dependencies and configuring the required API keys:

``` bash
python app.py
```

Open the local URL displayed by Flask in your browser.

## 🔄 How the System Works

### Step 1 --- Collect Soil Data

The Arduino Nano receives data from the soil sensors:

-   Soil moisture
-   Soil temperature
-   pH
-   NPK

### Step 2 --- Collect Environmental Data

The system retrieves:

-   Temperature
-   Humidity
-   Rainfall

from the weather API.

### Step 3 --- Data Processing

The collected values are validated and formatted.

### Step 4 --- Machine Learning Prediction

The trained classification model receives the nine numerical features
and predicts a suitable crop.

### Step 5 --- Display Prediction

The predicted crop is returned to the frontend and displayed to the
user.

### Step 6 --- Chatbot Assistance

Users can ask agriculture-related questions through the chatbot and
receive AI-generated guidance.

## 📊 Results

The project demonstrates the integration of:

-   IoT-based soil sensing
-   Weather API data
-   Machine learning-based crop prediction
-   Web-based interaction
-   AI-powered agricultural assistance

The system provides a unified platform for data-driven agricultural
recommendations and interactive farming assistance.

## 🔮 Future Scope

Future improvements include:

-   📡 Wireless sensor data transmission
-   📍 GPS-based location-aware recommendations
-   ☁️ Cloud and IoT middleware integration
-   🤖 Fully automated crop prediction
-   💧 Automated irrigation
-   📊 Real-time monitoring dashboard
-   📱 Mobile application
-   ☀️ Solar-powered deployment
-   ⚡ Low-power operation and intelligent sensor scheduling

These improvements can transform the current system into a fully
automated, location-aware, scalable smart agriculture platform.

## 👨‍💻 Team Members

  Name                     Role
  ------------------------ ---------------------
  **Upayan Chakraborty**   Project Team Member
  **Sandipan Rakshit**     Project Team Member
  **Prerana Maiti**        Project Team Member
  **Mrigangana Sarkar**    Project Team Member
  **Souvik Ghosh**         Project Team Member

## 👨‍🏫 Supervisor

**Prof. (Dr.) Subro S. Thakur**\
Professor\
Department of Computer Science & Engineering\
MCKV Institute of Engineering

## 🎓 Institution

**MCKV Institute of Engineering**\
Department of Computer Science & Engineering

Affiliated to:

**Maulana Abul Kalam Azad University of Technology, West Bengal**

## 📚 References

1.  Ferentinos, D.P. (2018). *Deep learning models for plant disease
    detection*. Computers and Electronics in Agriculture.
2.  Patil, A. et al. (2019). *Plant disease detection using machine
    learning techniques*.
3.  Singh, V. et al. (2020). *Crop disease classification using Random
    Forest*.
4.  Rajeswari, S. et al. (2022). *IoT-based smart agriculture monitoring
    system*.
5.  [Arduino Nano Documentation](https://docs.arduino.cc/hardware/nano/)
6.  [DS18B20
    Datasheet](https://cdn.sparkfun.com/datasheets/Sensors/Temp/DS18B20.pdf)

## 🌱 Conclusion

AgroCare demonstrates how **IoT, Machine Learning, Deep Learning, and
Generative AI** can be combined to address real-world agricultural
challenges.

By collecting soil parameters through sensors, retrieving environmental
information through APIs, processing the data using machine learning,
and providing conversational assistance through an AI chatbot, AgroCare
provides a unified platform for intelligent agricultural
decision-making.

------------------------------------------------------------------------

### ⭐ Project Highlights

``` text
🌱 Smart Crop Recommendation
🦠 Plant Disease Detection
📡 IoT Soil Monitoring
🤖 Machine Learning
🧠 Deep Learning / CNN
💬 Gemini AI Chatbot
🌦️ Weather API Integration
🔌 Arduino Nano
📊 Agricultural Data Analysis
🌾 Smart Agriculture
```

------------------------------------------------------------------------

## 📜 License

This project was developed as an academic B.Tech project.

If you reuse or modify this project, please provide appropriate
attribution to the original authors.

**Made with 🌱 + 🤖 + 💻 for Smart Agriculture**
