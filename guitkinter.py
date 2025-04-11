import tkinter as tk
from tkinter import filedialog, Label, Button, Text, Scrollbar
from PIL import Image, ImageTk
import threading
from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from tensorflow.keras.preprocessing import image
from difflib import get_close_matches


# === CONFIGURATION ===
IMAGE_SIZE = (224, 224)
FRUIT_MODEL_PATH = r"F:\\Downloads\\FR4model.keras"
LABELS_PATH = "labels.txt"
CSV_PATH = r"receipe.csv"
API_KEY = "62a1cb55dab2e669bff53381b95e6649"  # OpenWeather API key

# === LOAD MODEL AND LABELS ===
try:
    fruit_ripeness_model = tf.keras.models.load_model(FRUIT_MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

try:
    with open(LABELS_PATH, 'r') as f:
        class_names = [line.strip().title() for line in f.readlines()]
    print(f"✅ Loaded {len(class_names)} labels")
except Exception as e:
    print(f"❌ Error loading labels: {e}")
    exit()


# === RECIPE DATASET HANDLING ===
def load_recipe_data():
    try:
        # First, read the CSV to determine its structure
        df = pd.read_csv(CSV_PATH)
        
        # Check if the necessary columns exist, if not create them
        required_columns = ['fruit_type', 'season', 'temperature', 'weather', 
                           'dish_type', 'preferred_dish_name', 'fruit_condition']
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = ""  # Add empty column if missing
        
        # Clean all string columns
        str_cols = df.select_dtypes(include=['object']).columns
        df[str_cols] = df[str_cols].apply(lambda x: x.str.lower().str.strip())
        
        # Create a default dataset if the CSV is empty or couldn't be loaded properly
        if df.empty:
            print("Creating default recipe dataset...")
            data = {
                'fruit_type': ['banana', 'banana', 'banana', 'apple', 'apple', 'apple',
                              'mango', 'mango', 'mango', 'orange', 'orange', 'orange'],
                'season': ['all', 'all', 'all', 'all', 'all', 'all', 
                          'all', 'all', 'all', 'all', 'all', 'all'],
                'temperature': ['hot', 'cold', 'moderate', 'hot', 'cold', 'moderate',
                               'hot', 'cold', 'moderate', 'hot', 'cold', 'moderate'],
                'weather': ['humid', 'humid', 'humid', 'dry', 'rainy', 'sunny',
                           'humid', 'rainy', 'sunny', 'dry', 'rainy', 'sunny'],
                'dish_type': ['snack', 'drink', 'dessert', 'snack', 'dessert', 'salad',
                             'dessert', 'drink', 'fresh', 'juice', 'dessert', 'salad'],
                'preferred_dish_name': ['banana chips', 'banana smoothie', 'banana pudding',
                                       'baked apple', 'apple pie', 'apple salad',
                                       'mango ice cream', 'mango milkshake', 'fresh mango',
                                       'orange juice', 'orange cake', 'orange salad'],
                'fruit_condition': ['unripe', 'ripe', 'overripe', 'ripe', 'overripe', 'ripe',
                                   'ripe', 'ripe', 'ripe', 'ripe', 'overripe', 'ripe']
            }
            df = pd.DataFrame(data)
        
        return df
    except Exception as e:
        print(f"❌ Error loading recipe data: {e}")
        # Create a minimum default dataset
        data = {
            'fruit_type': ['banana', 'banana', 'banana', 'apple', 'apple'],
            'season': ['all', 'all', 'all', 'all', 'all'],
            'temperature': ['hot', 'cold', 'moderate', 'hot', 'cold'],
            'weather': ['humid', 'humid', 'humid', 'dry', 'rainy'],
            'dish_type': ['snack', 'drink', 'dessert', 'snack', 'dessert'],
            'preferred_dish_name': ['banana chips', 'banana smoothie', 'banana pudding',
                                  'baked apple', 'apple pie'],
            'fruit_condition': ['unripe', 'ripe', 'overripe', 'ripe', 'overripe']
        }
        return pd.DataFrame(data)

recipe_df = load_recipe_data()
print(f"📊 Loaded {len(recipe_df)} recipes")

# === FUZZY TEMPERATURE LOGIC ===
def compute_membership(diff):
    scores = {'cold': 0.0, 'cool': 0.0, 'moderate': 0.0, 'warm': 0.0, 'hot': 0.0}

    # Cold: ≤ -6 (1.0), fades to 0 at -4
    if diff <= -6:
        scores['cold'] = 1.0
    elif -6 < diff <= -4:
        scores['cold'] = (-4 - diff) / 2  # 1.0 at -6 → 0.0 at -4

    # Cool: peaks at -4 (1.0), spans -6 to 0 (overlaps Mild)
    if -6 <= diff <= -4:
        scores['cool'] = (diff + 6) / 2  # 0.0→1.0 (-6 to -4)
    elif -4 < diff <= 0:
        scores['cool'] = (-diff) / 4      # 1.0→0.0 (-4 to 0)

    # Moderate: peaks at 0 (1.0), spans -4 to +4 (overlaps Cool & Warm)
    if -4 <= diff <= 0:
        scores['moderate'] = (diff + 4) / 4   # 0.0→1.0 (-4 to 0)
    elif 0 < diff <= 4:
        scores['moderate'] = (4 - diff) / 4    # 1.0→0.0 (0 to +4)

    # Warm: peaks at +4 (1.0), spans 0 to +6 (overlaps Mild & Hot)
    if 0 <= diff <= 4:
        scores['warm'] = diff / 4          # 0.0→1.0 (0 to +4)
    elif 4 < diff <= 6:
        scores['warm'] = (6 - diff) / 2    # 1.0→0.0 (+4 to +6)

    # Hot: ≥ +6 (1.0), starts at +4 (overlaps Warm)
    if 4 <= diff < 6:
        scores['hot'] = (diff - 4) / 2    # 0.0→1.0 (+4 to +6)
    elif diff >= 6:
        scores['hot'] = 1.0

    return scores

def fuzzy_label_from_scores(scores):
    return max(scores, key=scores.get)

def adaptive_temp_label(state, temp_celsius):
    avg = state_avg_temps.get(state)
    if avg is None:
        return "moderate", 0, {"moderate": 1.0}  # Default fallback
    
    diff = temp_celsius - avg
    scores = compute_membership(diff)
    return fuzzy_label_from_scores(scores), diff, scores

def get_location():
    try:
        response = requests.get('https://ipinfo.io/json', timeout=5)
        data = response.json()
        city = data.get('city', 'Unknown')
        region = data.get('region', 'Delhi')  # Default to Delhi if unknown
        loc = data.get('loc', '28.6139,77.2090')  # Default to Delhi coords
        return loc, city, region
    except Exception as e:
        print(f"⚠️ Location detection error: {e}")
        return "28.6139,77.2090", "Unknown", "Delhi"  # Default fallback

def get_temperature(lat, lon, api_key):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        data = response.json()
        temp = data['main']['temp']
        weather = data['weather'][0]['main'].lower()
        return temp, weather
    except Exception as e:
        print(f"⚠️ Weather API error: {e}")
        return 25, "clear"  # Default fallback

def plot_membership(scores):
    plt.figure(figsize=(8, 4))
    labels = list(scores.keys())
    values = list(scores.values())
    plt.bar(labels, values, color='skyblue')
    plt.ylim(0, 1.1)
    plt.title("Temperature Classification")
    plt.xlabel("Temperature Category")
    plt.ylabel("Membership Score")
    plt.grid(True, axis='y')
    plt.show()

# === IMPROVED RECIPE RECOMMENDER ===
def recommend_recipe(fruit_type, fruit_condition, temperature, weather):
    # Normalize inputs
    fruit_type = fruit_type.lower().strip()
    fruit_condition = fruit_condition.lower().strip()
    temperature = temperature.lower().strip()
    weather = weather.lower().strip()
    #temperature = temperature.lower().strip()
    #weather = weather.lower().strip()

    print("\n🔍 Searching recipes for:")
    print(f"  Fruit: {fruit_type.title()} ({fruit_condition})")
    print(f"  Weather: {weather.title()}, {temperature.title()}")

    # Check for exact match first
    exact_match = recipe_df[
        (recipe_df['fruit_type'] == fruit_type) &
        (recipe_df['fruit_condition'] == fruit_condition) &
        (recipe_df['temperature'] == temperature) &
        (recipe_df['weather'] == weather)
    ]

    if not exact_match.empty:
        dish = exact_match.sample(1).iloc[0]['preferred_dish_name'].title()
        print(f"✅ Found perfect match!")
        return dish

    # Find most similar match using a scoring system
    def similarity_score(row):
        score = 0
        # Fruit type is most important
        if row['fruit_type'] == fruit_type:
            score += 10
        # Fruit condition is next most important
        if row['fruit_condition'] == fruit_condition:
            score += 5
        elif row['fruit_condition'] == 'all':
            score += 3
        # Temperature and weather are less important
        if row['temperature'] == temperature:
            score += 2
        if row['weather'] == weather:
            score += 2
        return score

    # Apply scoring to all recipes
    recipe_df['score'] = recipe_df.apply(similarity_score, axis=1)
    
    # Get the best matches
    best_matches = recipe_df[recipe_df['score'] > 0].sort_values('score', ascending=False)
    
    if not best_matches.empty:
        top_match = best_matches.iloc[0]
        dish = top_match['preferred_dish_name'].title()
        print(f"⚠️ Found most similar match (similarity score: {top_match['score']})")
        return dish

    # Ultimate fallback
    generic_suggestions = {
        'unripe': 'make chips or cook as vegetable',
        'ripe': 'eat fresh or make smoothie',
        'overripe': 'use in baking or pudding'
    }
    suggestion = generic_suggestions.get(fruit_condition, 'use in cooking')
    return f"{suggestion.title()} (no specific recipe found)"

# === IMAGE PREDICTION === (REPLACED AS REQUESTED)
def predict_fruit_and_ripeness(image_path):
    try:
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = fruit_ripeness_model.predict(img_batch)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[predicted_class_index]
        confidence = prediction[0][predicted_class_index]

        # Display results
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"{predicted_class} ({confidence * 100:.2f}%)", pad=20)
        plt.axis('off')
        plt.show()

        # Parse the label
        if '_' in predicted_class:
            parts = predicted_class.split('_')
            fruit = parts[0].lower()
            condition = parts[1].lower() if len(parts) > 1 else 'ripe'
        else:
            # Unexpected format, split by whitespace as fallback
            parts = predicted_class.split()
            if len(parts) > 1:
                fruit = parts[0].lower()
                condition = parts[1].lower()
            else:
                fruit = predicted_class.lower()
                condition = 'ripe'  # Default

        # Display detection results separately as requested
        print(f"\n🔍 Model prediction: {predicted_class}")
        print(f"🍓 Detected fruit type: {fruit.title()}")
        print(f"🥝 Detected fruit condition: {condition}")
        
        return fruit, condition

    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return None, None

# === WEATHER INPUT WITH AUTOMATIC OR MANUAL OPTIONS ===
def get_weather_input(default = 1):
    print("\n" + "-"*40)
    print(" WEATHER CONDITIONS ".center(40))
    print("-"*40)
    
    print("\nHow would you like to provide weather information?")
    print("1. Detect automatically (requires internet)")
    print("2. Enter manually")
    if(default):
        choice = input("Your choice (1/2): ").strip()
    else:
        print
        choice='1'

    if (choice=='1'):
        try:
            print("\n🔍 Detecting your location and weather...")
            location, city, region = get_location()
            lat, lon = map(float, location.split(','))
            temperature, weather_condition = get_temperature(lat, lon, API_KEY)
            
            print(f"📍 Location detected: {city}, {region}")
            print(f"🌡️ Current temperature: {temperature:.1f}°C")
            print(f"☁️ Current weather: {weather_condition.title()}")
            
            temp_label, diff, scores = adaptive_temp_label(region, temperature)
            
            print(f"📊 Temperature deviation from regional average: {diff:.1f}°C")
            print(f"🔍 Adaptive temperature classification: {temp_label.title()}")
            
            # Plot the fuzzy membership scores
            plot_membership(scores)
            
            return temp_label, weather_condition
            
        except Exception as e:
            print(f"⚠️ Error in automatic detection: {e}")
            print("Switching to manual input...")
            return get_manual_weather_input()
    else:
        return get_manual_weather_input()

def get_manual_weather_input():
    while True:
        try:
            degree = float(input("\nEnter current temperature in °C: "))
            break
        except ValueError:
            print("⚠️ Please enter a valid number (e.g., 23.5)")
    
    state = input("Enter your state (for temperature context): ").strip()
    
    if state in state_avg_temps:
        temp_label, diff, scores = adaptive_temp_label(state, degree)
        print(f"📊 Temperature deviation from state average: {diff:.1f}°C")
        print(f"🔍 Adaptive temperature classification: {temp_label.title()}")
        plot_membership(scores)
    else:
        # Simple fallback if state not in dataset
        if degree <= 15:
            temp_label = "cold"
        elif degree <= 25:
            temp_label = "moderate"
        else:
            temp_label = "hot"
        print(f"🔍 Temperature classified as: {temp_label.title()}")
    
    weather = input("Weather Condition (Sunny/Rainy/Humid/etc): ").strip().lower()
    return temp_label, weather


class FruitSnapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FruitSnap 🍓📸")
        self.root.configure(bg="#f0f2f5")

        self.canvas = tk.Canvas(root, width=300, height=300, bg="#ffffff", bd=0, highlightthickness=0)
        self.canvas.pack(pady=(20, 10))

        self.upload_btn = Button(root, text="📤 Upload Fruit Image", command=self.upload_image,
                                 bg="#10a37f", fg="white", font=("Segoe UI", 11, "bold"),
                                 relief="flat", padx=10, pady=5, bd=0, activebackground="#0e8a6a")
        self.upload_btn.pack(pady=5)

        self.result_frame = tk.Frame(root, bg="#f0f2f5")
        self.result_frame.pack(pady=10, fill="both", expand=True)

        self.result_text = Text(self.result_frame, height=10, wrap="word", font=("Consolas", 11),
                                bg="#ffffff", fg="#333333", relief="flat", padx=10, pady=10)
        self.result_text.pack(side="left", fill="both", expand=True)

        self.scrollbar = Scrollbar(self.result_frame, command=self.result_text.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.result_text.config(yscrollcommand=self.scrollbar.set)

        self.result_text.insert("end", "🤖 Welcome to FruitSnap! Upload a fruit image to begin...\n")
        self.result_text.config(state="disabled")

    def upload_image(self):
        filepath = filedialog.askopenfilename()
        if not filepath:
            return

        self.result_text.config(state="normal")
        self.result_text.insert("end", "🕐 Processing image...\n")
        self.result_text.config(state="disabled")

        threading.Thread(target=self.process_image, args=(filepath,)).start()

    def process_image(self, path):
        try:
            # Show image
            img = Image.open(path)
            img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(150, 150, image=img_tk)
            self.canvas.image = img_tk

            # Model processing
            img_model = image.load_img(path, target_size=IMAGE_SIZE)
            x = image.img_to_array(img_model) / 255.0
            x = np.expand_dims(x, axis=0)

            pred = fruit_ripeness_model.predict(x)
            pred_index = np.argmax(pred)
            predicted_class = class_names[pred_index]
            fruit_name, fruit_state = predicted_class.split("_")

            # Get location, weather
            loc, city, region = get_location()
            temp, weather = get_weather_input(0)
            recommendation = recommend_recipe(fruit_name, fruit_state, temp, weather)

            result = (
                f"\n📍 Location: {city}, {region}"
                f"\n🌡️ Temperature: {temp}"
                f"\n🌤️ Weather: {weather.title()}"
                f"\n\n🍎 Fruit Detected: {fruit_name.title()} ({fruit_state.title()})"
                f"\n👨‍🍳 Recommended Recipe: {recommendation}\n"
            )

            self.result_text.config(state="normal")
            self.result_text.insert("end", result)
            self.result_text.config(state="disabled")
            self.result_text.see("end")


        except Exception as e:
            import traceback
            self.result_text.config(state="normal")
            self.result_text.insert("end", f"\n❌ Error: {e}\n{traceback.format_exc()}")
            self.result_text.config(state="disabled")

    def show_plot(self, scores):
        buf = BytesIO()
        plot_membership(scores)
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_plot = Image.open(buf)
        img_plot.thumbnail((300, 150))
        img_tk = ImageTk.PhotoImage(img_plot)
        self.canvas.create_image(150, 275, image=img_tk)
        self.canvas.image = img_tk

# Run App
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("500x600")
    app = FruitSnapApp(root)
    root.mainloop()
