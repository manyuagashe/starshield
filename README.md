# StarShield - NASA CAD Asteroid Risk Prediction System

A production-ready asteroid risk prediction system built on real NASA JPL Close Approach Data (CAD).



### Setup

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

## 2. Process NASA Data (Already Done)
```bash
# Convert NASA CAD data to training format
python convert_cad_data.py
```

## 3. Train Model (Already Done)
```bash
# Train RandomForest model on NASA data
python train_model.py
```

## 4. Start API
```bash
# Start the FastAPI server
python asteroid_api.py
```

The API will be available at: http://localhost:8000

## 5. Setup Frontend and Run
open a new terminal window
```bash
cd frontend; npm i
npm run dev
```
The frontend will be available at: http://localhost:8080


Made by Vidur Shah, Nicolas Asanov, Arya Venkatesan, Abhimanyu Agashe
as part of Carolina Data Challenge 2025
