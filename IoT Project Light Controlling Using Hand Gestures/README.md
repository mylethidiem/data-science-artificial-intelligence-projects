# IoT Project: Light Controlling Using Hand Gestures

## Overview
This project applies [Google MediaPipe technology](https://ai.google.dev/edge/mediapipe/solutions/guide) for 
[hand gesture recognition](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md), 
enabling **real-time light control** through **Modbus RTU RS485**.  
The system is designed with two modes: **simulation mode** (tested in this project) and **real hardware mode** (conceptual, not yet tested).

---

## Step 1: Preparing Data and Environment
1. **Set up the work environment**
   - Create a new Conda environment with Python 3.10:
     ```bash
     conda create -n gesture_env python=3.10.0
     conda activate gesture_env
     ```
   - Install required libraries:
     ```bash
     pip install -r requirements.txt
     ```

2. **Configure Gesture Classes**
   - Define and customize gesture classes in `hand_gesture.yaml`.

3. **[Optional] Generate Landmark Data**
   - Run the script to collect landmark data:
     ```bash
     python generate_landmark_data.py
     ```

---

## Step 2: Building and Training the Hand Gesture Model
1. Load landmark data (collected or generated).  
2. Build an **MLP model** for gesture classification.  
3. Train the model.  
4. Evaluate model performance.  

👉 Training process is implemented in:  
`hand_gesture_recognition.ipynb`

---

## Step 3: Implementing Gesture-Based Light Control
1. **Goals**
   - Real-time gesture recognition: use webcam input to detect hand gestures.  
   - Gesture-based light control with two modes:  
     - **Simulation mode**: toggle light bulbs in a virtual environment (✔ tested in this project).  
     - **Real hardware mode**: designed for physical control of bulbs using relay modules and **Modbus RTU RS485** (🚧 not tested due to lack of hardware).  

2. **Operating Modes**
   - In the code, set:
     - `device = False` → simulation mode (default, tested).  
     - `device = True` → hardware mode (conceptual, not tested).  

3. **[Optional] Hardware Setup (Future Work)**
   - 4 Modbus RTU RS485 communication relays  
   - USB-to-RS485 converter  
   - 3 bulbs and lamp holders  
   - Run simulation mode with:
     ```bash
     python detect_simulation.py
     ```

---

## Tools & Libraries Used
- Python  
- MediaPipe  
- OpenCV  
- NumPy, Matplotlib

---

## Notes
- ✅ Current implementation has been tested only in **simulation mode**.  
- 🚧 Real hardware integration is a planned extension for future work.
