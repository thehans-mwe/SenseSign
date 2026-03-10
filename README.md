# SenseSign

**Multi-Modal ASL Translator & Accessibility Assistant**

SenseSign is an AI-powered system with **two modes**:

---

## Mode 1 – Camera-Based ASL Translation
- Captures hand gestures using a **camera**  
- Recognizes ASL signs in real time using deep learning models from [ASL Translator](https://github.com/Hamdan772/asl-translator)  
- Outputs translations as **text or speech**

---

## Mode 2 – Accessibility Mode for Visually Impaired Users
- Uses **TF-Luna LiDAR Range Finder (0.2m–8m, Single-Point, UART/I2C, 5V)** for depth and obstacle detection  
- AI answers questions such as:  
  - “Where am I?” → environment and obstacles  
  - “What is this?” → object recognition  
  - Reads **printed text** and **currency** aloud (OCR + TTS)  
- Serves as a **portable AI assistant**

---

## How It Works
1. **Data Capture:** Camera frames + LiDAR distance measurements  
2. **Preprocessing:** Extract hand shapes, movement, and 3D distance features  
3. **AI Analysis:**  
   - Mode 1: Classify ASL gestures  
   - Mode 2: Detect objects, read text/currency, interpret surroundings  
4. **Output:**  
   - Mode 1 → text/speech of ASL signs  
   - Mode 2 → spoken guidance and answers in real time  

---

## Features
- **Dual-Mode Operation:** ASL translation + accessibility assistant  
- **3D Spatial Awareness:** LiDAR provides real-time depth and obstacle detection  
- **Real-Time Feedback:** Instantly answers questions, reads text/currency, identifies objects  
- **Portable & Standalone:** Runs fully on Raspberry Pi 4  
- **Accessibility-Focused:** Assists deaf and visually impaired users  

---

## AI Prompt for Development
Create an AI system called SenseSign with two modes:

1) Camera-based ASL translation: use deep learning models to recognize hand gestures in real time and output text or speech  
2) LiDAR accessibility mode: use TF-Luna LiDAR sensor to detect objects, obstacles, read printed text and currency, and answer user questions audibly  

Include real-time processing, 3D spatial awareness, and mode switching via a button. Provide example code, integration with text-to-speech, and object/scene recognition suitable for Raspberry Pi 4. Ensure code is well-commented and modular for both modes.
