# Upper Limb Exoskeleton Real-Time Detection

This project provides a **real-time human intention recognition system** for an upper limb exoskeleton.  
It uses two pre-trained models:
- **YOLO** model for detecting painting tools and hands.
- **MoViNet + LSTM fusion model** for recognizing user actions based on both video frames and IMU data.  

Both models are trained on a custom **Paint Database** (link to be added).  
The system outputs:
- **Action label:** one of 10 classes (e.g., `Bimanual_Up`, `Bimanual_Right`, `Unimanual_Down`, `Prepare`).
- **Detected tool:** painting brush, roller, or none.

## How to Run Real-Time Detection

1. Place the following pre-trained models in the `Pre Trained Model/` folder:
   - Fusion model (`fusion_movinet_final.pt`)
   - YOLO model (`best.pt`)
2. Start the real-time prediction script:

```bash
python Local_Prediction.py
```

3. In another terminal, start collecting IMU and camera data:

```bash
python imu_data_collection.py
```

The system will read IMU and video data in real time and output the detected action and tool.
