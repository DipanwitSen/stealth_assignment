# ⚽ Football Video Analytics with Player Tracking, Team Clustering, and Ball Possession

This project is a computer vision system that detects and tracks players, referees, goalkeepers, and the ball in a soccer match video.
It assigns team IDs using clustering, determines ball possession, and annotates the video with rich analytics such as control percentage and speed.

## 📂 Project Structure
soccer_tracking/
├── main.py
├── tracker.py
├── video_utils.py
├── utils/
│ └── bbox_utils.py
├── model/
│ └── best.pt
├── input_video.mp4
├── output_video.mp4
├── requirements.txt
└── README.md

## 📌 Introduction

This project focuses on **soccer analytics** using computer vision. The aim was to detect and track players, referees, goalkeepers, and the ball, and determine **ball possession and team control** across video frames. The system uses:
- YOLOv8 for detection
- ByteTrack for object tracking
- KMeans for team clustering
- OpenCV + custom logic for annotation and visualization

---

## 🧠 Approach and Methodology

### 1. Detection:
- YOLOv8 custom-trained on soccer classes (`player`, `referee`, `ball`, `goalkeeper`)
- Batch inference across frames using Ultralytics API

### 2. Tracking:
- Supervision's ByteTrack used to associate detections across frames
- Track IDs help identify the same object over time

### 3. Team Assignment:
- KMeans clustering used to assign players into **Team 1** and **Team 2** based on movement patterns

### 4. Ball Possession:
- Euclidean distance used between player foot position and ball center
- Closest player is assumed to have possession

### 5. Annotation:
- Ellipses for players, triangles for the ball, rectangles for goalkeepers
- Ball possession and control stats shown using overlays

---

## 🧪 Techniques Tried (with Results)

### ✅ Successful Techniques

| Technique | Outcome |
|----------|---------|
| YOLOv8 with Ultralytics | High-speed, accurate multi-class detection |
| ByteTrack from Supervision | Robust multi-object tracking |
| KMeans Clustering | Allowed distinguishing teams based on trajectory |
| Distance-based ball possession | Efficient, interpretable results |
| Annotated overlays | Clear visualization of player/team control |

### ❌ Failed/Challenging Techniques

| Attempted | Issue |
|----------|-------|
| Deep ReID/ResNet-based player identity | Model loading issues, large weight size |
| Optical Flow for movement prediction | Overhead was too high for short clips |
| Using embeddings for clustering | Not enough training data, too complex |

---

## ⚠️ Challenges Encountered

- **Inconsistent tracking IDs** due to occlusion or misdetections
- **Ball flickering** when it disappeared momentarily (solved with interpolation)
- Difficulty in **distinguishing referees vs players** if color overlap exists
- Tuning clustering and tracking hyperparameters was time-consuming
- Large videos required **memory-efficient batch processing**

---

## ✅ Final Outcome

- A complete system that:
  - Detects and tracks soccer entities across frames
  - Annotates frames with team identities and ball control
  - Saves the result as a video with real-time overlays
- Works well on short clips (~15–30 seconds)

---

## 🚀 Future Improvements

- Integrate **player jersey number OCR** to enhance identification
- Use **LSTM or GRU-based ReID models** for more consistent player tracking
- Add **action recognition** (e.g., pass, shot, tackle) using temporal CNNs
- Enable real-time inference and dashboard integration
- Extend support to full-match processing with automatic event tagging

---

## 🛠️ Tools & Technologies Used

- Python, OpenCV
- Ultralytics YOLOv8
- Supervision Library (ByteTrack)
- Scikit-learn (KMeans)
- VS Code / Jupyter Notebook

---

## 🛠️ Features

- YOLOv8-based multi-object detection (`player`, `goalkeeper`, `referee`, `ball`)
- ByteTrack-based tracking for consistent IDs
- KMeans-based team clustering
- Ball possession logic
- Annotated output video with ellipse, rectangle, and triangle visuals
- Real-time team ball control statistics

---

## 💻 Prerequisites

- Python 3.8 or higher
- Visual Studio Code (or any IDE)
- GPU (optional but recommended for faster YOLOv8 inference)

---

## ✅ Installation & Setup (Step-by-Step)

### 1. Clone or Download the Repository

```bash
git clone https://github.com/DipanwitSen/stealth_assignment.git
cd stealth_assignment

2. Install Python Environment

Make sure Python 3.8+ is installed. If not:
Windows: Download Python
Linux/macOS: Use sudo apt install python3

Create a virtual environment:
python -m venv venv
Activate it:

Windows:
venv\Scripts\activate
macOS/Linux:

source venv/bin/activate

3. Install Required Packages
Install all dependencies:

pip install -r requirements.txt
If requirements.txt is missing, here are the core packages:
pip install ultralytics opencv-python pandas scikit-learn supervision
⚠️ Make sure ultralytics is correctly installed to access YOLOv8.

📦 YOLOv8 Model Weights
Place your custom-trained best.pt YOLOv8 model inside the model/ folder.
If you don’t have one, you can use a pretrained YOLOv8 model for demo (e.g., yolov8n.pt):
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

🧪 Running the Code
```bash
# Clone repo
git clone https://github.com/DipanwitSen/stealth_assignment.git
cd stealth_assignment

# Create environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the code
python main.py --input video.mp4 --output annotated_video.mp4


Console logs:
Frame 12
Team 1 avg dist to ball: 42.8
🎯 Ball Possession: Player 7 (Team 1)

📸 Sample Output Frame
🔴 Player (Ellipse, Team 1)
🔵 Player (Ellipse, Team 2)
⚪ Goalkeeper (Rectangle)
🟡 Referee (Ellipse)
🟣 Ball (Triangle)
📊 Team 1 Ball Control: 53.21%
📊 Team 2 Ball Control: 46.79%

🧠 Tips
For better results, fine-tune your YOLO model with soccer-specific datasets.
Set conf=0.1 or lower for detecting smaller objects like the ball.
Use tracker.get_object_tracks(read_from_stub=True) to avoid re-processing video.

🧪 Dependencies
Copy
Edit
opencv-python
pandas
scikit-learn
ultralytics
supervision

🔗 Author
Dipanwita Sen
GitHub: @DipanwitSen
Email: 22052204@kiit.ac.in
Project: stealth_assignment
Year: 2025




