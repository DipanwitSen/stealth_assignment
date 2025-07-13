# ⚽ Soccer Video Analytics with Player Tracking, Team Clustering, and Ball Possession

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
1. Put Input Video
Place your input video in the root directory and rename it as input_video.mp4 (or change it in main.py).
2. Run from VS Code Terminal
Open folder in VS Code and press Ctrl + ~ to open terminal.
Run the script:
python main.py
3. Output
output_video.mp4 → Annotated soccer video with:
Tracked players and teams
Ball position and possession

Ball control stats in real time

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

📞 Contact
For queries, contact
gmail: 22052204@kiit.ac.in
phone no:7439711097




