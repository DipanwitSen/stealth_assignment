# ⚽ Football Video Analytics with Player Tracking, Team Clustering, and Ball Possession

This project is a computer vision system that detects and tracks players, referees, goalkeepers, and the ball in a soccer match video.
It assigns team IDs using clustering, determines ball possession, and annotates the video with rich analytics such as control percentage and speed.

## 📂 Project Structure

soccer_tracking/
├── main.py # Entry point to run the tracking pipeline
├── tracker.py # Tracker class with detection, tracking, clustering, and annotation logic
├── video_utils.py # Functions for reading and saving video frames
├── utils/
│ └── bbox_utils.py # Utility functions for bounding box processing and distance calculation
├── model/
│ └── best.pt # Trained YOLOv8 model weights
├── input_video.mp4 # Sample input soccer video
├── output_video.mp4 # Processed output video with annotations
├── requirements.txt # Python dependencies
└── README.md # Documentation and setup instructions

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
## 📚 References & Resources

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
  State-of-the-art object detection framework used for detecting players, referees, balls, and goalkeepers.

- [ByteTrack Tracker](https://github.com/ifzhang/ByteTrack)  
  A high-performance multi-object tracker used to maintain consistent IDs across frames.

- [Supervision Library by Roboflow](https://github.com/roboflow/supervision)  
  Middleware to convert YOLO detections into a format compatible with ByteTrack and enable structured tracking.

- [Scikit-learn KMeans Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)  
  Used to group detected players into 2 teams based on their movement patterns and locations.

- [OpenCV](https://opencv.org/)  
  Used for reading input video frames, drawing bounding boxes, shapes, and writing the output video.

- **Video Dataset**:  
  Public match footage from open-access platforms such as:  
  - [FIFA Open Data](https://www.fifa.com/technical/football-technology/standards-and-data/data)  
  - [YouTube Match Footage](https://www.youtube.com/results?search_query=soccer+match+full+game)

📈 Performance Notes
Processing time depends on the number of frames and batch size
The closer the ball is to a player’s foot position, the more accurate the ball possession attribution
KMeans works best with clearer team separation; jersey color-based team detection could be a future enhancement

---------------

## 🚀 Future Improvements

To further enhance the soccer analytics system, the following improvements can be considered:

- **Integrate Player Jersey Number OCR**  
  Use Optical Character Recognition (OCR) to extract jersey numbers from players for accurate and consistent identification, especially helpful during player substitutions or occlusions.

- **Adopt Advanced Re-Identification Models**  
  Replace clustering with LSTM or GRU-based ReID models to maintain player identity across longer temporal windows and occlusions.

- **Incorporate Jersey Color-Based Team Classification**  
  Improve team assignment accuracy by detecting dominant jersey colors instead of relying on unsupervised clustering methods.

- **Add Action Recognition**  
  Leverage temporal Convolutional Neural Networks (CNNs) or transformers to recognize high-level actions like passes, shots, tackles, and dribbles.

- **Enable Real-Time Inference and Dashboard Integration**  
  Optimize the pipeline for real-time analysis and visualize results live via a web-based dashboard using frameworks like Flask or FastAPI.

- **Support Full-Match Video Processing with Event Tagging**  
  Extend current short-clip support to full 90-minute matches, and auto-tag events like goals, fouls, substitutions, and corner kicks.

- **Enhance Ball Possession Logic with Temporal Consistency**  
  Use frame-wise historical tracking and context to more reliably infer which player controls the ball across time.

- **Track Player Speed and Movement Patterns**  
  Calculate speed by measuring positional change across frames, and visualize player heatmaps and movement trends.

- **Export Analytics Results**  
  Allow CSV export of statistics and generate visual plots (e.g., ball possession chart, player speed graph, pass network).

These enhancements can greatly elevate the system from a basic tracker to a comprehensive sports analytics platform.

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




