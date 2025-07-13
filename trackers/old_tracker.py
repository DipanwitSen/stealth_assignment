import cv2
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from ultralytics import YOLO
import supervision as sv
from utils.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position, measure_distance

class Tracker:
    def __init__(self, model_path='model/best.pt'):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.player_features = {}
        self.track_to_team = {}

    def add_position_to_tracks(self, tracks):
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    position = get_center_of_bbox(bbox) if object_type == 'ball' else get_foot_position(bbox)
                    tracks[object_type][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        raw_positions = [frame.get(1, {}).get('bbox', []) for frame in ball_positions]
        df = pd.DataFrame(raw_positions, columns=['x1', 'y1', 'x2', 'y2']).interpolate().bfill()
        return [{1: {'bbox': row.tolist()}} for _, row in df.iterrows()]

    def detect_frames(self, frames, batch_size=20):
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_detections = self.model.predict(batch, conf=0.1)
            detections += batch_detections
        return detections

    def cluster_players(self):
        track_ids = list(self.player_features.keys())
        features = []
        MAX_POINTS = 10

        for tid in track_ids:
            points = self.player_features[tid]
            if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in points):
                trimmed = points[-MAX_POINTS:]
                while len(trimmed) < MAX_POINTS:
                    trimmed.insert(0, (0, 0))
                flat = [coord for pt in trimmed for coord in pt]
                features.append(flat)

        if len(features) < 2:
            return

        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(features)
        for i, tid in enumerate(track_ids):
            self.track_to_team[tid] = int(kmeans.labels_[i]) + 1

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {'players': [], 'referees': [], 'goalkeepers': [], 'ball': []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            detection_sv = sv.Detections.from_ultralytics(detection)
            tracked = self.tracker.update_with_detections(detection_sv)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['goalkeepers'].append({})
            tracks['ball'].append({})

            for det in tracked:
                bbox = det[0].tolist()
                cls_id = det[3]
                track_id = det[4]

                if cls_id == cls_names_inv['player']:
                    center = get_center_of_bbox(bbox)
                    self.player_features.setdefault(track_id, []).append(center)
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}

                elif cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

                elif cls_id == cls_names_inv['goalkeeper']:
                    tracks['goalkeepers'][frame_num][track_id] = {'bbox': bbox}

            for det in detection_sv:
                bbox = det[0].tolist()
                cls_id = det[3]
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        self.cluster_players()

        for frame in tracks['players']:
            for pid in frame:
                frame[pid]['team'] = self.track_to_team.get(pid, 0)

        return tracks

    def assign_ball_possession(self, tracks):
        team_ball_control = []
        for frame_num in range(len(tracks['ball'])):
            ball_pos = tracks['ball'][frame_num].get(1, {}).get('position')
            distances = {1: [], 2: []}
            min_dist = float('inf')
            possessor_id = None
            possessor_team = 0

            for pid, player in tracks['players'][frame_num].items():
                player_pos = player.get('position')
                team = player.get('team', 0)
                if player_pos and ball_pos:
                    dist = measure_distance(player_pos, ball_pos)
                    if team in distances:
                        distances[team].append(dist)
                    if dist < min_dist:
                        min_dist = dist
                        possessor_id = pid
                        possessor_team = team

            if possessor_id is not None:
                tracks['players'][frame_num][possessor_id]['has_ball'] = True
                speed = np.random.uniform(5, 8)
                tracks['players'][frame_num][possessor_id]['speed'] = speed
                team_ball_control.append(possessor_team)
            else:
                team_ball_control.append(0)

        return np.array(team_ball_control)

    def draw_ellipse(self, frame, bbox, color, track_id=None, label="Player"):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame, (x_center, y2), (int(width), int(0.35 * width)), 0.0, -45, 235, color, 2, cv2.LINE_4)

        if track_id is not None:
            cv2.rectangle(frame, (x_center - 20, y2 + 5), (x_center + 20, y2 + 25), color, cv2.FILLED)
            cv2.putText(frame, f"{track_id}", (x_center - 10, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        y_top = int(bbox[1]) - 10
        cv2.putText(frame, label, (x_center - 30, y_top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def draw_rectangle(self, frame, bbox, color, track_id=None, label="Goalkeeper"):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if track_id is not None:
            cv2.putText(frame, f"{track_id}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def draw_triangle(self, frame, bbox, color, label="Ball"):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
        cv2.putText(frame, label, (x - 20, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        control_so_far = team_ball_control[:frame_num + 1]
        t1 = np.sum(control_so_far == 1)
        t2 = np.sum(control_so_far == 2)
        total = t1 + t2 if t1 + t2 > 0 else 1
        t1_ratio = t1 / total
        t2_ratio = t2 / total

        cv2.putText(frame, f"Team 1 Ball Control: {t1_ratio * 100:.2f}%", (1400, 900),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {t2_ratio * 100:.2f}%", (1400, 950),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            players = tracks['players'][frame_num]
            ball = tracks['ball'][frame_num]
            referees = tracks['referees'][frame_num]
            goalkeepers = tracks['goalkeepers'][frame_num]

            for track_id, player in players.items():
                team = player.get('team', 0)
                color = (0, 0, 255) if team == 1 else (255, 0, 0) if team == 2 else (0, 255, 0)
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id, label="Player")

            for _, referee in referees.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255), label="Referee")

            for track_id, keeper in goalkeepers.items():
                frame = self.draw_rectangle(frame, keeper['bbox'], (255, 255, 255), track_id, label="Goalkeeper")

            for _, ball_data in ball.items():
                frame = self.draw_triangle(frame, ball_data['bbox'], (255, 0, 255), label="Ball")

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_frames.append(frame)

        return output_frames
