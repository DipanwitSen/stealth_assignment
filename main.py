import cv2
import os
from utils import read_video, save_video
from trackers import Tracker

def main():
    # Get input
    input_path = input("Enter input video path (e.g., input_video/test.mp4): ").strip()
    if not os.path.exists(input_path):
        print(" Invalid video path.")
        return

    output_dir = "output_video"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, base_name + "_output.mp4")

    # Read video
    video_frames = read_video(input_path)
    print(f" Total frames read: {len(video_frames)}")

    # Initialize tracker
    tracker = Tracker(model_path="models/best.pt")

    # Tracking & processing
    tracks = tracker.get_object_tracks(video_frames)
    tracker.add_position_to_tracks(tracks)
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    team_ball_control = tracker.assign_ball_possession(tracks)

    # Annotate all frames
    annotated_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Display and save
    for frame in annotated_frames:
        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(" Interrupted by user.")
            break

    cv2.destroyAllWindows()
    save_video(annotated_frames, output_path)
    print(f" Saved processed video: {output_path}")

if __name__ == "__main__":
    main()
