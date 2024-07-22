from utils import read_video, save_video, save_video_mp4
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from pitch_detector import Pitch
import logging
logger = logging.getLogger(__name__)

def main():
    # Read Video
    print(f'---Read video---')
    video_name = 0
    video_frames = read_video(f'input_videos/{video_name}.mp4')
    # video_frames = read_video('input_videos/08fd33_4.mp4')
    print(f'---Done read---')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    # print(len(video_frames))

    tracks = tracker.get_object_tracks(video_frames,
                                        read_from_stub=True,
                                        stub_path=f'stubs/track_stubs_{video_name}.pkl')

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path=f'stubs/camera_movement_stub_{video_name}.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.add_assigned_team_to_tracks(video_frames, tracks)

    view_transformer = ViewTransformer('models/best.pt')
    transform_matrices = view_transformer.get_transform_matrices(video_frames)
    view_transformer.add_transformed_position_to_tracks(tracks, transform_matrices)

    pitch_detector = Pitch()

    video_frames = pitch_detector.draw_pitchs(video_frames)

    video_frames = pitch_detector.draw_graphic_positions(tracks,video_frames)


    save_video_mp4(video_frames, f'output_videos/output_video_{video_name}_graphic.mp4')

    
if __name__ == '__main__':
    main()