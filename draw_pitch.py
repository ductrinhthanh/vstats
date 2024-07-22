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

    

    # Draw lines
    pitch_detector = Pitch()
    print(f'---draw pitch---')
    video_frames = pitch_detector.draw_pitchs(video_frames)
    save_video_mp4(video_frames, f'output_videos/output_video_{video_name}_pitch.mp4')

    
if __name__ == '__main__':
    main()