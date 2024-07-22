from utils import read_video, save_video, save_video_mp4
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from pitch_detector import Pitch
import logging
logger = logging.getLogger(__name__)

def main():
    # Read Video
    print(f'---Read video---')
    video_name = 4
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
    # tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path=f'stubs/camera_movement_stub_{video_name}.pkl')
    # camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer('models/best.pt')
    # view_transformer.add_transformed_position_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    frame_range = [0,5]
    team_assigner.assign_team_color(video_frames[frame_range[0]:frame_range[1]], tracks['players'][frame_range[0]:frame_range[1]])
    # team_assigner.assign_team(video_frames, tracks)

    # transform_matrices = view_transformer.get_transform_matrices(video_frames)


    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= [-1]
    matrices = [np.float32([[1,1,1],[1,1,1],[1,1,1]])]
    for object, object_tracks in tracks.items():
        for frame_num, track in enumerate(object_tracks):
            print(frame_num)
            camera_movement = camera_movement_per_frame[frame_num]
            # perspective_transformer = transform_matrices[frame_num]
 
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            ball_position = get_center_of_bbox(ball_bbox)
                
            perspective_transformer = view_transformer.get_transform_matrix(video_frames[frame_num])
            if perspective_transformer.any():
                matrices.append(perspective_transformer)
            else:
                # matrices.append(matrices[-1])
                perspective_transformer = matrices[-1]

            # loop each items
            for track_id, track_info in track.items():
                # add position
                bbox = track_info['bbox']
                if object == 'ball':
                    position= get_center_of_bbox(bbox)
                else:
                    position = get_foot_position(bbox)
                
                tracks[object][frame_num][track_id]['position'] = position
                
                # add adjusted position
                position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])

                tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

                if object == 'players':
                    # add assigned team
                    team = team_assigner.get_player_team(video_frames[frame_num],   
                                                            bbox,
                                                            track_id)
                    tracks[object][frame_num][track_id]['team'] = team 
                    tracks[object][frame_num][track_id]['team_color'] = team_assigner.team_colors[team]

                    # add ball control
                    assigned_player = player_assigner.assign_ball_to_player(track_id,track_info,ball_position)

                    if assigned_player != -1:
                        tracks[object][frame_num][assigned_player]['has_ball'] = True
                        team_ball_control.append(tracks[object][frame_num][assigned_player]['team'])
                    else:
                        if not team_ball_control:
                            team_ball_control= [-1]
                        else:
                            team_ball_control.append(team_ball_control[-1])
                # add transformed position
                position_transformed = view_transformer.transform_point(np.array(position_adjusted), perspective_transformer)
                if position_transformed is not None:
                    position_transformed = position_transformed.squeeze().tolist()
                tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
    
    team_ball_control= np.array(team_ball_control)
    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)




    # Draw lines
    pitch_detector = Pitch()
    # print(f'---draw pitch---')
    # video_frames = pitch_detector.draw_pitchs(video_frames)
    # save_video_mp4(video_frames, f'output_videos/output_video_{video_name}.mp4')

    # Draw output 
    # Draw object Tracks
    print(f'---draw tracks---')
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    ## Draw Camera movement
    print(f'---draw Camera movement---')
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    ## Draw Speed and Distance
    print(f'---draw Speed and Distance---')
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    print(f'---draw graphic position---')
    output_video_frames = pitch_detector.draw_graphic_positions(tracks,output_video_frames)

    # Save Video
    save_video_mp4(output_video_frames, f'output_videos/output_video_{video_name}.mp4')

if __name__ == '__main__':
    main()