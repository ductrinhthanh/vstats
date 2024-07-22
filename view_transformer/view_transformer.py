import numpy as np 
import cv2
# import sys 
# sys.path.append('../')
from pitch_detector import Pitch
from trackers import Tracker
from pitch_detector import Pitch
from team_assigner import TeamAssigner

import numpy as np 
import cv2

class ViewTransformer():
    def __init__(self, model_path):
        # hardcode
#         '''
#         D____Q_________K_________P____C
#         |____          |          ____|
#         |    |        _|_        |    |
#         |    |\    D./ O \.E    /|.G  |
#         |    |/      \_|_/      \|    |
#         |____|         |         |____|
#         A____M_________H_________N____B
#         '''
        self.points_in_reality = {
            "M" : (16.5,68),
            "H" : (105/2,68),
            "N" : (105-16.5,68),
            "Q" : (16.5,0),
            "K" : (105/2,0),
            "P" : (105-16.5,0),
            "O" : (105/2,68/2),
            "D" : (105/2-9.15,68/2),
            "E" : (105/2+9.15,68/2),
        }
        self.tracker = Tracker(model_path)
        self.pitch_detector = Pitch()
        self.team_assigner = TeamAssigner()
        court_width = 68
        court_length = 5.83

        self.pixel_vertices = np.array([[110, 1035], 
                               [265, 275], 
                               [910, 260], 
                               [1640, 915]])
        self.target_vertices = np.array([
            [court_length*8,court_width],
            [court_length*8, 0],
            [court_length*12, 0],
            [court_length*12, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self,point, perspective_transformer):

        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,perspective_transformer)
        return tranform_point.reshape(-1,2)

    # def add_transformed_position_to_tracks(self,tracks, perspective_transformer, position_type='position_adjusted'):
    #     for object, object_tracks in tracks.items():
    #         for frame_num, track in enumerate(object_tracks):
    #             for track_id, track_info in track.items():
    #                 position = track_info[position_type]
    #                 position = np.array(position)
    #                 position_trasnformed = self.transform_point(position, perspective_transformer)
    #                 if position_trasnformed is not None:
    #                     position_trasnformed = position_trasnformed.squeeze().tolist()
    #                 tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed

    def get_transform_points(self, frame):
        priority = [
            ['H', 'K', 'P', 'N'],
            ['Q', 'K', 'H', 'M'],
            ['D', 'K', 'P', 'N'],
            ['Q', 'K', 'E', 'M']
        ]

        pts = self.pitch_detector.get_pitch_featured_points(frame)
        filtered_pts = dict(filter(lambda item: (item[1] != ()) & (item[1] is not None), pts.items()))
        for p in priority:
            if set(p).issubset(list(filtered_pts.keys())):
                return {key: pts[key] for key in p}
                break

    
    def get_transform_matrix(self, frame):
        pts = self.get_transform_points(frame)
        if pts:
            p1,p2,p3,p4 = pts
            src = np.array([
                pts[p1],
                pts[p2],
                pts[p3],
                pts[p4]
            ]).astype(np.float32)

            dst = np.array([
                self.points_in_reality[p1],
                self.points_in_reality[p2],
                self.points_in_reality[p3],
                self.points_in_reality[p4]
            ]).astype(np.float32)
            perspective_transformer = cv2.getPerspectiveTransform(src, dst)
            return perspective_transformer
        else:
            return np.array([])
    
    def get_transform_matrices(self, video_frames):
        matrices = []
        for frame_num, frame in enumerate(video_frames):
            # print(frame_num)
            perspective_transform = self.get_transform_matrix(frame)
            if perspective_transform.any():
                matrices.append(perspective_transform)
            else:
                matrices.append(matrices[-1])

        return matrices
    
    def add_transformed_position_to_tracks(self,tracks, transform_matrices, position_type='position_adjusted'):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                perspective_transformer = transform_matrices[frame_num]
                for track_id, track_info in track.items():
                    position = track_info[position_type]
                    position = np.array(position)
                    position_trasnformed = self.transform_point(position, perspective_transformer)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed


    def test(self, frame):
        perspective_transformer = self.get_matrix_transform(frame)
        tracks = self.tracker.get_object_tracks([frame])
        # tracks["ball"] = self.tracker.interpolate_ball_positions(tracks["ball"])
        self.tracker.add_position_to_tracks(tracks)
        self.add_transformed_position_to_tracks(tracks, perspective_transformer, position_type='position')
        self.team_assigner.assign_team([frame], tracks)
        return tracks
        

# class ViewTransformer():
#     def __init__(self):
#         # hardcode
#         '''
#         D____Q_________K_________P____C
#         |____          |          ____|
#         |    |        _|_        |    |
#         |    |\    D./ O \.E    /|.G  |
#         |    |/      \_|_/      \|    |
#         |____|         |         |____|
#         A____M_________H_________N____B
#         '''
#         self.M = (16.5,0)
#         self.H = (60,0)
#         self.N = (103.5,0)
#         self.Q = (16.5,90)
#         self.K = (60,90)
#         self.P = (103.5,90)

#         self.pitch_detector = Pitch()
    
#     def transform_point(self,frame,point):
#         '''
#         D____Q_________K_________P____C
#         |____          |          ____|
#         |    |        _|_        |    |
#         |    |\    D./ O \.E    /|.G  |
#         |    |/      \_|_/      \|    |
#         |____|         |         |____|
#         A____M_________H_________N____B
#         '''
#         featured_points = self.pitch_detector.get_pitch_featured_points(frame)
#         M,N,P,Q = featured_points["M"], featured_points["N"], featured_points["P"], featured_points["Q"]
#         H,K = featured_points["H"], featured_points["K"]
#         if Q:
#             pixel_vertices = np.array([M, H, K, Q])
#             target_vertices = np.array([self.M,self.H,self.K,self.Q])
#         if P:
#             pixel_vertices = np.array([H, N, P, K])
#             target_vertices = np.array([self.H,self.N,self.P,self.K])
#         pixel_vertices = pixel_vertices.astype(np.float32)
#         target_vertices = target_vertices.astype(np.float32)

#         persepctive_trasnformer = cv2.getPerspectiveTransform(pixel_vertices, target_vertices)

#         p = (int(point[0]),int(point[1]))
#         is_inside = cv2.pointPolygonTest(pixel_vertices,p,False) >= 0 
#         if not is_inside:
#             return None

#         reshaped_point = point.reshape(-1,1,2).astype(np.float32)
#         tranform_point = cv2.perspectiveTransform(reshaped_point,persepctive_trasnformer)
#         return tranform_point.reshape(-1,2)


#     def add_transformed_position_to_tracks(self,tracks,video_frames):
#         for object, object_tracks in tracks.items():
#             for frame_num, track in enumerate(object_tracks):
#                 print(frame_num)
#                 for track_id, track_info in track.items():
#                     position = track_info['position_adjusted']
#                     position = np.array(position)
#                     try:
#                         position_trasnformed = self.transform_point(video_frames[frame_num], position)
#                         if position_trasnformed is not None:
#                             position_trasnformed = position_trasnformed.squeeze().tolist()
#                         tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed
#                     except:
#                         tracks[object][frame_num][track_id]['position_transformed'] = position