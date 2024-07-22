import numpy as np
import cv2
from shapely.geometry import LineString, Point

class Pitch():
    # def __init__(self) -> None:
        # self.ball_img = cv2.imread('assets/ball.png')

    def get_green_region(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the green range in HSV
        lower_green = np.array([10, 25, 25])
        upper_green = np.array([75, 255, 255])

        # Create a mask for green
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply the mask to the original image
        green_objects = np.zeros_like(frame, np.uint8)
        green_objects[mask > 0] = frame[mask > 0]

        return green_objects
    
    def get_edges(self, frame):
        frame = self.get_green_region(frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        # smooth image
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
        
        # Canny Edges
        low_threshold = 25
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

        return edges

    def get_lines(self, frame):
        Y, X, _ = frame.shape
        edges = self.get_edges(frame)
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 100  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = Y/2  # minimum number of pixels making up a line
        max_line_gap = X/12  # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
        
        return lines
    
    def get_possible_pitch_lines(self, frame):
        possible_pitches = {
            "near_horizontal_line":[],
            "far_horizontal_line":[],
            "penbox_vertical_line":[],
            "halfway_line":[]
        }

        lines = self.get_lines(frame)
        Y, X, _ = frame.shape
        margin = 50

        for line in lines:
            for x1,y1,x2,y2 in line:
                angle_degrees = np.arctan2(abs(y1 - y2), abs(x1 - x2)) * 180 / np.pi
                # halfway line
                if 70 <= angle_degrees < 90: 
                    possible_pitches["halfway_line"].append(line)
                elif angle_degrees == 90 and max(x1,x2) <= X-margin and min(x1,x2) >= margin: # exclude 90degree vertical line at the end/start
                    possible_pitches["halfway_line"].append(line)
                # penalty box lines
                elif 15 <= angle_degrees <= 60:
                    possible_pitches["penbox_vertical_line"].append(line)
                # horizontal lines
                elif 0 <= angle_degrees <= 15 and min(y1,y2) >= Y*0.8:
                    possible_pitches["near_horizontal_line"].append(line)
                elif 0 <= angle_degrees <= 5 and max(y1,y2) <= Y*0.5:
                    possible_pitches["far_horizontal_line"].append(line)
        
        return possible_pitches
    
    def get_extended_lines(self, frame, line, axis=0):
        Y, X, _ = frame.shape
        x1,y1,x2,y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if axis==0: # cut horizontal frame edges
            # 1 point with y=0 and 1 point with y = Y
            if y1==y2:
                x1_,y1_,x2_,y2_=0,y1,X,y2
            else:
                x1_=x2-(y2-Y)/slope
                y1_=Y
                x2_=x2-y2/slope
                y2_=0
        elif axis==1: # cut vertical frame edges
            # 1 point with x=0 and 1 point with x = X
            if x1==x2:
                x1_,y1_,x2_,y2_=x1,0,x2,Y
            else:
                x1_=0
                y1_=y2-x2*slope
                x2_=X
                y2_=y2-(x2-X)*slope
            
        return np.array([[x1_, y1_, x2_, y2_]], dtype=int)
    
    def get_penalty_area_by_lines(self,frame, line):
        '''
         _____a____
        |          |
        d          b
        |_____c____|
        '''
        Y, X, _ = frame.shape
        x1,y1,x2,y2 = line[0]
        if y1==y2:
            a,c = 0, 0
            b,d = y1,y2
        elif x1==x2:
            b,d = 0, 0
            a,c = x1,x2
        else:
            slope = (y2 - y1) / (x2 - x1)
            d = y2 - x2*slope
            a = x2-y2/slope
            b = y2-(x2-X)*slope
            c = x2-(y2-Y)/slope

        # triangle
        area1=abs(a*d)/2

        # trapezium
        if d >0 and b >0:
            area2 = (d+b)*X/2
        else:
            area2 = 0
        return area1, area2
    
    def get_lines_intersection(self, line1, line2):
        line1 = LineString(line1.reshape(2,2))
        line2 = LineString(line2.reshape(2,2))

        int_pt = line1.intersection(line2)
        if type(int_pt) == Point:
            return int(int_pt.x), int(int_pt.y)
            
    def get_parallel_lines(self, point, line):
        "draw a line paralleled with a given line and also cut a given point"
        x1,y1,x2,y2 = line[0]
        x,y = point
        slope = (y2 - y1) / (x2 - x1)
        intercept = y-slope*x
        return np.array([[0,intercept,x,y]], dtype=int)

    def get_pitch_lines(self, frame):
        Y, X, _ = frame.shape
        possible_pitches = self.get_possible_pitch_lines(frame)
        pitches = dict.fromkeys(possible_pitches.keys(),np.array([]))
        for line_type, filtered_lines in possible_pitches.items():
            if filtered_lines:
                x_of_lines = []
                y_of_lines = []
                areas = []
                areas2 = []
                for filtered_line in filtered_lines:
                    areas.append(self.get_penalty_area_by_lines(frame,filtered_line)[0])
                    areas2.append(self.get_penalty_area_by_lines(frame,filtered_line)[1])
                    for x1,y1,_,_ in filtered_line:
                        x_of_lines.append(x1) # x1~x2 for halfway
                        y_of_lines.append(y1) # y1~y2 for horizontal line
                if line_type == 'far_horizontal_line':
                    # get the lowest (highest y) in the top region
                    detected_line = filtered_lines[areas2.index(max(areas2))]
                    # detected_line = filtered_lines[y_of_lines.index(max(y_of_lines))]
                elif line_type in 'penbox_vertical_line':
                    # get the line by largest area
                    detected_line = filtered_lines[areas.index(max(areas))]
                elif line_type in 'near_horizontal_line':
                    # get the line by largest area
                    detected_line = filtered_lines[areas2.index(min(areas2))]
                elif line_type in 'halfway_line':
                    # may return multiple lines with same x1 nearest median, get the first
                    x1 = min(x_of_lines, key=lambda x:abs(x-np.median(x_of_lines)))
                    detected_line = filtered_lines[x_of_lines.index(x1)]
                # x1,y1,x2,y2 = detected_line[0]
                x1_,y1_,x2_,y2_ = self.get_extended_lines(frame, detected_line)[0]
                pitches[line_type] = np.array([[x1_, y1_, x2_, y2_]], dtype=int)
        return pitches

    def draw_lines(self, frame, lines):
         # creating a blank to draw lines on
        img = frame.copy()*0

        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img,(x1,y1),(x2,y2),(0,140,150),5)
        frame_with_lines = cv2.addWeighted(frame, 1, img, 1, 0)
        return frame_with_lines
    
    def draw_pitchs(self, video_frames):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            print(frame_num)
            img = frame.copy()*0
            # frame = frame.copy()
            points = self.get_pitch_featured_points(frame)
            pitches = self.get_pitch_lines(frame=frame)
            for label, line in pitches.items():
                if line.any():
                    x1,y1,x2,y2 = line[0]
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,255),5)
            output_frame = cv2.addWeighted(frame, 1, img, 1, 0)
            for label, point in points.items():
                if point:
                    cv2.circle(output_frame, point, 10, (0, 0, 255), -1)
                    cv2.putText(output_frame,f"{label}",point, cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,0),3)
            output_video_frames.append(output_frame)
        return output_video_frames
    
    def mask_out_horizontal_lines(self, frame, pitches):   
        Y, X, _ = frame.shape
        frame_with_mask = frame.copy()
        if pitches['far_horizontal_line'].any():
            x1,y1,x2,y2 = self.get_extended_lines(frame, pitches['far_horizontal_line'], axis=1)[0]
            pts = np.array([[0,0], [X,0], [x2,y2], [x1,y1]], dtype=int)
            cv2.drawContours(frame_with_mask, np.int32([pts]),0, 0, -1)

        if pitches['near_horizontal_line'].any():
            x1,y1,x2,y2 = self.get_extended_lines(frame, pitches['near_horizontal_line'], axis=1)[0]
            pts = np.array([[0,Y], [X,Y], [x2,y2], [x1,y1]], dtype=int)
            cv2.drawContours(frame_with_mask, np.int32([pts]),0, 0, -1)
        return frame_with_mask
    
    def get_contours(self, frame, pitches):
        frame_with_mask = self.mask_out_horizontal_lines(frame, pitches)
        edges = self.get_edges(frame_with_mask)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours
    
    def get_extreme_points_of_contour(self, contour):
        left = np.array(contour[contour[:, :, 0].argmin()][0])
        right = np.array(contour[contour[:, :, 0].argmax()][0])
        top = np.array(contour[contour[:, :, 1].argmin()][0])
        bottom = np.array(contour[contour[:, :, 1].argmax()][0])
        return left, right, top, bottom
    
    def get_perpendicular_distance(self, point, line):
        x1,y1,x2,y2 = line[0]
        c1 = np.array([x1,y1])
        c2 = np.array([x2,y2])

        distance = np.linalg.norm(np.cross(c2-c1, c1-point))/np.linalg.norm(c2-c1)
        return distance
    
    def get_distance_two_point(self, p1, p2):
        return np.linalg.norm(p1 - p2)
        

    def get_center_circle(self, frame, pitches):
        Y, X, _ = frame.shape
        contours = self.get_contours(frame, pitches)
        halfway_line = pitches['halfway_line']
        areas = []
        distance_perpendicular = []
        peri = []
        min_distance_perpendicular = 0
        max_distance_perpendicular = X/2
        min_area = 10
        max_area = X/2
        max_peri = X*1.5
        for c in contours:
            # Obtain outer coordinates
            left, right, _, _ = self.get_extreme_points_of_contour(c)
            d_left = self.get_perpendicular_distance(left, halfway_line)
            d_right = self.get_perpendicular_distance(right, halfway_line)
            d_perpendicular = (max(d_left, d_right), min(d_left, d_right))
            distance_perpendicular.append(d_perpendicular)
            areas.append(cv2.contourArea(c))
            peri.append(cv2.arcLength(c, True))

        idx = [
            index for index, element in enumerate(zip(areas, distance_perpendicular, peri)) 
            if 
            element[0] >= min_area 
            and element[0] <= max_area
            and element[1][1] >= min_distance_perpendicular
            and element[1][0] <= max_distance_perpendicular
            and element[2] <= max_peri
            ]
        filtered_contours = [contours[i] for i in idx ]

        def fn(c):
            return cv2.arcLength(c, True)
        cnts = sorted(filtered_contours,key=fn, reverse=True)
        if cnts:
            return cnts[0]
        else:
            return []

    def get_pitch_featured_points(self, frame):
        '''
        D____Q_________K_________P____C
        |____          |          ____|
        |    |        _|_        |    |
        |    |\    D./ O \.E    /|.G  |
        |    |/      \_|_/      \|    |
        |____|         |         |____|
        A____M_________H_________N____B
        '''
        featured_points = {
            "O":(),
            "H":(),
            "K":(),
            "M":(),
            "N":(),
            "P":(),
            "Q":(),
            "D":(),
            "E":()
        }
        pitches = self.get_pitch_lines(frame)
        Y, X, _ = frame.shape
        halfway_line = pitches['halfway_line']
        near_horizontal_line = pitches['near_horizontal_line']
        far_horizontal_line = pitches['far_horizontal_line']
        penbox_vertical_line = pitches['penbox_vertical_line']
        if halfway_line.any():
            circle_contour = self.get_center_circle(frame, pitches)
            if len(circle_contour)>0:
                D, E, _, _ = self.get_extreme_points_of_contour(circle_contour)
                dD = self.get_perpendicular_distance(D, halfway_line)
                dE = self.get_perpendicular_distance(E, halfway_line)
                if dD > dE:
                    D_or_E = D
                    featured_points["D"] = tuple(D)
                else:
                    D_or_E = E
                    featured_points["E"] = tuple(E)
                if near_horizontal_line.any():
                    l = self.get_parallel_lines(D_or_E, near_horizontal_line)
                elif far_horizontal_line.any():
                    l = self.get_parallel_lines(D_or_E, far_horizontal_line)
                l = self.get_extended_lines(frame, l, axis=1)
                O = self.get_lines_intersection(halfway_line,l)
                featured_points["O"] = O
        if near_horizontal_line.any() and halfway_line.any():
            H = self.get_lines_intersection(halfway_line,near_horizontal_line)
            featured_points["H"] = H
        if far_horizontal_line.any() and halfway_line.any():
            K = self.get_lines_intersection(halfway_line,far_horizontal_line)
            featured_points["K"] = K
        if near_horizontal_line.any() and penbox_vertical_line.any():
            N_or_M = self.get_lines_intersection(near_horizontal_line,penbox_vertical_line)
            if N_or_M and N_or_M[0] > X/2: # right or left penalty
                N = N_or_M
                featured_points["N"] = N
            else:
                M = N_or_M
                featured_points["M"] = M
        if far_horizontal_line.any() and penbox_vertical_line.any():
            P_or_Q = self.get_lines_intersection(far_horizontal_line,penbox_vertical_line)
            if P_or_Q:
                if P_or_Q[0] > X/2: # right or left penalty
                    P = P_or_Q
                    featured_points["P"] = P
                else:
                    Q = P_or_Q
                    featured_points["Q"] = Q
        return featured_points
    
    def interpolate_feature_points(self, video_frames):
        featured_points = {
            "O":[],
            "H":[],
            "K":[],
            "M":[],
            "N":[],
            "P":[],
            "Q":[],
            "D":[],
            "E":[],
        }
        for frame in video_frames:
            points = self.get_pitch_featured_points(frame)
            for key in points.keys():
                featured_points[key].append(np.array(points[key]))

        return featured_points
    
    def draw_graphic_pitch(self, scale=10, thickness=10):
        '''
        D____Q_________K_________P____C
        P2___          |          ____|
        Z_   |        _|_        |    |
        | |  |\    D./ O \.E    /|.G  |
        R_V  |/      \_|_/      \|    |
        P1___|         |         |____|
        A____M_________H_________N____B
        '''
        line_color = (255,255,255)
        pitch_color = (0,0,0)
        AD = 68*scale #width
        AB = 105*scale #length
        AM = int(16.5*scale) #penbox length
        P1P2 = int(40.2*scale)
        RZ = int(18.5*scale)
        RV = int(5.5*scale)
        AH = int(AB/2)
        HO = int(AD/2)
        AP2 = int((AD+P1P2)/2)
        AP1 = int((AD-P1P2)/2)
        #goal
        bar = 7.32*scale
        AG1 = int((AD-bar)/2)
        AG2 = int((AD+bar)/2)
        # goal area 5m50
        AR = int((AD-RZ)/2)
        AZ = int((AD+RZ)/2)
        # circle
        r = 9*scale
        pen_distance = 11*scale

        pitch_image = np.full((AD, AB, 3), pitch_color, dtype=np.uint8)
        pitch_image.shape
        # 4 border line
        pitch_image[:,0:thickness,:] = line_color
        pitch_image[AD-thickness:AD,:,:] = line_color
        pitch_image[:,AB-thickness:AB,:] = line_color
        pitch_image[0:thickness,:,:] = line_color
        # halfway
        pitch_image[:,int(AH-thickness/2):int(AH+thickness/2),:] = line_color
        # penbox partial circle
        cv2.ellipse(pitch_image, (pen_distance,HO), (r,r), 0, 0, 360, line_color, int(thickness*0.8))
        cv2.ellipse(pitch_image, (AB-pen_distance,HO), (r,r), 0, 0, 360, line_color, int(thickness*0.8))
        # penaltybox area
        pitch_image[AP1:AP2,0:AM,:] = line_color
        pitch_image[AP1+thickness:AP2-thickness,thickness:AM-thickness,:] = pitch_color
        pitch_image[AP1:AP2,AB-AM:AB,:] = line_color
        pitch_image[AP1+thickness:AP2-thickness,AB-AM+thickness:AB-thickness,:] = pitch_color
        # center circle
        cv2.circle(pitch_image, (AH,HO), r, line_color, int(thickness*0.8))
        # goalkeeper zone 5m50
        pitch_image[AR:AZ,0:RV,:] = line_color
        pitch_image[AR+thickness:AZ-thickness,thickness:RV-thickness,:] = pitch_color
        pitch_image[AR:AZ,AB-RV:AB,:] = line_color
        pitch_image[AR+thickness:AZ-thickness,AB-RV+thickness:AB-thickness,:] = pitch_color
        # goal
        pitch_image[AG1:AG2,0:thickness*2,:] = line_color
        pitch_image[AG1:AG2,AB-thickness*2:AB,:] = line_color

        return pitch_image
    
    def draw_tracks_on_graphic_pitch(self,tracks, scale=10, thickness=10, r = 20, draw_ball=True):
        output_video_frames = []
        pitch_img = self.draw_graphic_pitch(scale, thickness)
        # object_tracks=tracks['players']
        for player_tracks, ball_tracks in zip(tracks['players'], tracks['ball']):
            team1_color = (0,255,0)
            team2_color = (0,0,255)
            position_img = pitch_img.copy()
            for _, track_info in player_tracks.items():
                if track_info['position_transformed']:
                    position = track_info['position_transformed']
                    position_scale = (np.array(position)*scale).astype(int)
                    team = track_info['team']
                    color = track_info['team_color']
                    if team == 1:
                        cv2.circle(position_img, position_scale, r, team1_color, -1)
                    if team == 2:
                        cv2.circle(position_img, position_scale, r, team2_color, -1)
            if draw_ball:
                if ball_tracks[1]['position_transformed']:
                    position = ball_tracks[1]['position_transformed']
                    position_scale = (np.array(position)*scale).astype(int)
                    cv2.circle(position_img, position_scale, r, (255,255,255), -1)
                    cv2.circle(position_img, position_scale, int(r/2), (0,0,0), -1)
            output_video_frames.append(position_img)
        return output_video_frames
    
    def draw_graphic_positions(self, tracks, video_frames, scale=10, thickness=10, r=20):
        bg_color = (255,255,255)
        y,x,_ = self.draw_graphic_pitch(scale,thickness).shape
        Y,X,_ = video_frames[0].shape
        margin =10
        blank_frame = np.full((Y+y+margin,X+x+margin,3), bg_color, dtype=np.uint8)
        graphic_frames = self.draw_tracks_on_graphic_pitch(tracks, scale, thickness, r)
        output_video_frames = []
        for frame, graphic_frame in zip(video_frames, graphic_frames):
            frame_extended = blank_frame.copy()
            frame_extended[0:Y, 0:X,:] = frame
            frame_extended[Y+margin:Y+y+margin, X+margin:X+x+margin,:] = graphic_frame
            output_video_frames.append(frame_extended)
        return output_video_frames
