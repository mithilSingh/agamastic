import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import asyncio
import threading
from pyzbar.pyzbar import decode

from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

import tkinter as tk
from tkinter import ttk

QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0

STOP_DISTANCE_THRESHOLD = 0.05      # 5 cm
STOP_FRAMES_REQUIRED = 3            # must be stationary for 5 consecutive checks
MOVE_RESET_DISTANCE = 0.1           # if moved >10 cm, reset halt_counter

PROGRESS_TABLE_GUI = False


class WindowProgressTable:
    def __init__(self, root, shelf_count):
        self.root = root
        self.root.title("Shelf Objects & QR Link")
        self.root.attributes("-topmost", True)

        self.row_count = 2
        self.col_count = shelf_count

        self.boxes = []
        for row in range(self.row_count):
            row_boxes = []
            for col in range(self.col_count):
                box = tk.Text(root, width=10, height=3, wrap=tk.WORD, borderwidth=1,
                          relief="solid", font=("Helvetica", 14))
                box.insert(tk.END, "NULL")
                box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
                row_boxes.append(box)
            self.boxes.append(row_boxes)

        # Make the grid layout responsive.
        for row in range(self.row_count):
            self.root.grid_rowconfigure(row, weight=1)
        for col in range(self.col_count):
            self.root.grid_columnconfigure(col, weight=1)

    def change_box_color(self, row, col, color):
        self.boxes[row][col].config(bg=color)

    def change_box_text(self, row, col, text):
        self.boxes[row][col].delete(1.0, tk.END)
        self.boxes[row][col].insert(tk.END, text)

box_app = None
def run_gui(shelf_count):
    global box_app
    root = tk.Tk()
    box_app = WindowProgressTable(root, shelf_count)
    root.mainloop()




class shelf:
    
    def __init__(self,com,qr_coords,obj_scan_coords,orientation,ortho):
    
        self.com = com
        self.qr_coords = qr_coords
        self.obj_scan_coords = obj_scan_coords
        self.orientation_of_shelf = orientation
        self.ortho_to_orientation = ortho
        self.scanned=False
        
shelves=[]
detected_com=[]
class WarehouseExplore(Node):
    """ Initializes warehouse explorer node with the required publishers and subscriptions.

        Returns:
            None
    """
    def __init__(self):
        super().__init__('warehouse_explore')

        self.action_client = ActionClient(
            self,
            NavigateToPose,
            '/navigate_to_pose')

        self.subscription_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            '/pose',
            self.pose_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_global_map = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap',
            self.global_map_callback,
            100)

        self.subscription_simple_map = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.simple_map_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_status = self.create_subscription(
            Status,
            '/cerebri/out/status',
            self.cerebri_status_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_behavior = self.create_subscription(
            BehaviorTreeLog,
            '/behavior_tree_log',
            self.behavior_tree_log_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_shelf_objects = self.create_subscription(
            WarehouseShelf,
            '/shelf_objects',
            self.shelf_objects_callback,
            QOS_PROFILE_DEFAULT)

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)

        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        # Publisher for output image (for debug purposes).
        self.publisher_qr_decode = self.create_publisher(
            CompressedImage,
            "/debug_images/qr_code",
            QOS_PROFILE_DEFAULT)

        self.publisher_shelf_data = self.create_publisher(
            WarehouseShelf,
            "/shelf_data",
            QOS_PROFILE_DEFAULT)

        self.declare_parameter('shelf_count', 1)
        self.declare_parameter('initial_angle', 0.0)

        self.shelf_count = \
            self.get_parameter('shelf_count').get_parameter_value().integer_value
        self.initial_angle = \
            self.get_parameter('initial_angle').get_parameter_value().double_value

        # --- Robot State ---
        self.armed = False
        self.logger = self.get_logger()

        # --- Robot Pose ---
        self.pose_curr = PoseWithCovarianceStamped()
        self.buggy_pose_x = 0.0
        self.buggy_pose_y = 0.0
        self.buggy_center = (0.0, 0.0)
        self.world_center = (0.0, 0.0)

        # --- Map Data ---
        self.simple_map_curr = None
        self.global_map_curr = None

        # --- Goal Management ---
        self.xy_goal_tolerance = 0.5
        self.goal_completed = True  # No goal is currently in-progress.
        self.goal_handle_curr = None
        self.cancelling_goal = False
        self.recovery_threshold = 10

        # --- Goal Creation ---
        self._frame_id = "map"

        # --- Exploration Parameters ---
        self.max_step_dist_world_meters = 7.0
        self.min_step_dist_world_meters = 4.0
        self.full_map_explored_count = 0
        self.dirn=1
        # --- QR Code Data ---
        self.qr_code_str = "Empty"
        if PROGRESS_TABLE_GUI:
            self.table_row_count = 0
            self.table_col_count = 0

        # --- Shelf Data ---
        self.shelf_objects_curr = WarehouseShelf()

        self.flag = 0
        self.coms=None
        self.current_com_x= 0
        self.current_com_y= 0
        self.qr_done = False
        self.goal_status = ''
        self.shelf_no = 0.5
        self.prev_no_qr = 0
        self.next_shelf = False
        # self.next_count = 0
        self.shelf_table_no = 0
        # self.forcefull_switch=False
        # self.increment_mark=True
        # self.increment_count=0
        self.qr_angle = 0
        self.explore_toggle = True
        self.qr_array = ['0' for _ in range(self.shelf_count)]
        self.obj_counter = 0
        self.current_count = []
        self.prev_pos=[0,0]
        self.stop_count=0
        self.distant_pre_pos=[0,0]
        self.miss_check = False

        self.halt = False
        self.prev_sit = [0.0, 0.0]
        self.halt_counter = 0
        self.last_moved_time = self.get_clock().now().nanoseconds
        self.robot_initial_angle = None


    def pose_callback(self, message):
        """Callback function to handle pose updates.

        Args:
            message: ROS2 message containing the current pose of the rover.

        Returns:
            None
        """
        self.pose_curr = message
        self.buggy_pose_x = message.pose.pose.position.x
        self.buggy_pose_y = message.pose.pose.position.y
        if self.coms == None:
            self.logger.info(f"initial pose {self.buggy_pose_x, self.buggy_pose_y}")
        self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)
        if self.robot_initial_angle is None:
            quat = message.pose.pose.orientation
            siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
            cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            self.robot_initial_angle = yaw
            angle = str(math.degrees(self.robot_initial_angle))
            self.logger.info("Robot's initial angle in degrees = " + angle)

    def simple_map_callback(self, message):
        """Callback function to handle simple map updates.

        Args:
            message: ROS2 message containing the simple map data.

        Returns:
            None
        """
        self.simple_map_curr = message
        map_info = self.simple_map_curr.info
        self.world_center = self.get_world_coord_from_map_coord(
            map_info.width / 2, map_info.height / 2, map_info
        )


    def reach_shelves(self, shelf_index, map_info, map_array, img):
        # self.logger.info(f"shelf_index: {shelf_index}")
        # self.logger.info(f"qr_array: {self.qr_array}")

        if self.next_shelf:
            # self.logger.info(f"reaching next shelf..")
            for index in self.current_count:
                # self.logger.info(f"current_count: {self.current_count}")
                if index in shelf_index:
                    shelf_index.remove(index)
            # shelf_index  = shelf_index[self.next_count:]
        else:
            pass




        if len(shelf_index) > 1:
            min_dist = float('inf')
            
            for index in shelf_index:
                qr_world_coords = self.get_world_coord_from_map_coord(shelves[index].qr_coords[0], shelves[index].qr_coords[1], map_info)
                
                # self.logger.info(f"dist_coords: {qr_world_coords, self.buggy_center}")
                dist = euclidean(qr_world_coords, self.buggy_center)
                
                if dist < min_dist:
                    min_dist = dist
                    counter = index
                    # self.logger.info(f"min_dist: {min_dist}")
            self.current_count.append(index)
        else:
            counter = shelf_index[0]
        






        # if self.qr_array[self.prev_no_qr] != '0':
        #     self.qr_random = self.qr_array[self.prev_no_qr]
        #     self.qr_done = True
        #     # self.logger.info(f"qr_code_str_new: {self.qr_random}")
        # else:
        #     self.qr_random = self.qr_code_str
        self.qr_random = self.qr_code_str





        if not self.qr_done:#qr_done false

            # self.logger.info(f"shelf index-->: {counter}")
            #takes the robot to qr of next shelf

            fx, fy = shelves[counter].qr_coords

            shelves[counter].qr_coords = self.get_world_coord_from_map_coord(shelves[counter].qr_coords[0],shelves[counter].qr_coords[1], map_info)

            dx = float(shelves[counter].com[0] - shelves[counter].qr_coords[0])
            dy = float(shelves[counter].com[1] - shelves[counter].qr_coords[1])
            self.logger.warn(f"com: {shelves[counter].com} qr_coords: {shelves[counter].qr_coords}")
 
            angle = self.create_yaw_from_vector(shelves[counter].com[0], shelves[counter].com[1], shelves[counter].qr_coords[0], shelves[counter].qr_coords[1])
            self.logger.warn(f"send_angle: {angle}")
            # self.qr_done = True #right now the current shelves[counter] refers to the next shelf even in the below else block shelves[counter]		
            
        else:	# qr detected
            if self.qr_random.split("_")[0]  != "Empty":
                no_qr = int(self.qr_random.split("_")[0])

                # self.logger.info(f"return {no_qr}")
                # self.logger.info("check 3")
                
                self.next_shelf = False
                # self.next_count = 0
                self.current_count = []
            else: 
                return
            self.prev_no_qr = no_qr
            
            # self.logger.info(f"counter obj: {counter}")
            #the robot reaches the qr of that shelf refered in the if block comment
            #below code takes the robot in front of next shelf
            fx, fy =shelves[counter].obj_scan_coords
            shelves[counter].obj_scan_coords = self.get_world_coord_from_map_coord(shelves[counter].obj_scan_coords[0],shelves[counter].obj_scan_coords[1], map_info)
            # self.logger.info("check 1")
            # dx = shelves[counter].com[0] - shelves[counter].obj_scan_coords[0]
            # dy = shelves[counter].com[1] - shelves[counter].obj_scan_coords[1]
            self.logger.warn(f"com: {shelves[counter].com} obj_coords: {shelves[counter].obj_scan_coords}")
            angle = self.create_yaw_from_vector(shelves[counter].com[0], shelves[counter].com[1], shelves[counter].obj_scan_coords[0], shelves[counter].obj_scan_coords[1] )

            self.current_com_x,self.current_com_y=shelves[counter].com
            self.node_x, self.node_y = self.get_map_coord_from_world_coord(self.current_com_x,self.current_com_y, map_info)
            self.logger.info(f"reached shelf no: {no_qr} at com: {self.current_com_x,self.current_com_y} ")
#due to this counter value will update in the next itteration to find index of next shelf
            self.qr_angle=float(self.qr_random.split("_")[1])+math.degrees(self.robot_initial_angle)
            self.logger.warn(f"qr qr {self.qr_angle}")
            
            # self.logger.info(f"next shelf com-->: {shelves[counter].com}")
            detected_com.append((shelves[counter].com))
            
        # fx_world, fy_world = self.get_world_coord_from_map_coord(fx, fy, map_info)
        # self.logger.info(f"fx_map: {fx}, fy_map: {fy}, angle: {angle}")

        # self.logger.info(f"fx_world: {fx_world}, fy_world: {fy_world}")
        goal = self.create_goal_from_map_coord(fx,fy,map_info,angle) 
        self.send_goal_from_world_pose(goal)
        
        if self.goal_status == 'accepted':
            if not self.qr_done:
                self.qr_done = True
            else:
                self.qr_done = False


    def explore(self, img,map_array, map_info):
        """Explores in the warehouse.

        This function identifies the shelves in the warehouse.
        It uses the global map to find the coordinates of the shelves.

        Returns:
            None    
        """
        frontiers, gain = self.get_frontiers_for_space_exploration(map_array)
        height,width=np.shape(img)
        img[img==255]=1
        img[img==127]=-1
        

        if len(frontiers)>5:
            closest_frontier = None
            closest = None
            max_optimal_curr = float(0)
            dist=20
            
            
            self.logger.info(f"checking point at :{self.node_x,self.node_y}")
            
            dirn_toggle=True
            while 1:
                if not self.goal_completed:
                    return
                self.node_x+= np.cos(np.deg2rad(self.qr_angle))*dist*self.dirn
                self.node_y+= np.sin(np.deg2rad(self.qr_angle))*dist*self.dirn
                self.logger.info(f"checking point at :{self.node_x,self.node_y}")
                self.node_x = int(self.node_x)
                self.node_y = int(self.node_y)
                
                if (0<self.node_x<width and 0<self.node_y<height):
                    self.logger.info(f"checking point at :{self.node_x,self.node_y},val of point is  {img[self.node_y][self.node_x] } wit qr agle {self.qr_angle}")#
                    self.logger.info(" node point within bounds") 
                    if img[self.node_y][self.node_x]==-1  : #unexplored
                        mind=10000
                        x,y=0,0
                        for fy, fx in frontiers:
                            if euclidean((self.node_x,self.node_y),(fx,fy))<mind:
                                mind= euclidean((self.node_x,self.node_y),(fx,fy))
                                x,y=fx,fy
                        
                                
                        self.logger.info("unexpored  region found, going to nearest frontier ")


                        goal = self.create_goal_from_map_coord(x, y, map_info)
                        self.send_goal_from_world_pose(goal)

                        
                        break
                    elif  img[self.node_y][self.node_x]==1:
                        self.logger.info("obstacle region found")
                        k=0
                        brk=False
                        while not brk:
                            k+=1
                            if k>200:
                                brk=True
                                break
                            for i in range(-k,k):
                                for j in range(-k,k):
                                    
                                    if 0<self.node_x+i<height and 0<self.node_y+j<width and img[self.node_y+j][self.node_x+i]==0:
                                        self.logger.info("explored point found near obstacle ,passing it as goal")

                                        goal= self.create_goal_from_map_coord(self.node_x+i,self.node_y+j,map_info)
                                        self.send_goal_from_world_pose(goal)
                                        brk=True
                                        break
                                if brk:
                                    
                                    break

                            
                        if brk == False:
                            mind=10000
                            x,y=0,0
                            for fy, fx in frontiers:
                                if euclidean((self.node_x,self.node_y),(fx,fy))<mind:
                                    mind= euclidean((self.node_x,self.node_y),(fx,fy))
                                    x,y=fx,fy
                            self.logger.info("no explored point found near obstacle ,passing nearest forienter as a goal")
                            goal = self.create_goal_from_map_coord(x, y, map_info)
                            self.send_goal_from_world_pose(goal)



                    elif img[self.node_y][self.node_x] == 0 :# explored no obstacles

                        self.logger.info(f"free region {img[self.node_y][self.node_x]}")

                        

                        
                    else:
                        self.logger.info(f"unknown region {img[self.node_y][self.node_x]}")

                else:
                    if dist<3 :

                        self.logger.info("no unexplored point found in the dirn of next shelf,changing dirn ")

                        
                        

                        self.node_x-= np.cos(np.deg2rad(self.qr_angle))*dist*self.dirn
                        self.node_y-= np.sin(np.deg2rad(self.qr_angle))*dist*self.dirn
                        self.logger.info(f"checking point at:{self.node_x,self.node_y} with changing direction ")
                        self.dirn*=-1
            
                        goal = self.create_goal_from_map_coord(self.node_x, self.node_y, map_info)
                        self.send_goal_from_world_pose(goal)

                        break
                    else :
                        self.node_x-= np.cos(np.deg2rad(self.qr_angle))*dist*self.dirn
                        self.node_y-= np.sin(np.deg2rad(self.qr_angle))*dist*self.dirn
                        self.logger.info(f"checking point at:{self.node_x,self.node_y} while reduding dist")
                        self.logger.info("reducing dist")
                        dist-=2

    def shelf_coords(self,th,cx, cy, angle, dist, m):
        x1=int(cx+dist*m*np.cos(angle))
        y1=int(cy+dist*m*np.sin(angle))
        x2= int(cx-dist*m*np.cos(angle))
        y2= int(cy-dist*m*np.sin(angle))
        C1=x1>0 and  x1<th.shape[1] and y1>0 and y1<th.shape[0] and  th[y1][x1]==0
        C2=x2>0 and x2<th.shape[1] and y2>0 and y2<th.shape[0] and th[y2][x2]==0
        
        self.logger.info(f"for shelves x1 ,y1{x1,y1}    x2 ,y2{x2,y2}  height,width{th.shape}")
        self.logger.info(f"for shelf coords x1>0  {x1>0 } x1<th.shape[1] {x1<th.shape[1]}  y1>0 {y1>0} y1<th.shape[0] {y1<th.shape[0]}  ")
        self.logger.info(f"for shelf coords x2>0:{x2>0}   x2<th.shape[1]:{x2<th.shape[1]}   y2>0:{y2>0}   y2<th.shape[0]:{y2<th.shape[0]} ")
        if x1<th.shape[1] and y1<th.shape[0]:
            self.logger.info(f"th[y1][x1]==0 {th[y1][x1]}")
        if x2<th.shape[1] and y2<th.shape[0]:
            self.logger.info(f"th[y1][x1]==0 {th[y2][x2]}")
            
        dist1=euclidean(self.buggy_center,(x1,y1))
        dist2=euclidean(self.buggy_center,(x2,y2))
        if C1 and C2:
            if dist1<dist2:
                o1,o2=x1,y1
            else:
                o1,o2=x2,y2
            return o1,o2
        elif C1 :

            o1,o2=x1,y1
            return o1,o2
        elif C2 :
            o1,o2=x2,y2
            return o1,o2
        else:
            if m<0.5:
                self.logger.info(f"extream conds met for shelf {cx,cy,angle,dist,m}")  
                return 
            
            return self.shelf_coords(th,cx, cy, angle, dist, m-0.1)
    def qr_coords(self,th,cx, cy, angle, dist, n):
        x1=int(cx+dist*n*np.cos(angle))
        y1=int(cy+dist*n*np.sin(angle))
        x2= int(cx-dist*n*np.cos(angle))
        y2= int(cy-dist*n*np.sin(angle))
        C1=x1>0 and  x1<th.shape[1] and y1>0 and y1<th.shape[0] and  th[y1][x1]==0
        C2=x2>0 and x2<th.shape[1] and y2>0 and y2<th.shape[0] and th[y2][x2]==0
        
        self.logger.info(f"for qr x1 ,y1{x1,y1}    x2 ,y2{x2,y2}  height,width{th.shape}")
        self.logger.info(f"for qr coords x1>0  {x1>0 } x1<th.shape[1] {x1<th.shape[1]}  y1>0 {y1>0} y1<th.shape[0] {y1<th.shape[0]}  ")
        self.logger.info(f"for qr coords x2>0:{x2>0}   x2<th.shape[1]:{x2<th.shape[1]}   y2>0:{y2>0}   y2<th.shape[0]:{y2<th.shape[0]} ")
        if x1<th.shape[1] and y1<th.shape[0]:
            self.logger.info(f"th[y1][x1]==0 {th[y1][x1]}")
        if x2<th.shape[1] and y2<th.shape[0]:
            self.logger.info(f"th[y1][x1]==0 {th[y2][x2]}")
            
        self.logger.info(f"C1,C2: {C1,C2}")
        
        dist1=euclidean(self.buggy_center,(x1,y1))
        dist2=euclidean(self.buggy_center,(x2,y2))
        if C1 and C2:
            if dist1<dist2:
                c1,c2=x1,y1
            else:
                c1,c2=x2,y2
            return c1,c2
        elif C1 :

            c1,c2=x1,y1
            return c1,c2
        elif C2 :
            c1,c2=x2,y2
            return c1,c2
        else:
            if n<0.5:
                self.logger.info(f"extream conds met for qr {cx,cy,angle,dist,n}")  
                return 
            return self.qr_coords(th,cx, cy, angle, dist, n-0.05)
    def get_shelves(self, img,th, height, width):
        global shelves
        shelves=[]
        self.height=height
        self.width=width
        # img = np.array(data).reshape((height, width)).astype(np.uint8)
        # # img2=img.copy()
        # img[(img != 0)&(img != -1)] = 255
        # img[img == -1] = 127
        # # img[img==0]=0
        # _, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        # th=cv2.flip(th,0)
        # img=cv2.flip(img,0)
        self.logger.info("shelf detection started")
        
        ct, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # self.logger.info(f"total contours found: {len(ct)}")


        for i, cnt in enumerate(ct):
            area = cv2.contourArea(cnt)
            # self.logger.info(f"contour {i} with area {area}")

            if area < 100 or area>10000:  
                continue
            perimeter = cv2.arcLength(cnt, True)
            epsilon = 0.04 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            vertices = len(approx)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)  
            box = np.intp(box) 
            dim = np.sort(rect[1])
            
            # if not rect[1][0]>10000:
            #     cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

            # print((cx,cy))
            
            if   0.5 < dim[0]/dim[1] < 0.8 and area / (dim[0]*dim[1]) > 0.8:#
            
                if  (np.sqrt((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2)>np.sqrt((box[1][0]-box[2][0])**2+(box[1][1]-box[2][1])**2)):
                    if (box[1][0]-box[2][0])!= 0:
                        slope= (box[1][1]-box[2][1])/(box[1][0]-box[2][0])
                        angle = np.arctan(slope) 
                    else:
                        angle=np.pi/2

                    dist=np.sqrt((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2)
                
                else :
                    if (box[0][0]-box[1][0])!=0:
                        slope= (box[0][1]-box[1][1])/(box[0][0]-box[1][0])
                        angle = np.arctan(slope) 
                        
                    else :
                        angle=np.pi/2
                    dist=np.sqrt((box[1][0]-box[2][0])**2+(box[1][1]-box[2][1])**2)
                ortho=angle

                # angle = np.arctan(slope) 
                # print("--->",slope)
                # cv2.drawContours(output,[box],0,(0,0,255),2)
                # print(rect[1])
                M = cv2.moments(cnt)
                self.logger.info(f"shelf candidate at area {area} with dim {dim} and angle {angle*180/np.pi}")
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    n=1
                    m=1.5
                    

                    o1,o2= self.shelf_coords(th,cx, cy, angle, dist, m)
                    self.logger.info(f"shelf found at: {o1,o2} with angle {angle*180/np.pi}")
                    
                    

                    if angle!=np.pi/2:
                        slope=-1/slope
                        angle = np.arctan(slope)
                    else:
                        angle= 0
                    

                    # if int(cx+dist*n*np.cos(angle))<0 or int(cx+dist*n*np.cos(angle))>img.shape[1] or int(cy+dist*n*np.sin(angle))<0 or int(cy+dist*n*np.sin(angle))>img.shape[0] or  th[int(cy+dist*n*np.sin(angle))][int(cx+dist*n*np.cos(angle))]!=1 :
                    #     c1,c2=int(cx-dist*n*np.cos(angle)), int(cy-dist*n*np.sin(angle))
                    # else:
                    #     c1,c2=int(cx+dist*n*np.cos(angle)), int(cy+dist*n*np.sin(angle))	
                    
                    c1,c2= self.qr_coords(th,cx, cy, angle, dist, n)
                    q=False

                    for i in detected_com:
                        if euclidean((cx,height-cy),i)<30:
                            q=True
                            break
                    if not q:
                        shelves.append(shelf((cx,height-cy),(c1,height-c2),(o1,height-o2),(angle*np.pi/180),(ortho*np.pi/180)))
                    self.logger.info(f"complete shelf added {len(shelves)}")
                    continue
            self.logger.info("perfect shape not found")


    def find_shelves(self, img,th, height, width, map_info):
        self.full_map_explored_count += 1
        self.logger.info(f"finf_shelves; count = {self.full_map_explored_count}")
        # if self.flag == 0:
        self.get_shelves(img,th, height, width)
        if self.coms==None:
            
            self.coms = "done"
            self.qr_angle = self.initial_angle + math.degrees(self.robot_initial_angle)
            self.logger.info(f"initial qr angle set to {self.qr_angle}")
            self.node_x, self.node_y = self.get_map_coord_from_world_coord(self.buggy_pose_x, self.buggy_pose_y, map_info)
            self.logger.info(f"initial node x ,node y {self.node_x,self.node_y}")

            # self.logger.info(f"points-->: {len(shelves)}")

        # self.logger.info(f"Array-->: {[x.qr_coords for x in shelves]}")
        
        error = 10 #in degrees

        counter=0
        shelf_index = []
        
        for shelf in shelves:
            shelf.com = self.get_world_coord_from_map_coord(shelf.com[0],shelf.com[1], map_info)
            
            if shelf.com != (self.current_com_x,self.current_com_y): #for skiping current shelf from checking
                # dx = (shelf.com[0] - self.current_com_x)
                # dy = (shelf.com[1] - self.current_com_y)
                # # self.logger.info(f"dx: {dx}, dy: {dy}")
                # dirn = np.arctan2(dy, dx)
                # if dirn < 0:
                #     dirn += 2 * np.pi
                # self.logger.info(f"real --> {dirn*180/np.pi} qr_angle--> {self.qr_angle}")
                # qr_angle_rad = np.deg2rad(self.qr_angle)  # Convert to radians
                # error_rad = np.deg2rad(error)

                # dirn_norm = dirn % (2 * np.pi)
                # qr_angle_norm = qr_angle_rad % (2 * np.pi)

                # angle_diff = (dirn_norm - qr_angle_norm + np.pi) % (2 * np.pi) - np.pi

                # if abs(angle_diff) < error_rad:
                shelf_index.append(counter)
                    
            counter+=1
        
        return shelf_index



    def global_map_callback(self, message):
        """Callback function to handle global map updates.

        Args:
            message: ROS2 message containing the global map data.

        Returns:
            None
        """
        # return
        self.global_map_curr = message

        if not self.goal_completed:
            return
        if self.qr_array[self.prev_no_qr-1] != '0':
            self.qr_code_str = self.qr_array[self.prev_no_qr-1]
            self.logger.info("new qr............")
        self.logger.info(str(self.get_map_coord_from_world_coord(0,0,message.info)))

        height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
        # self.logger.info(f'height: {height}')
        # self.logger.info(f'width: {width}')
        map_array = np.array(self.global_map_curr.data).reshape((height, width))
        map_info = self.global_map_curr.info
        # resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
        # self.logger.info(f"res:{resolution}, ox: {origin_x}, oy: {origin_y}")

        if euclidean(self.buggy_center,self.prev_pos)<0.1:
            self.stop_count+=1
        else:
            self.stop_count=0
            # if self.goal_completed:
            self.distant_pre_pos= self.prev_pos
        if self.stop_count>5:
            # self.logger.info("stuck now moving to a new random location")
            self.stop_count=0
            goal = self.create_goal_from_world_coord(self.distant_pre_pos[0],self.distant_pre_pos[1])
            self.send_goal_from_world_pose(goal)
        self.logger.info(f"stop count -->{self.stop_count}")
        self.prev_pos = self.buggy_center

        img = np.array(self.global_map_curr.data).reshape((height, width))
        img = img.astype(np.int16)
        img[(img != 0)&(img != -1)] = 255
        img[img == -1] = 127
        
        _, th = cv2.threshold(img.astype(np.uint8), 130, 255, cv2.THRESH_BINARY)
        # self.logger.info(f"unique img element{np.unique(img)}")
        # self.logger.info(f"unique element{np.unique(th)}")
        th=cv2.flip(th,0)
        img=cv2.flip(img,0)
        # print(f"--- Overwriting '{self.filename}'... ---")
        # try:
            # Open the file in 'w' (write) mode.
            # This will create the file if it doesn't exist,
            # or completely overwrite it if it does.
        #     with open(self.filename, 'w') as f:
        #         for item in img.flatten():
        #             # Convert each item to a string and add a newline
        #             f.write(str(item) + '\n')
        #     print(f"Successfully overwrite '{self.filename}'.")
        # except IOError as e:
        #     print(f"Error writing to file {self.filename}: {e}")

       
        # ct, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        shelf_index = self.find_shelves(img,th, height, width, map_info)
        self.logger.info(f"shelf_index found: {shelf_index}")

        if len(shelf_index) > 0:
            self.explore_toggle = False
        else:
            self.explore_toggle = True  

        self.logger.info(f"explore_toggle: {self.explore_toggle}")

        if self.explore_toggle:
            # self.logger.info("Exploring the warehouse...")
            self.explore(img,map_array, map_info)
        else:
            # self.logger.info(f"Found shelves: {shelf_index}")
            self.reach_shelves(shelf_index, map_info, map_array, img)
        

    def get_frontiers_for_space_exploration(self, map_array):
        """Identifies frontiers for space exploration.

        Args:
            map_array: 2D numpy array representing the map.

        Returns:
            frontiers: List of tuples representing frontier coordinates.
        """
        frontiers = []
        free_space = 0

        for y in range(1, map_array.shape[0] - 1):
            for x in range(1, map_array.shape[1] - 1):
                if map_array[y, x] == -1:  # Unknown space and not visited.
                    neighbors_complete = [
                        (y, x - 1),
                        (y, x + 1),
                        (y - 1, x),
                        (y + 1, x),
                        (y - 1, x - 1),
                        (y + 1, x - 1),
                        (y - 1, x + 1),
                        (y + 1, x + 1)
                    ]


                    near_obstacle = False
                    for ny, nx in neighbors_complete:
                        if map_array[ny, nx] > 0:  # Obstacles.
                            near_obstacle = True
                            break
                        
                            
                    if near_obstacle:
                        continue

                    neighbors_cardinal = [
                        (y, x - 1),
                        (y, x + 1),
                        (y - 1, x),
                        (y + 1, x),
                    ]

                    for ny, nx in neighbors_cardinal:
                        if map_array[ny, nx] == 0:  # Free space.
                            for py, px in neighbors_complete:
                                if map_array[py, px] == -1:
                                    free_space += 1
                            frontiers.append((ny, nx))
                            break

        return frontiers, free_space



    def publish_debug_image(self, publisher, image):
        """Publishes images for debugging purposes.

        Args:
            publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
            image: Image given by an n-dimensional numpy array.

        Returns:
            None
        """
        if image.size:
            message = CompressedImage()
            _, encoded_data = cv2.imencode('.jpg', image)
            message.format = "jpeg"
            message.data = encoded_data.tobytes()
            publisher.publish(message)

    def camera_image_callback(self, message):
        """Callback function to handle incoming camera images.

        Args:
            message: ROS2 message of the type sensor_msgs.msg.CompressedImage.

        Returns:
            None
        """
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        for barcode in decode(image):
            self.qr_code_str = barcode.data.decode('utf-8')
            if self.qr_code_str.split("_")[0] != "Empty":
                self.qr_array[int(self.qr_code_str.split("_")[0]) - 1] = self.qr_code_str
            self.logger.info(f"QR code data: {self.qr_code_str}")
            # pts = np.array([barcode.polygon],np.int32)
            # pts.reshape((-1,1,2))
            # cv2.polylines(image,[pts],True,(0,255,0),5)
            # pts2 = barcode.rect
            # cv2.putText(image,self.qr_code_str,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)

        # Process the image from front camera as needed.

        # Optional line for visualizing image on foxglove.
        self.publish_debug_image(self.publisher_qr_decode, image)

    def cerebri_status_callback(self, message):
        """Callback function to handle cerebri status updates.

        Args:
            message: ROS2 message containing cerebri status.

        Returns:
            None
        """
        if message.mode == 3 and message.arming == 2:
            self.armed = True
        else:
            # Initialize and arm the CMD_VEL mode.
            msg = Joy()
            msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
            msg.axes = [0.0, 0.0, 0.0, 0.0]
            self.publisher_joy.publish(msg)

    def behavior_tree_log_callback(self, message):
        """Alternative method for checking goal status.

        Args:
            message: ROS2 message containing behavior tree log.

        Returns:
            None
        """
        for event in message.event_log:
            if (event.node_name == "FollowPath" and
                event.previous_status == "SUCCESS" and
                event.current_status == "IDLE"):
                # self.goal_completed = True
                # self.goal_handle_curr = None
                pass

    def shelf_objects_callback(self, message):
        """Callback function to handle shelf objects updates.

        Args:
            message: ROS2 message containing shelf objects data.

        Returns:
            None
        """
        count = 0
        for object in message.object_count:
            count += object

        distance = euclidean(self.buggy_center, self.prev_sit)

        # Check if moved significantly to reset counter
        if distance > MOVE_RESET_DISTANCE:
            self.halt_counter = 0
            self.last_moved_time = self.get_clock().now().nanoseconds

        # Increment counter if very small movement
        if distance < STOP_DISTANCE_THRESHOLD:
            self.halt_counter += 1
        else:
            self.halt_counter = 0

        # Robot must be still for required frames AND for at least 1 second
        elapsed_sec = (self.get_clock().now().nanoseconds - self.last_moved_time) / 1e9
        self.halt = (self.halt_counter >= STOP_FRAMES_REQUIRED and elapsed_sec > 0.25)

        self.prev_sit = self.buggy_center

        # self.logger.info(
        #     f"halt: {self.halt}"
        # )

        if self.qr_code_str.split("_")[0] != "Empty":
            if 7 > count > 3 and int(self.qr_code_str.split("_")[0]) == (self.shelf_table_no + 1) and self.halt:
                self.miss_check = True
                # for i in range(10):
                self.logger.info(f"hi...")
                # self.logger.info(f"Detected {count} objects on shelf.")

                self.shelf_objects_curr = message


                shelf_data_message = WarehouseShelf()

                shelf_data_message.object_name = message.object_name
                shelf_data_message.object_count = message.object_count
                shelf_data_message.qr_decoded = self.qr_code_str


                self.publisher_shelf_data.publish(shelf_data_message)
                # self.obj_counter += 1
                if count > 5:
                    self.shelf_table_no = int(self.qr_code_str.split("_")[0])
            if not self.halt and self.miss_check:
                self.shelf_table_no = int(self.qr_code_str.split("_")[0])
                self.miss_check = False


            # if 7 > count > 5 and int(self.qr_code_str.split("_")[0]) == self.shelf_table_no + 1 and self.goal_completed:
            #     # self.logger.info(f"Detected {count} objects on shelf.")

            #     self.shelf_objects_curr = message

            #     shelf_data_message = WarehouseShelf()

            #     shelf_data_message.object_name = message.object_name
            #     shelf_data_message.object_count = message.object_count
            #     shelf_data_message.qr_decoded = self.qr_code_str


            #     self.publisher_shelf_data.publish(shelf_data_message)
                

            # if self.obj_counter == 30:
            #     self.shelf_table_no = int(self.qr_code_str.split("_")[0])
                # self.obj_counter = 0

        # Process the shelf objects as needed.

        # How to send WarehouseShelf messages for evaluation.
        """
        * Example for sending WarehouseShelf messages for evaluation.
            shelf_data_message = WarehouseShelf()

            shelf_data_message.object_name = ["car", "clock"]
            shelf_data_message.object_count = [1, 2]
            shelf_data_message.qr_decoded = "test qr string"

            self.publisher_shelf_data.publish(shelf_data_message)

        * Alternatively, you may store the QR for current shelf as self.qr_code_str.
            Then, add it as self.shelf_objects_curr.qr_decoded = self.qr_code_str
            Then, publish as self.publisher_shelf_data.publish(self.shelf_objects_curr)
            This, will publish the current detected objects with the last QR decoded.
        """

        # Optional code for populating TABLE GUI with detected objects and QR data.
        """
        if PROGRESS_TABLE_GUI:
            shelf = self.shelf_objects_curr
            obj_str = ""
            for name, count in zip(shelf.object_name, shelf.object_count):
                obj_str += f"{name}: {count}\n"

            box_app.change_box_text(self.table_row_count, self.table_col_count, obj_str)
            box_app.change_box_color(self.table_row_count, self.table_col_count, "cyan")
            self.table_row_count += 1

            box_app.change_box_text(self.table_row_count, self.table_col_count, self.qr_code_str)
            box_app.change_box_color(self.table_row_count, self.table_col_count, "yellow")
            self.table_row_count = 0
            self.table_col_count += 1
        """

    def rover_move_manual_mode(self, speed, turn):
        """Operates the rover in manual mode by publishing on /cerebri/in/joy.

        Args:
            speed: The speed of the car in float. Range = [-1.0, +1.0];
                   Direction: forward for positive, reverse for negative.
            turn: Steer value of the car in float. Range = [-1.0, +1.0];
                  Direction: left turn for positive, right turn for negative.

        Returns:
            None
        """
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, speed, 0.0, turn]
        self.publisher_joy.publish(msg)



    def cancel_goal_callback(self, future):
        """
        Callback function executed after a cancellation request is processed.

        Args:
            future (rclpy.Future): The future is the result of the cancellation request.
        """
        cancel_result = future.result()
        if cancel_result:
            self.logger.info("Goal cancellation successful.")
            self.cancelling_goal = False  # Mark cancellation as completed (success).
            return True
        else:
            self.logger.error("Goal cancellation failed.")
            self.cancelling_goal = False  # Mark cancellation as completed (failed).
            return False

    def cancel_current_goal(self):
        """Requests cancellation of the currently active navigation goal."""
        if self.goal_handle_curr is not None and not self.cancelling_goal:
            self.cancelling_goal = True  # Mark cancellation in-progress.
            self.logger.info("Requesting cancellation of current goal...")
            cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
            cancel_future.add_done_callback(self.cancel_goal_callback)

    def goal_result_callback(self, future):
        """
        Callback function executed when the navigation goal reaches a final result.

        Args:
            future (rclpy.Future): The future that is result of the navigation action.
        """
        status = future.result().status
        # NOTE: Refer https://docs.ros2.org/foxy/api/action_msgs/msg/GoalStatus.html.

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.logger.info("Goal completed successfully!")
        else:
            self.logger.warn(f"Goal failed with status: {status}")

        self.goal_completed = True  # Mark goal as completed.
        self.goal_handle_curr = None  # Clear goal handle.

    def goal_response_callback(self, future):
        """
        Callback function executed after the goal is sent to the action server.

        Args:
            future (rclpy.Future): The future that is server's response to goal request.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.logger.warn('Goal rejected :(')
            self.goal_completed = True  # Mark goal as completed (rejected).
            self.goal_status = 'rejected'
            self.goal_handle_curr = None  # Clear goal handle.
        else:
            self.logger.info('Goal accepted :)')
            self.goal_completed = False  # Mark goal as in progress.
            self.goal_status = 'accepted'
            self.goal_handle_curr = goal_handle  # Store goal handle.

            get_result_future = goal_handle.get_result_async()
            get_result_future.add_done_callback(self.goal_result_callback)

    def goal_feedback_callback(self, msg):
        """
        Callback function to receive feedback from the navigation action.

        Args:
            msg (nav2_msgs.action.NavigateToPose.Feedback): T+he feedback message.
        """
        distance_remaining = msg.feedback.distance_remaining
        number_of_recoveries = msg.feedback.number_of_recoveries
        navigation_time = msg.feedback.navigation_time.sec
        estimated_time_remaining = msg.feedback.estimated_time_remaining.sec

        self.logger.debug(f"Recoveries: {number_of_recoveries}, "
                  f"Navigation time: {navigation_time}s, "
                  f"Distance remaining: {distance_remaining:.2f}, "
                  f"Estimated time remaining: {estimated_time_remaining}s")

        if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
            self.logger.warn(f"Cancelling. Recoveries = {number_of_recoveries}.")
            self.cancel_current_goal()  # Unblock by discarding the current goal.

    def send_goal_from_world_pose(self, goal_pose):
        """
        Sends a navigation goal to the Nav2 action server.

        Args:
            goal_pose (geometry_msgs.msg.PoseStamped): The goal pose in the world frame.

        Returns:
            bool: True if the goal was successfully sent, False otherwise.
        """
        if not self.goal_completed or self.goal_handle_curr is not None:
            return False

        self.goal_completed = False  # Starting a new goal.

        goal = NavigateToPose.Goal()
        goal.pose = goal_pose

        if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
            self.logger.error('NavigateToPose action server not available!')
            return False

        # Send goal asynchronously (non-blocking).
        goal_future = self.action_client.send_goal_async(goal, self.goal_feedback_callback)
        goal_future.add_done_callback(self.goal_response_callback)

        return True



    def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float]]:
        """Helper function to get map origin and resolution."""
        if map_info:
            origin = map_info.origin
            resolution = map_info.resolution
            return resolution, origin.position.x, origin.position.y
        else:
            return None

    def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) \
                       -> Tuple[float, float]:
        """Converts map coordinates to world coordinates."""
        if map_info:
            resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
            world_x = (map_x + 0.5) * resolution + origin_x
            world_y = (map_y + 0.5) * resolution + origin_y
            # self.logger.info(f"world_x: {world_x}, world_y: {world_y}")
            return (world_x, world_y)
        else:
            return (0.0, 0.0)

    def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) \
                       -> Tuple[int, int]:
        """Converts world coordinates to map coordinates."""
        if map_info:
            resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
            map_x = int((world_x - origin_x) / resolution)
            map_y = int((world_y - origin_y) / resolution)
            return (map_x, map_y)
        else:
            return (0, 0)

    def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:
        """Helper function to create a Quaternion from a yaw angle."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = sy
        q.w = cy
        return q

    def create_yaw_from_vector(self, dest_x: float, dest_y: float,
                   source_x: float, source_y: float) -> float:
        """Calculates the yaw angle from a source to a destination point.
            NOTE: This function is independent of the type of map used.

            Input: World coordinates for destination and source.
            Output: Angle (in radians) with respect to x-axis.
        """
        delta_x = dest_x - source_x
        delta_y = dest_y - source_y
        yaw = math.atan2(delta_y, delta_x)

        return yaw

    def create_goal_from_world_coord(self, world_x: float, world_y: float,
                     yaw: Optional[float] = None) -> PoseStamped:
        """Creates a goal PoseStamped from world coordinates.
            NOTE: This function is independent of the type of map used.
        """
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = self._frame_id

        goal_pose.pose.position.x = world_x
        goal_pose.pose.position.y = world_y

        if yaw is None and self.pose_curr is not None:
            # Calculate yaw from current position to goal position.
            source_x = self.pose_curr.pose.pose.position.x
            source_y = self.pose_curr.pose.pose.position.y
            yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
        elif yaw is None:
            yaw = 0.0
        else:  # No processing needed; yaw is supplied by the user.
            pass

        goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)

        pose = goal_pose.pose.position
        print(f"Goal created: ({pose.x:.2f}, {pose.y:.2f}, yaw={yaw:.2f})")
        return goal_pose

    def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info,
                       yaw: Optional[float] = None) -> PoseStamped:
        """Creates a goal PoseStamped from map coordinates."""
        world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)

        return self.create_goal_from_world_coord(world_x, world_y, yaw)


def main(args=None):
    rclpy.init(args=args)

    warehouse_explore = WarehouseExplore()

    if PROGRESS_TABLE_GUI:
        gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
        gui_thread.start()

    rclpy.spin(warehouse_explore)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    warehouse_explore.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()