import pyrealsense2 as rs
import numpy as np
import cv2
import cmath
import math
import matplotlib.pyplot as plt
import os

import torch
import torchvision.transforms as transforms

from PIL import Image, ImageDraw, ImageFont
from pytorch_retinanet.model.retinanet import RetinaNet
from bite_selection_package.model.spnet import DenseSPNet
from bite_selection_package.model.spanet import SPANet
from food_pos_ori_net.model.minispanet import MiniSPANet
from food_pos_ori_net.model.recenternet import RecenterNet
from pytorch_retinanet.utils.encoder import DataEncoder
import shelve

try:
    import cPickle as pickle
except ImportError:
    import pickle

import torch
import torchvision.transforms as transforms
import argparse

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
## detect bounding box for food images


from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Vector3
import bt_msgs.action as bta
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

class GetFoodInfoActionServer(Node):

    def __init__(self, torch_device="cpu", write_dir=None, label_map_path=None):
        super().__init__('get_food_info_node')

        #if write_dir is None:
        #    self.write_dir = os.path.join(os.path.dirname(__file__), "3d_output")
        if write_dir is None:
            self.write_dir = os.path.join(os.path.dirname(__file__), "preds")

        if label_map_path is not None:
            self.items = self.read_label_map(label_map_path)

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        self.align_to_color = rs.align(rs.stream.color)

        # load checkpoint
        self.net = RetinaNet()
        ckpt = torch.load('../checkpoint/retinanet_ckpt.pth', map_location=torch_device)
        self.net.load_state_dict(ckpt['net'])
        self.net.eval()
        self.net.to(torch_device)
        self.device = torch_device
        self.encoder = DataEncoder()

        self.spnet = DenseSPNet()
        self.spnet_crop_size = 136
        self.spnet_grid_size = 17
        self.spnet_discrete_rots = 19
        checkpoint = torch.load('../checkpoint/spnet_ckpt.pth', map_location=torch_device)
        self.spnet.load_state_dict(checkpoint['net'])
        self.spnet.eval()

        self.spanet = SPANet(use_rgb=True, use_depth=False)
        self.spanet_crop_size = 144
        self.spanet_action_types = ['v0', 'v90', 'tv0', 'tv90', 'ta0', 'ta90']
        checkpoint = torch.load('../checkpoint/food_spanet_all_rgb_wall_ckpt_best.pth', map_location=torch_device)
        self.spanet.load_state_dict(checkpoint['net'])
        self.spanet.eval()

        self.minispanet = MiniSPANet()
        self.minispanet_crop_size = 136
        checkpoint = torch.load('../checkpoint/minispanet_ckpt.pth', map_location=torch_device)
        self.minispanet.load_state_dict(checkpoint)
        self.minispanet.eval()

        # TODO: @priya
        self.recenterrotnet = MiniSPANet(out_features=1)
        self.recenterrotnet_crop_size = 136
        #checkpoint = torch.load('../checkpoint/recenterrotnet.pth', map_location=torch_device)
        checkpoint = torch.load('../checkpoint/recenterrotnet_new.pth', map_location=torch_device)
        self.recenterrotnet.load_state_dict(checkpoint)
        self.recenterrotnet.eval()

        self.recenternet = RecenterNet()
        self.recenternet_crop_size = 136
        #checkpoint = torch.load('../checkpoint/recenter.pth', map_location=torch_device)
        checkpoint = torch.load('../checkpoint/recenter_newest.pth', map_location=torch_device)
        self.recenternet.load_state_dict(checkpoint)
        self.recenternet.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.pred_idx = 0

        # ROS
        self._action_server = ActionServer(
            self,
            bta.GetFoodInfoWriteImage,
            'get_food_info',
            self.execute_callback)

        # Start streaming
        self.pipe_profile = self.pipeline.start(self.config)

        # Intrinsics & Extrinsics
        frames = self.pipeline.wait_for_frames()
        self.get_logger().info("Initial frames received. Waiting...")
        frames = self.align_to_color.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        self.depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(
            color_frame.profile)
        self.colorizer = rs.colorizer()

    def read_label_map(self, label_map_path):

        item_id = None
        item_name = None
        items = {}

        with open(label_map_path, "r") as file:
            for line in file:
                line.replace(" ", "")
                if line == "item{":
                    pass
                elif line == "}":
                    pass
                elif "id" in line:
                    item_id = int(line.split(":", 1)[1].strip())
                elif "name" in line:
                    item_name = line.split(":", 1)[1].replace("'", "").strip()

                if item_id is not None and item_name is not None:
                    items[item_id] = item_name
                    item_id = None
                    item_name = None

        return items

    def resize_box(self, box):
        left,top,right,bottom = box.astype(int)
        box_height = bottom-top
        box_width = right-left
        if box_height > box_width:
            diff = box_height - box_width
            offset = 1 if diff%2 else 0
            left -= diff//2
            right += diff//2 + offset
        else:
            diff = box_width - box_height
            offset = 1 if diff%2 else 0
            top -= diff//2
            bottom += diff//2 + offset
        return left,top,right,bottom

    def vis_spanet_pred(self, cv_img, keypoints, scores):
        vis = cv_img.copy()
        vis = cv2.resize(vis, (400,400))
        text_sidebar = np.zeros((200,400,3)).astype(cv_img.dtype)
        rescale_factor = vis.shape[0]/cv_img.shape[0]
        keypoints = keypoints.reshape((2,2))*400
        #keypoints *= rescale_factor
        (u1,v1),(u2,v2) = keypoints.astype(int)
        best_action_idx = np.argmax(scores)
        for i, (label,score) in enumerate(zip(self.spanet_action_types, scores)):
            color = (0,255,0) if i == best_action_idx else (255,255,255)
            text_sidebar = cv2.putText(text_sidebar, "%s: %02f"%(label,score), (20, (i+1)*30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, color, 2, cv2.LINE_AA)
        vis = cv2.line(vis, (u1,v1), (u2,v2), (255,255,255), 2)
        cv2.imwrite(os.path.join(self.write_dir, 'spanet_%02d.png'%self.pred_idx), np.vstack((text_sidebar, vis)))
        cv2.imshow('vis', np.vstack((text_sidebar, vis)))
        cv2.waitKey(0)

    def run_spanet_inference(self, box, pil_img, cv_img, loc_type='ISO'):
        x, y, x2, y2 = box.astype(int)

        center_x_box = int((x + x2) / 2)
        center_y_box = int((y + y2) / 2)

        left,top,right,bottom = self.resize_box(box)
        crop = pil_img.crop((left,top,right,bottom))

        cv_crop = cv_img[top:bottom, left:right]
        crop_resized = crop.resize((self.spanet_crop_size, self.spanet_crop_size))
        inp = torch.stack([self.transform(crop_resized)])

        if loc_type == 'STACKED':
            loc_type = torch.Tensor([[0,0,1]]) # stacked
        elif loc_type == 'WALL':
            loc_type = torch.Tensor([[0,1,0]]) # stacked
        else:
            loc_type = torch.Tensor([[1,0,0]]) # stacked
        pred_vector, feature_map = self.spanet(inp,None,loc_type=loc_type)

        pred_vector = pred_vector.detach().cpu().numpy().squeeze()

        action_scores = pred_vector[4:].tolist()
        best_score_idx = np.argmax(action_scores)
        pred_action = self.spanet_action_types[best_score_idx]
        if best_score_idx < 2:
            skewer_approach = 'vertical_skewer'
        elif best_score_idx < 4:
            skewer_approach = 'vertical_tines'
        else:
            skewer_approach = 'angled_tines'

        u1,v1,u2,v2 = pred_vector[:4]*self.spanet_crop_size
        keypoints = np.array([[u1,v1],[u2,v2]]) if v1 < v2 else np.array([[u2,v2],[u1,v1]])

        vis = self.vis_spanet_pred(cv_crop, pred_vector[:4], action_scores)

        keypoints[:,1] = self.spanet_crop_size - keypoints[:,1] 
        keypoints -= np.array([self.spanet_crop_size//2, self.spanet_crop_size//2])

        angle = angle_between([-self.spanet_crop_size,0], keypoints[0]) # 0 to 180

        #angle_offset = 0 if (best_score_idx % 2 == 1) else -90
        angle_offset = 0 # always go perp

        angle_final = angle+angle_offset
        print('spanet angle', angle_final)

        if angle_final > 180:
            angle_final -= 180

        return angle_final, center_x_box, center_y_box, skewer_approach

    def run_recenterrotnet_inference(self, box, cv_img):
        x, y, x2, y2 = box.astype(int)

        center_x_box = int((x + x2) / 2)
        center_y_box = int((y + y2) / 2)

        left,top,right,bottom = self.resize_box(box)

        cv_crop = cv_img[top:bottom, left:right]
        cv_crop_resized = cv2.resize(cv_crop, (self.recenterrotnet_crop_size, self.recenterrotnet_crop_size))
        rescale_factor = cv_crop.shape[0]/self.recenterrotnet_crop_size

        img_t = self.transform(cv_crop_resized)
        img_t = img_t.unsqueeze(0)
        H,W = self.recenterrotnet_crop_size, self.recenterrotnet_crop_size

        heatmap, pred = self.recenterrotnet(img_t)

        heatmap = heatmap.detach().cpu().numpy()
        pred_rot = pred.detach().cpu().numpy().squeeze()

        heatmap = heatmap[0][0]
        pred_y, pred_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(cv_crop_resized, 0.55, heatmap, 0.45, 0)
        cv2.circle(heatmap, (pred_x,pred_y), 2, (255,255,255), -1)
        cv2.circle(heatmap, (W//2,H//2), 2, (0,0,0), -1)
        pt = cmath.rect(20, np.pi/2-pred_rot)  
        x2 = int(pt.real)
        y2 = int(pt.imag)
        rot_vis = cv2.line(cv_crop_resized, (pred_x-x2,pred_y+y2), (pred_x+x2, pred_y-y2), (255,255,255), 2)
        cv2.putText(heatmap,"Skewer Point",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(rot_vis,"Skewer Angle",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.circle(rot_vis, (pred_x,pred_y), 4, (255,255,255), -1)
        result = np.hstack((heatmap, rot_vis))
        cv2.imwrite(os.path.join(self.write_dir, 'recenterrotnet_%02d.png'%self.pred_idx), result)
        cv2.imshow('vis-minispanet', result)
        cv2.waitKey(0)

        global_x = left + pred_x*rescale_factor
        global_y = top + pred_y*rescale_factor

        if abs(global_x-center_x_box) > 30 or abs(global_y-center_y_box) > 30:
            global_x, global_y = center_x_box, center_y_box

        pred_rot = math.degrees(pred_rot)

        if pred_rot > 90:
            pred_rot -= 180

        print('pred_rot', pred_rot)

        return pred_rot, int(global_x), int(global_y)

    def run_minispanet_inference(self, box, cv_img):
        x, y, x2, y2 = box.astype(int)

        center_x_box = int((x + x2) / 2)
        center_y_box = int((y + y2) / 2)

        left,top,right,bottom = self.resize_box(box)

        cv_crop = cv_img[top:bottom, left:right]
        cv_crop_resized = cv2.resize(cv_crop, (self.minispanet_crop_size, self.minispanet_crop_size))
        rescale_factor = cv_crop.shape[0]/self.minispanet_crop_size

        img_t = self.transform(cv_crop_resized)
        img_t = img_t.unsqueeze(0)
        H,W = self.minispanet_crop_size, self.minispanet_crop_size

        heatmap, pred = self.minispanet(img_t)

        heatmap = heatmap.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy().squeeze()
        pred_rot = pred[0]
        pred_cls = pred[1:]

        mapping = {0: 'ISO', 1: 'WALL', 2: 'STACKED'}
        pred_cls = np.argmax(softmax(pred_cls))
        
        heatmap = heatmap[0][0]
        pred_y, pred_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(cv_crop_resized, 0.55, heatmap, 0.45, 0)
        cv2.circle(heatmap, (pred_x,pred_y), 2, (255,255,255), -1)
        cv2.circle(heatmap, (W//2,H//2), 2, (0,0,0), -1)
        pt = cmath.rect(20, np.pi/2-pred_rot)  
        x2 = int(pt.real)
        y2 = int(pt.imag)
        rot_vis = cv2.line(cv_crop_resized, (pred_x-x2,pred_y+y2), (pred_x+x2, pred_y-y2), (255,255,255), 2)
        cv2.putText(heatmap,"Skewer Point",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(rot_vis,"Skewer Angle",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(rot_vis,"Cls: %s"%(mapping[pred_cls]),(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.circle(rot_vis, (pred_x,pred_y), 4, (255,255,255), -1)
        result = np.hstack((heatmap, rot_vis))
        cv2.imwrite(os.path.join(self.write_dir, 'minispanet_%02d.png'%self.pred_idx), result)
        cv2.imshow('vis-minispanet', result)
        cv2.waitKey(0)

        global_x = left + pred_x*rescale_factor
        global_y = top + pred_y*rescale_factor

        if abs(global_x-center_x_box) > 30 or abs(global_y-center_y_box) > 30:
            global_x, global_y = center_x_box, center_y_box

        pred_rot = math.degrees(pred_rot)

        if pred_rot > 90:
            pred_rot -= 180

        print('pred_rot', pred_rot)

        skew_approach_class = mapping[pred_cls]

        return pred_rot, int(global_x), int(global_y), skew_approach_class

    def run_recenter_inference(self, box, cv_img):
        x, y, x2, y2 = box.astype(int)

        center_x_box = int((x + x2) / 2)
        center_y_box = int((y + y2) / 2)

        left,top,right,bottom = self.resize_box(box)

        cv_crop = cv_img[top:bottom, left:right]
        cv_crop_resized = cv2.resize(cv_crop, (self.minispanet_crop_size, self.minispanet_crop_size))
        rescale_factor = cv_crop.shape[0]/self.minispanet_crop_size

        img_t = self.transform(cv_crop_resized)
        img_t = img_t.unsqueeze(0)
        H,W = self.minispanet_crop_size, self.minispanet_crop_size

        heatmap = self.recenternet(img_t)
        heatmap = heatmap.detach().cpu().numpy()

        heatmap = heatmap[0][0]
        pred_y, pred_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(cv_crop_resized, 0.55, heatmap, 0.45, 0)
        cv2.circle(heatmap, (pred_x,pred_y), 2, (255,255,255), -1)
        cv2.circle(heatmap, (W//2,H//2), 2, (0,0,0), -1)
        cv2.putText(heatmap,"Skewer Point",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        result = heatmap.copy()
        cv2.imwrite(os.path.join(self.write_dir, 'recenter_%02d.png'%self.pred_idx), result)
        cv2.imshow('vis-minispanet', result)
        cv2.waitKey(0)

        global_x = left + pred_x*rescale_factor
        global_y = top + pred_y*rescale_factor

        if abs(global_x-center_x_box) > 30 or abs(global_y-center_y_box) > 30:
            global_x, global_y = center_x_box, center_y_box

        return int(global_x), int(global_y)
    
    def run_spnet_inference(self, box, img, vis=True):
        x, y, x2, y2 = box.astype(int)

        center_x_box = int((x + x2) / 2)
        center_y_box = int((y + y2) / 2)

        left,top,right,bottom = self.resize_box(box)
        crop = img.crop((left,top,right,bottom))
        rescale_factor = crop.size[0]/self.spnet_crop_size
        crop_resized = crop.resize((self.spnet_crop_size, self.spnet_crop_size))
        inp = torch.stack([self.transform(crop_resized)])

        # get predictions
        pred_bmasks, pred_rmasks = self.spnet(inp)

        # print and visualize predicted masks
        rotation_probs = pred_rmasks.data[0].softmax(dim=1).view(self.spnet_grid_size,self.spnet_grid_size,self.spnet_discrete_rots).numpy()
        rotation_grid = np.argmax(rotation_probs, axis=2)
        nonzero_idxs = np.where(rotation_grid != 0)
        rotation_grid[nonzero_idxs] -= 1
        rotation_grid[nonzero_idxs] *= 10

        bm_arr = pred_bmasks.data[0].sigmoid().view(self.spnet_grid_size, self.spnet_grid_size).numpy()
        skew_y, skew_x = np.unravel_index(bm_arr.argmax(), bm_arr.shape)

        #global_x = left + int(skew_x/self.spnet_grid_size*self.spnet_crop_size*rescale_factor) # often off-center
        #global_y = top + int(skew_y/self.spnet_grid_size*self.spnet_crop_size*rescale_factor) # often off-center
        global_x = center_x_box
        global_y = center_y_box

        skew_angle = rotation_grid[skew_y, skew_x] - 90
        pt = cmath.rect(1, math.radians(skew_angle))
        x_delta, y_delta  = pt.real, pt.imag

        skew_axis_x = [skew_x-x_delta, skew_x+x_delta]
        skew_axis_y = [skew_y-y_delta, skew_y+y_delta]

        if vis:
            plt.figure(figsize=(10, 5))
            plt.subplot(121)
            plt.imshow(crop_resized)
            plt.subplot(122)
            plt.imshow(bm_arr)
            plt.plot(skew_x, skew_y, marker='o', color="red", markersize=12)
            plt.plot(skew_axis_x,skew_axis_y, color="red", linewidth=4)
            plt.tight_layout()
            plt.show()

        return skew_angle, global_x, global_y

    def sort_boxes_by_distance(self, boxes):
        print(boxes)
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        x_centers = 0.5*(x1+x2)
        y_centers = 0.5*(y1+y2)
        x_mean = x_centers.mean()
        y_mean = y_centers.mean()
        xy_mean = np.array([x_mean, y_mean])
        xy_centers = np.vstack((x_centers, y_centers)).T
        dist_fn = lambda idx: -np.linalg.norm(xy_centers[idx] - xy_mean)
        idxs_sorted = sorted(list(range(len(boxes))), key=dist_fn)
        return idxs_sorted

    def execute_callback(self, goal_handle):
        fp = goal_handle.request.file_path
        self.get_logger().info('Executing food info goal...')

        result = bta.GetFoodInfoWriteImage.Result()

        if len(fp) > 0:
            result.success = False
            result.error = "write to non-empty filepath has not been implemented yet"
            return result

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        frames = self.align_to_color.process(frames)  # align to color, to make projection accurate
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            result.success = False
            result.error = "Could not read from RealSense!"
            return result

        # print(depth_intrin.ppx, depth_intrin.ppy)

        # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        color_image_vis = color_image.copy()
        depth_image_vis = depth_image.copy()
        new_dir = f"{self.write_dir}"
        os.system(f'mkdir -p {new_dir}')
        # write to file.

        save_path = os.path.join(new_dir, "color.jpg")
        save_path_depth = os.path.join(new_dir, "depth.jpg")
        cv2.imwrite(save_path, color_image)
        cv2.imwrite(save_path_depth, depth_image)

        # read as PIL
        img = Image.open(save_path)
        # might be quite slow
        boxes, labels, scores = self.detect_box(img)
        self.get_logger().debug(f'boxes: {boxes} / labels: {labels} / scores: {scores}')

        # TODO @ priya
        print(boxes)
        idxs = self.sort_boxes_by_distance(boxes)
        boxes = boxes[idxs]
        labels = labels[idxs]
        scores = scores[idxs]

        for box_idx, (box, l, s) in enumerate(zip(boxes, labels, scores)):

            x, y, x2, y2 = box

            x = int(x)
            y = int(y)
            x2 = int(x2)
            y2 = int(y2)
            l = int(l)
            lname = self.items[l] if self.items else ""
            s = float(s)

            skew_angle, center_x, center_y = self.run_recenterrotnet_inference(box, color_image)

            # PRIYA BEFORE
            #skew_angle, center_x, center_y, skew_approach_class  = self.run_minispanet_inference(box, color_image)
            #center_x, center_y = self.run_recenter_inference(box, color_image)
            ##skew_angle, _, _, skew_approach = self.run_spanet_inference(box, img, color_image, loc_type=skew_approach_class)
            ##skew_angle, _, _, skew_approach = self.run_spanet_inference(box, img, color_image, loc_type=skew_approach_class)

            if lname in ['banana', 'kiwi']:
                skew_approach = 'angled_tines'
            else:
                skew_approach = 'vertical_skewer'

            # HACKY ASF todo @ priya
            while skew_angle < -90:
                skew_angle += 180
            while skew_angle > 15:
                skew_angle -= 180

            #_,_,_, skew_approach = self.run_spanet_inference(box, img, color_image, loc_type=skew_approach_class)
            #skew_angle, center_x, center_y, skew_approach = self.run_spanet_inference(box, img, color_image, loc_type=skew_approach_class)
            #print('action', skew_angle, center_x, center_y, skew_approach, skew_approach_class)
            print('action', skew_angle, center_x, center_y, skew_approach)

            center_depth = depth_frame.get_distance(center_x, center_y)
            self.get_logger().debug(f'depth: {center_depth} @ (x,y) = ({center_x}, {center_y})')
            center_pt = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [center_x, center_y], center_depth)
            center_vec = Vector3(x=center_pt[0], y=center_pt[1], z=center_pt[2])

            # add to result
            result.boxes.extend([x, y, x2, y2])
            result.labels.extend([l])
            result.label_names.extend([lname])
            result.scores.extend([s])
            result.center_points.extend([center_vec])
            result.skew_angles.extend([skew_angle])
            result.skew_approach_types.extend([skew_approach])
            
            cv2.circle(color_image_vis, (center_x, center_y), 3, (0, 255, 0), -1)
            cv2.rectangle(color_image_vis, (x, y), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(depth_image_vis, (x, y), (x2, y2), (255, 0, 0), 2)

            break

        save_path = os.path.join(new_dir, "boxes_%02d.jpg"%(self.pred_idx))
        cv2.imwrite(save_path, color_image_vis)
        #save_path_depth = os.path.join(new_dir, "depth_boxes.jpg")
        #cv2.imwrite(save_path_depth, depth_image_vis)
        cv2.imshow('vis', color_image_vis)
        cv2.waitKey(0)


        result.success = True
        self.pred_idx += 1

        goal_handle.succeed()

        return result

    def detect_box(self, img):

        print('Loading image..')
        w, h = img.size
        # summary(net,(3,w,h))
        print('Predicting..')
        x = self.transform(img)
        x = x.unsqueeze(0)
        with torch.no_grad():
            loc_preds, cls_preds = self.net.forward(x.to(self.device))
            print('Decoding..')
            boxes, labels, scores = self.encoder.decode(
                loc_preds.cpu().data.squeeze(),
                cls_preds.cpu().data.squeeze(),
                (w, h))

            # label_map = load_pickled_label_map()
            if boxes is not None:
                boxes = boxes.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                scores = np.zeros_like(labels)  #TODO
                print('labels', labels)
                # print('scores', scores[ind])
            else:
                boxes = []
                labels = []
                scores = []

        return boxes, labels, scores


if __name__ == '__main__':
    rclpy.init()

    parser = argparse.ArgumentParser(description='Test SPNet')
    parser.add_argument('--label_map', required=True)
    parser.add_argument('--use_cpu', action='store_true')
    args = parser.parse_args()

    food_info_server = GetFoodInfoActionServer("cpu" if args.use_cpu else "cuda", label_map_path=args.label_map)

    rclpy.spin(food_info_server)
