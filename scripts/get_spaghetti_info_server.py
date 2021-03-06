import pyrealsense2 as rs
import argparse
import os
import numpy as np
import cv2
from PIL import Image
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Vector3
import bt_msgs.action as bta
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
import logging
from sklearn.neighbors import NearestNeighbors

from src.unet import UNet
import torch
from torchvision import transforms
import torch.nn.functional as F

class GetSpaghettiInfoActionServer(Node):

    def __init__(self, torch_device="cpu"):
        super().__init__('get_food_info_node')
        self.goal_idx = 0

        self.device = torch_device

        # load checkpoint
        self.seg_net = UNet(n_channels=3, n_classes=2)
        ckpt = torch.load('../checkpoints/spaghetti_seg_unet.pth', map_location=torch_device)
        self.seg_net.load_state_dict(ckpt)
        self.seg_net.eval()
        self.seg_net.to(torch_device)
        self.seg_net_transform = transforms.Compose([transforms.ToTensor()])

        self.write_dir = os.path.join(os.path.dirname(__file__), "preds")
        if not os.path.exists(self.write_dir):
            os.mkdir(self.write_dir)

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align_to_color = rs.align(rs.stream.color)

        # ROS
        self._action_server = ActionServer(
            self,
            bta.GetSpaghettiInfoWriteImage,
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

        self.pred_idx = 0

    def segmentation_preprocess(self, img):
        H,W,C = img.shape
        H_new = 550
        dim_diff = W-H_new
        img = img[:H_new, int(0.6*dim_diff):W-int(0.4*dim_diff)]
        img = cv2.resize(img, (480,480))
        scale_factor = 550/480
        global_pixel_offset = np.array([int(0.6*dim_diff), 0])
        return img, (scale_factor, global_pixel_offset)

    def run_segmentation_inference(self, img):
        img_crop, (scale_factor, global_pixel_offset) = self.segmentation_preprocess(img)
        img_PIL = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
        inp = self.seg_net_transform(img_PIL).to(device=self.device, dtype=torch.float32)
        inp = inp.unsqueeze(0)
        with torch.no_grad():
            output = self.seg_net(inp)
            probs = torch.sigmoid(output)[0]

            tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_crop.shape[1], img_crop.shape[0])),
                transforms.ToTensor()
            ])

            full_mask = probs.cpu()

        mask = F.one_hot(full_mask.argmax(dim=0), self.seg_net.n_classes).permute(2, 0, 1).numpy()
        probs = np.argmax(mask, axis=0)

        mask = cv2.normalize(probs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mask_3ch_vis = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
        #cv2.imshow('vis', np.hstack((img_crop, mask_3ch_vis)))
        #cv2.waitKey(0)
        return img_crop, mask, scale_factor, global_pixel_offset, np.hstack((img_crop, mask_3ch_vis))

    def prune_close_pixels(self, pixels):
        min_thresh = 110
        neigh = NearestNeighbors()
        pruned = []
        for pixel in pixels:
            if len(pruned):
                dists, idxs = neigh.kneighbors(pixel.reshape(1,-1), 1, return_distance=True)
                min_dist = dists[0]
                if min_dist > min_thresh:
                    pruned.append(pixel)
            else:
                pruned.append(pixel)
            neigh.fit(pruned)
        return np.array(pruned)

    def detect_plate(self, img):
        H,W,C = img.shape
        img_orig = img.copy()
        img = cv2.resize(img, (W//2, H//2))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 3))
        detected_circles = cv2.HoughCircles(gray_blurred,
                           cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                       param2 = 30, minRadius = 100, maxRadius = 130)
        plate_mask = np.zeros((H,W))
        # Draw circles that are detected.
        if detected_circles is not None:
            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                print('radius', r)
                #cv2.circle(img, (a, b), r, (0, 255, 0), 2)
                #cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
                #cv2.imshow("Detected Circle", img)

                #cv2.circle(img_orig, (a*2, b*2), r*2, (0, 255, 0), 2)
                #cv2.circle(img_orig, (a*2, b*2), 1, (0, 0, 255), 3)
                #cv2.imshow("Detected Circle", img_orig)

                cv2.circle(plate_mask, (a*2, b*2), int(r*1.85), (255,255,255), -1)
                #cv2.imshow("Detected Circle", plate_mask)
                #cv2.waitKey(0)
                break
            return plate_mask

    def segmask_to_convex_hull(self, img, mask):
        plate_mask = self.detect_plate(img)
        mask_resulting = plate_mask*mask
        cv2.imshow('mask', np.hstack((plate_mask, mask, mask_resulting)))
        cv2.waitKey(0)
        mask = mask_resulting.astype(np.uint8)

        contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        for c in contours:
            print(cv2.contourArea(c))
        #cont = np.vstack((contours[i] for i in range(length) if (cv2.contourArea(contours[i]) > 1600)))
        cont = np.vstack((contours[i] for i in range(length)))
        hull = cv2.convexHull(cont)
        hull_filtered = self.prune_close_pixels(hull.reshape(len(hull), 2)).astype(int)
        print(hull_filtered)
        center = np.mean(hull_filtered, axis=0).astype(int)
        hull_result = []
        for u,v in hull_filtered:
            hull_result.append((np.array([u,v]) - 0.15*(center - np.array([u,v]))).astype(int))
        hull_filtered = np.array(hull_result)
        #    cv2.circle(img, (u,v), 5, (0,255,0), -1)
        #cv2.drawContours(img, [hull], -1, (255,255,0), 2)
        #cv2.circle(img, tuple(center), 5, (0,0,255), -1)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        return hull_filtered, center

    def rescale_px(self, pixels, scale_factor, global_pixel_offset):
        #print(scale_factor, global_pixel_offset)
        pixels = pixels.astype(float)
        pixels *= scale_factor
        pixels += global_pixel_offset
        return pixels.astype(int)

    def angle_between_pixels(self, source_px, target_px, image_width, image_height):
        def angle_between(p1, p2):
            ang1 = np.arctan2(*p1[::-1])
            ang2 = np.arctan2(*p2[::-1])
            return np.rad2deg((ang1 - ang2) % (2 * np.pi))
        if source_px[1] > target_px[1]:
            source_px, target_px = target_px, source_px
        source_px_cartesian = np.array([source_px[0], image_height-source_px[1]])
        target_px_cartesian = np.array([target_px[0], image_height-target_px[1]])
        angle = angle_between(np.array([-image_width,0]), source_px_cartesian-target_px_cartesian)
        robot_angle_offset = -90
        return angle + robot_angle_offset
            
    def execute_callback(self, goal_handle):
        fp = goal_handle.request.file_path
        self.get_logger().info('Executing food info goal...')

        result = bta.GetSpaghettiInfoWriteImage.Result()

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

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        color_image_vis = color_image.copy()
        depth_image_vis = depth_image.copy()

        img_crop, mask, scale_factor, global_pixel_offset, mask_rgb_vis  = self.run_segmentation_inference(color_image)
        hull_px, center_px = self.segmask_to_convex_hull(img_crop, mask)

        hull_px = self.rescale_px(hull_px, scale_factor, global_pixel_offset)
        center_px = self.rescale_px(center_px, scale_factor, global_pixel_offset)
        center_depth = depth_frame.get_distance(center_px[0], center_px[1])
        center_pt_3d = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [center_px[0],center_px[1]], center_depth)
        center_vec_3d = Vector3(x=center_pt_3d[0], y=center_pt_3d[1], z=center_pt_3d[2])

        result.center_point = center_vec_3d

        for (i,(u,v)) in enumerate(hull_px):
            cv2.circle(color_image_vis, (u,v), 5, (0,255,0), -1)
            cv2.arrowedLine(color_image_vis, (u,v), tuple(center_px),(0,255,0), 1)
            cv2.putText(color_image_vis, str(i), (u,v-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,100), 1, 2)
            push_angle = self.angle_between_pixels(np.array([u,v]), center_px, color_image.shape[1], color_image.shape[0])
            uv_depth = depth_frame.get_distance(u, v)
            hull_pt_3d = rs.rs2_deproject_pixel_to_point(self.depth_intrin, [u,v], uv_depth)
            z_thresholded = np.clip(hull_pt_3d[2], 0.46, 0.47)
            hull_vec_3d = Vector3(x=hull_pt_3d[0], y=hull_pt_3d[1], z=z_thresholded)
            result.push_angles.extend([push_angle])
            result.push_points.extend([hull_vec_3d])

        cv2.circle(color_image_vis, tuple(center_px), 5, (0,0,255), -1)
        cv2.imwrite('%s/%03d_push.jpg'%(self.write_dir, self.goal_idx), color_image_vis)
        cv2.imwrite('%s/%03d_vis_debug.jpg'%(self.write_dir, self.goal_idx), mask_rgb_vis)
        cv2.imshow('img',color_image_vis)
        cv2.waitKey(0)


        self.goal_idx += 1
        result.success = True
        goal_handle.succeed()
        self.pipeline.stop()
        return result

if __name__ == '__main__':
    rclpy.init()

    parser = argparse.ArgumentParser(description='Test Spaghetti Action Server')
    parser.add_argument('--use_cpu', action='store_true')
    args = parser.parse_args()

    food_info_server = GetSpaghettiInfoActionServer("cpu" if args.use_cpu else "cuda")

    rclpy.spin(food_info_server)
