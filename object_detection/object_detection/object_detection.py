import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import Point
import cv2
import torch
import numpy as np
import pyrealsense2 as rs
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized,\
    TracedModel
from unitree_crowd_nav_interfaces.msg import PixelArray, Pixel


class ObjectDetection(Node):
    def __init__(self):
        super().__init__("ObjectDetection")
        # Parameters
        self.declare_parameter("weights", "yolov7.pt", ParameterDescriptor(description="Weights file"))
        self.declare_parameter("conf_thres", 0.25, ParameterDescriptor(description="Confidence threshold"))
        self.declare_parameter("iou_thres", 0.45, ParameterDescriptor(description="IOU threshold"))
        self.declare_parameter("device", "cpu", ParameterDescriptor(description="Name of the device"))
        self.declare_parameter("img_size", 640, ParameterDescriptor(description="Image size"))
        self.declare_parameter("use_RGB", False, ParameterDescriptor(description="Use realsense RGB camera"))
        self.declare_parameter("use_depth", False, ParameterDescriptor(description="Use realsense Depth camera"))
        self.declare_parameter("use_dog_cam", False, ParameterDescriptor(description="Use onboard Unitree camera"))
        self.declare_parameter("dog_cam_location", "head/front", ParameterDescriptor(description="Frame for the Unitree camera"))

        self.weights = self.get_parameter("weights").get_parameter_value().string_value
        self.conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        self.iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.img_size = self.get_parameter("img_size").get_parameter_value().integer_value
        self.use_RGB = self.get_parameter("use_RGB").get_parameter_value().bool_value
        self.use_depth = self.get_parameter("use_depth").get_parameter_value().bool_value
        self.use_dog_cam = self.get_parameter("use_dog_cam").get_parameter_value().bool_value
        self.dog_cam_location = self.get_parameter("dog_cam_location").get_parameter_value().string_value

        # Camera info and frames
        self.depth = None
        self.depth_color_map = None
        self.rgb_image = None
        self.intr = None

        # Flags
        self.camera_RGB = False
        self.camera_depth = False
        self.camera_dog = False

        # Timer callback
        self.frequency = 20  # Hz
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)

        # Publishers for Classes
        self.pub_person = self.create_publisher(Point, "/person", 10)
        self.person = Point()
        self.pub_door = self.create_publisher(Point, "/door", 10)
        self.door = Point()
        self.pub_stairs = self.create_publisher(Point, "/stairs", 10)
        self.stairs = Point()

        # Realsense package
        self.bridge = CvBridge()

        # Subscribers
        if self.use_RGB:
            self.rs_sub = self.create_subscription(CompressedImage, '/camera/color/image_raw/compressed', self.rs_callback, 10)
        if self.use_depth:
            self.align_depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.align_depth_callback, 10)
            self.intr_sub = self.create_subscription(CameraInfo, 'camera/aligned_depth_to_color/camera_info', self.intr_callback, 10)

        if self.use_dog_cam:
            self.dog_left_sub = self.create_subscription(CompressedImage, '/'+self.dog_cam_location+'/cam/image_rect/left/compressed', self.dog_left_cb, 10)
            self.dog_right_sub = self.create_subscription(CompressedImage, '/'+self.dog_cam_location+'/cam/image_rect/right/compressed', self.dog_right_cb, 10)
            self.dog_depth_sub = self.create_subscription(Image, '/'+self.dog_cam_location+'/cam/image_depth', self.dog_depth_cb, 10)

        # Initialize YOLOv7
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        # Load model
        self.model = attempt_load(self.weights, map_location=self.device) # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.img_size, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1

        # tracking
        self.left_tracking = PixelArray()
        self.right_tracking = PixelArray()
        self.dog_left_image = None
        self.dog_right_image = None
        self.dog_depth_image = None

        self.rgb_result = None
        self.depth_result = None
        self.dog_left_result = None
        self.dog_right_result = None

        self.pub_left = self.create_publisher(PixelArray, self.dog_cam_location+"/pixels_left", 10)
        self.pub_right = self.create_publisher(PixelArray, self.dog_cam_location+"/pixels_right", 10)

    def intr_callback(self, cameraInfo):
        """
        Get the camera information of the depth frame.

        Args: cameraInfo: Camera information obtained from the aligned_depth_to_color/camera_info
                          topic

        Returns: None
        """
        if self.intr:
            return
        self.intr = rs.intrinsics()
        self.intr.width = cameraInfo.width
        self.intr.height = cameraInfo.height
        self.intr.ppx = cameraInfo.k[2]
        self.intr.ppy = cameraInfo.k[5]
        self.intr.fx = cameraInfo.k[0]
        self.intr.fy = cameraInfo.k[4]
        if cameraInfo.distortion_model == 'plumb_bob':
            self.intr.model = rs.distortion.brown_conrady
        elif cameraInfo.distortion_model == 'equidistant':
            self.intr.model = rs.distortion.kannala_brandt4
        self.intr.coeffs = [i for i in cameraInfo.d]

    def align_depth_callback(self, data):
        """
        Subscription to the depth camera topic.

        Args: data (sensor_msgs/msg/Image): Frames obtained from the
                                                      /camera/aligned_depth_to_color/image_raw topic

        Returns: None
        """
        self.depth  = self.bridge.imgmsg_to_cv2(data)
        self.depth = cv2.flip(cv2.flip(np.asanyarray(self.depth),0),1) # Camera is upside down on the Go1
        self.depth_color_map = cv2.applyColorMap(cv2.convertScaleAbs(self.depth, alpha=0.08), cv2.COLORMAP_JET)
        self.camera_depth = True

    def rs_callback(self, data):
        """
        Subscription to the compressed RGB camera topic.

        Args: data (sensor_msgs/msg/CompressedImage): Frames obtained from the 
                                                      /camera/color/image_raw/compressed topic

        Returns: None
        """
        self.rgb_image = self.bridge.compressed_imgmsg_to_cv2(data)
        self.camera_RGB = True


    def dog_left_cb(self, data):
        """
        Subscription to the compressed RGB camera topic.

        Args: data (sensor_msgs/msg/CompressedImage): Frames obtained from the 
                                                      /head/front/cam/image_rect/left/compressed topic

        Returns: None
        """
        # self.get_logger().info("Got dog cam image!")
        self.dog_left_image = self.bridge.compressed_imgmsg_to_cv2(data)
        self.camera_dog = True

    def dog_right_cb(self, data):
        """
        Subscription to the compressed RGB camera topic.

        Args: data (sensor_msgs/msg/CompressedImage): Frames obtained from the 
                                                      /head/front/cam/image_rect/right/compressed topic

        Returns: None
        """
        # self.get_logger().info("Got dog cam image!")
        self.dog_right_image = self.bridge.compressed_imgmsg_to_cv2(data)
        self.camera_dog = True

    def dog_depth_cb(self, data):
        """
        Subscription to the compressed depth camera topic.

        Args: data (sensor_msgs/msg/CompressedImage): Frames obtained from the 
                                                      /head/front/cam/image_rect/left/compressedDepth topic

        Returns: None
        """
        # self.get_logger().info("Got dog cam image!")
        self.dog_depth_image = self.bridge.imgmsg_to_cv2(data)
        # self.camera_dog = True
        # cv2.imshow("DepthImage", self.dog_depth_image)
        # cv2.waitKey(1)

    def YOLOv7_detect(self, dog_frame=None):
        """ Preform object detection with YOLOv7"""
        
        shape_before = None
        shape_after = None

        if self.camera_RGB:
             # Flip realsense image as it is mounted upside down on dog
            img = cv2.flip(cv2.flip(np.asanyarray(self.rgb_image),0),1)
            # self.get_logger().info(f"REALSENSE SHAPE:{img.shape}")
        elif self.camera_dog and dog_frame=="left":
            # dont flip from onboard unitree camera
            img = self.dog_left_image
            img_og = img.copy()
            shape_before = img.shape
            img = cv2.resize(img, [640,480])
            shape_after = img.shape
        elif self.camera_dog and dog_frame=="right":
            # dont flip from onboard unitree camera
            img = self.dog_right_image
            img_og = img.copy()
            shape_before = img.shape
            img = cv2.resize(img, [640,480])
            shape_after = img.shape

        scale_x = shape_before[1]/shape_after[1]
        scale_y = shape_before[0]/shape_after[0]

        im0 = img.copy()
        # img is the "tensor" object
        # this is the thing that is the wrong shape
        img = img[np.newaxis, :, :, :]
        img = np.stack(img, 0)
        img = img[..., ::-1].transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # self.get_logger().info(f"img:\n\n\n\n{img}")
        # self.get_logger().info(f"shape:\n\n\n\n{img.shape}")

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        t3 = time_synchronized()

        detected = False

        # Process detections   
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'

                    if conf > 0.8: # Limit confidence threshold to 80% for all classes
                        detected = True
                        # Draw a boundary box around each object
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)
                        # Get box top left & bottom right coordinates
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        x = int((c2[0]+c1[0])/2)
                        y = int((c2[1]+c1[1])/2)
                        # self.get_logger().info(f"x,y:{x} {y}")
                        im0 = cv2.circle(im0, (x,y), radius=5, color=(0, 0, 255), thickness=-1)
                        img_og = cv2.circle(img_og, (int(scale_x*x),int(scale_y*y)), radius=5, color=(0, 0, 255), thickness=-1)
                        
                        pix = Pixel()
                        pix.x = int(scale_x*x)
                        pix.y = int(scale_y*y)
                        if dog_frame == "left":
                            self.left_tracking.pixels.append(pix)
                        elif dog_frame == "right":
                            self.right_tracking.pixels.append(pix)
                        if self.use_depth == True:
                            plot_one_box(xyxy, self.depth_color_map, label=label, color=self.colors[int(cls)], line_thickness=2)

                            label_name = f'{self.names[int(cls)]}'

                            # Limit location and distance of object to 480x680 and 5meters away
                            if x < 480 and y < 640 and self.depth[x][y] < 5000:
                                # Get depth using x,y coordinates value in the depth matrix
                                if self.intr:
                                    real_coords = rs.rs2_deproject_pixel_to_point(self.intr, [x, y], self.depth[x][y])

                                if real_coords != [0.0,0.0,0.0]:
                                    depth_scale = 0.001
                                    # Choose label for publishing position Relative to camera frame
                                    if label_name == 'person':
                                        self.person.x = real_coords[0]*depth_scale
                                        self.person.y = real_coords[1]*depth_scale
                                        self.person.z = real_coords[2]*depth_scale # Depth
                                        self.pub_person.publish(self.person)
                                    if label_name == 'door':
                                        self.door.x = real_coords[0]*depth_scale
                                        self.door.y = real_coords[1]*depth_scale
                                        self.door.z = real_coords[2]*depth_scale # Depth
                                        self.pub_door.publish(self.door)
                                    if label_name == 'stairs':
                                        self.stairs.x = real_coords[0]*depth_scale
                                        self.stairs.y = real_coords[1]*depth_scale
                                        self.stairs.z = real_coords[2]*depth_scale # Depth
                                        self.pub_stairs.publish(self.stairs)
                                    self.get_logger().info(f"depth_coord = {real_coords[0]*depth_scale}  {real_coords[1]*depth_scale}  {real_coords[2]*depth_scale}")
            if dog_frame is not None:
                if dog_frame == 'left':
                    self.dog_left_result = img_og # im0
                    # cv2.imshow("original left", self.dog_left_image)
                elif dog_frame == 'right':
                    self.dog_right_result = img_og # im0
                    # cv2.imshow("original right", self.dog_right_image)
            else:
                self.rgb_result = im0
                if self.use_depth:
                    self.depth_result = self.depth_color_map
            # cv2.waitKey(1)
    
    def publish_points(self):
        # Print last positions in left and right frame
        # self.get_logger().info(f"\nL: {self.left_tracking.pixels}\nR: {self.right_tracking.pixels}")
        self.pub_left.publish(self.left_tracking)
        self.pub_right.publish(self.right_tracking)
        # reset tracking information
        self.left_tracking = PixelArray()
        self.right_tracking = PixelArray()

    def timer_callback(self):
        if self.use_RGB and self.camera_RGB:
            self.YOLOv7_detect()
            cv2.imshow("Realsense YOLO Results", self.rgb_result)
            if self.use_depth:
                cv2.imshow("Realsense YOLO Results: Depth", self.depth_result)
            cv2.waitKey(1)
        elif self.use_dog_cam and (self.dog_left_image is not None) and (self.dog_right_image is not None):
            self.YOLOv7_detect(dog_frame='left')
            self.YOLOv7_detect(dog_frame='right')
            # publish, reset tracking information
            self.publish_points()
            # display image
            concat = np.concatenate((self.dog_left_result, self.dog_right_result), axis=1)
            cv2.imshow("Dog YOLO Results", concat)
            cv2.waitKey(1)

def main(args=None):
    """Run the main function."""
    rclpy.init(args=args)
    with torch.no_grad():
        node = ObjectDetection()
        rclpy.spin(node)
        rclpy.shutdown()

if __name__ == '__main__':
    main()
