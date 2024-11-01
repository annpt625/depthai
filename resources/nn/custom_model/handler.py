import cv2
import numpy as np

from depthai_sdk import frameNorm


def decode(nnManager, packet):
    bboxes = np.array(packet.getFirstLayerFp16())
    bboxes = bboxes.reshape((bboxes.size // 7, 7))
    bboxes = bboxes[bboxes[:, 2] > 0.5]
    labels = bboxes[:, 1].astype(int)
    confidences = bboxes[:, 2]
    bboxes = bboxes[:, 3:7]
    return {
        "labels": labels,
        "confidences": confidences,
        "bboxes": bboxes
    }


decoded = ["unknown", "face"]


def draw(nnManager, data, frames):
    for name, frame in frames:
        if name == nnManager.source:
            for label, conf, raw_bbox in zip(*data.values()):
                bbox = frameNorm(frame, raw_bbox)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame, decoded[label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
# import cv2
# import numpy as np
# import depthai as dai
# from depthai_sdk import frameNorm

# def get_camera_intrinsics(device):
#     """Get and print camera calibration data"""
#     calibData = device.readCalibration()
#     intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB)
    
#     # Debug print all intrinsics
#     print("\nCamera Intrinsics Matrix:")
#     print(f"fx (focal length x): {intrinsics[0][0]}")  # Expected: ~1000-2000 for OAK-D Lite
#     print(f"fy (focal length y): {intrinsics[1][1]}")  # Should be very close to fx
#     print(f"cx (principal point x): {intrinsics[0][2]}")  # Expected: ~frame_width/2 (~960)
#     print(f"cy (principal point y): {intrinsics[1][2]}")  # Expected: ~frame_height/2 (~540)
    
#     # Sanity checks for intrinsics
#     if not (900 < intrinsics[0][0] < 2100):
#         print("WARNING: fx value seems unusual for OAK-D Lite")
#     if abs(intrinsics[0][0] - intrinsics[1][1]) > 100:
#         print("WARNING: Large difference between fx and fy")
    
#     return intrinsics

# def calculate_physical_dimensions(bbox, z, intrinsics, frame_width=1920, frame_height=1080): #assuming camera resolution is hd 1080p
#     """
#     Calculate all physical dimensions in meters
    
#     Expected ranges:
#     - Z (depth): 0.2m to 20m (OAK-D Lite effective range)
#     - bbox coordinates: 0.0 to 1.0 (normalized)
#     - Resulting width/height: Generally 0.01m to 5m for typical objects
#     """
#     fx = intrinsics[0][0]
#     fy = intrinsics[1][1]
    
#     # Sanity check for depth
#     if not (0.2 <= z <= 20):
#         print(f"WARNING: Depth {z}m is outside expected range (0.2m-20m)")
    
#     # Sanity check for bbox normalization
#     if not all(0 <= x <= 1 for x in bbox):
#         print("WARNING: Bounding box coordinates should be normalized (0-1)")
    
#     print("\nCalculation Input Values:")
#     print(f"Bounding Box (normalized): {bbox}")
#     print(f"Depth (Z): {z:.3f}m")
#     print(f"Frame dimensions: {frame_width}x{frame_height}")
    
#     # Calculate width
#     bbox_width_pixels = (bbox[2] - bbox[0]) * frame_width
#     physical_width = (bbox_width_pixels * z) / fx
#     print(f"\nWidth Calculation:")
#     print(f"Bbox width in pixels: {bbox_width_pixels:.1f}")  # Expected: 1-1920 pixels
#     print(f"Calculated physical width: {physical_width:.3f}m")
    
#     # Sanity check for width
#     if not (0.01 <= physical_width <= 5):
#         print(f"WARNING: Calculated width {physical_width}m seems unusual")
    
#     # Calculate height
#     bbox_height_pixels = (bbox[3] - bbox[1]) * frame_height
#     physical_height = (bbox_height_pixels * z) / fy
#     print(f"\nHeight Calculation:")
#     print(f"Bbox height in pixels: {bbox_height_pixels:.1f}")  # Expected: 1-1080 pixels
#     print(f"Calculated physical height: {physical_height:.3f}m")
    
#     # Sanity check for height
#     if not (0.01 <= physical_height <= 5):
#         print(f"WARNING: Calculated height {physical_height}m seems unusual")
    
#     # Calculate surface area
#     surface_area = physical_width * physical_height
#     print(f"Calculated surface area: {surface_area:.3f}m²")
    
#     # Sanity check for surface area
#     if not (0.0001 <= surface_area <= 25):  # 1cm² to 25m²
#         print(f"WARNING: Surface area {surface_area}m² seems unusual")
    
#     # Calculate center point
#     center_x_pixels = ((bbox[2] + bbox[0]) / 2) * frame_width
#     center_y_pixels = ((bbox[3] + bbox[1]) / 2) * frame_height
    
#     # Convert center to physical coordinates
#     physical_center_x = (center_x_pixels - frame_width/2) * z / fx
#     physical_center_y = (center_y_pixels - frame_height/2) * z / fy
    
#     print(f"\nCenter Point Calculation:")
#     print(f"Center in pixels: ({center_x_pixels:.1f}, {center_y_pixels:.1f})")
#     print(f"Physical center: ({physical_center_x:.3f}m, {physical_center_y:.3f}m, {z:.3f}m)")
    
#     # Sanity check for center coordinates
#     # At depth Z, x and y should generally be less than Z in magnitude
#     if abs(physical_center_x) > z or abs(physical_center_y) > z:
#         print("WARNING: Center coordinates seem unusually large relative to depth")
    
#     return {
#         "width": physical_width,
#         "height": physical_height,
#         "surface_area": surface_area,
#         "center": {
#             "x": physical_center_x,
#             "y": physical_center_y,
#             "z": z
#         }
#     }

# def decode(nnManager, packet):
#     """
#     Decode neural network output and calculate physical dimensions
    
#     Expected ranges:
#     - Confidence scores: 0.5 to 1.0 (filtered above 0.5)
#     - Number of detections: typically 0-20 per frame
#     - Spatial coordinates (after conversion):
#         X: ±5m (horizontal)
#         Y: ±5m (vertical)
#         Z: 0.2m to 20m (depth)
#     """
#     # Get raw detections
#     bboxes = np.array(packet.getFirstLayerFp16())
#     bboxes = bboxes.reshape((bboxes.size // 7, 7))
#     bboxes = bboxes[bboxes[:, 2] > 0.5]  # Only take high confidence detections
    
#     print("\nRaw Detections:")
#     print(f"Number of detections: {len(bboxes)}")
    
#     # Sanity check for number of detections
#     if len(bboxes) > 20:
#         print("WARNING: Unusually high number of detections")
    
#     labels = bboxes[:, 1].astype(int)
#     confidences = bboxes[:, 2]
#     bboxes = bboxes[:, 3:7]
    
#     # Get camera intrinsics
#     intrinsics = get_camera_intrinsics(nnManager.device)
    
#     spatials = []
#     dimensions = []
    
#     print("\nProcessing each detection:")
#     for i, detection in enumerate(packet.detections):
#         print(f"\nDetection #{i+1}:")
#         print(f"Confidence: {confidences[i]:.3f}")  # should be around 0.5-1.0
        
#         if hasattr(detection, 'spatialCoordinates') and detection.spatialCoordinates:
#             # mm to meters from documentation 
#             x = detection.spatialCoordinates.x / 1000  
#             y = detection.spatialCoordinates.y / 1000
#             z = detection.spatialCoordinates.z / 1000
            
#             # checks
#             if abs(x) > 5:
#                 print(f"WARNING: X coordinate {x}m seems unusually large")
#             if abs(y) > 5:
#                 print(f"WARNING: Y coordinate {y}m seems unusually large")
#             if not (0.2 <= z <= 20):
#                 print(f"WARNING: Z coordinate {z}m is outside expected range")
            
#             print(f"Raw spatial coordinates (meters):")
#             print(f"X: {x:.3f}m")
#             print(f"Y: {y:.3f}m")
#             print(f"Z: {z:.3f}m")
            
#             dims = calculate_physical_dimensions(bboxes[i], z, intrinsics)
            
#             spatials.append({'x': x, 'y': y, 'z': z})
#             dimensions.append(dims)
            
#             print("\nFinal calculated dimensions:")
#             print(f"Width: {dims['width']:.3f}m")
#             print(f"Height: {dims['height']:.3f}m")
#             print(f"Surface Area: {dims['surface_area']:.3f}m²")
#             print(f"Center: X={dims['center']['x']:.3f}m, Y={dims['center']['y']:.3f}m, Z={dims['center']['z']:.3f}m")
#         else:
#             print("WARNING: No spatial coordinates available for this detection")
#             spatials.append({'x': None, 'y': None, 'z': None})
#             dimensions.append(None)
    
#     return {
#         "labels": labels,
#         "confidences": confidences,
#         "bboxes": bboxes,
#         "spatials": spatials,
#         "dimensions": dimensions
#     }

# def draw(nnManager, data, frames):
#     for name, frame in frames:
#         if name == nnManager.source:
#             for label, conf, raw_bbox, spatial, dims in zip(
#                 data["labels"], 
#                 data["confidences"], 
#                 data["bboxes"],
#                 data.get("spatials", []),
#                 data.get("dimensions", [])
#             ):
#                 bbox = frameNorm(frame, raw_bbox)
                
#                 # Bounding box
#                 cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                
#                 # Center point
#                 center = (int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2))
#                 cv2.circle(frame, center, 5, (0, 255, 0), -1)
                
#                 # All values test
#                 texts = [
#                     f"Width: {dims['width']:.3f}m",
#                     f"Height: {dims['height']:.3f}m",
#                     f"Surface Area: {dims['surface_area']:.3f}m²",
#                     f"X: {dims['center']['x']:.3f}m",
#                     f"Y: {dims['center']['y']:.3f}m",
#                     f"Z: {dims['center']['z']:.3f}m"
#                 ]
                
#                 for i, text in enumerate(texts):
#                     cv2.putText(
#                         frame,
#                         text,
#                         (bbox[0] + 10, bbox[1] + 60 + i*20),
#                         cv2.FONT_HERSHEY_TRIPLEX,
#                         0.5,
#                         (0, 255, 0),
#                         1
#                     )