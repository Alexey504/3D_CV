import open3d as o3d
import numpy as np

def draw_scene(poses, pointcloud, camera_model):
    WIDTH = 1280
    HEIGHT = 720

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(width=WIDTH, height=HEIGHT)

    for pose in poses:
        if pose is not None:
            camera_obj = visualizer.get_view_control().convert_to_pinhole_camera_parameters()
            camera_obj.intrinsic.set_intrinsics(camera_model.size[0], camera_model.size[1], camera_model.focal_length, camera_model.focal_length, camera_model.size[0]//2, camera_model.size[1]//2)
            extrinsic = np.eye(4)
            extrinsic[:3, :] = pose[:3, :]
            camera_lines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=camera_model.size[0], view_height_px=camera_model.size[1], intrinsic=camera_obj.intrinsic.intrinsic_matrix, extrinsic=extrinsic, scale = 0.2)
            visualizer.add_geometry(camera_lines)
        
    #add points
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(pointcloud)
    colors = np.zeros((len(pointcloud), 3))
    points.colors = o3d.utility.Vector3dVector(colors)
        
    visualizer.add_geometry(points)
    visualizer.run()