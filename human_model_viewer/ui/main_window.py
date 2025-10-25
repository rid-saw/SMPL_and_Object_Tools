import sys
import numpy as np
import cv2
import os

import configparser

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import Qt  # ADD THIS LINE
from opendr.camera import ProjectPoints, Rodrigues
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight

from .gen.main_window import Ui_MainWindow as Ui_MainWindow_Base
from .camera_widget import Ui_CameraWidget
from .util import *
from .draw_utils import *

from OpenGL.GL import *
from OpenGL.GLUT import *

import yaml
import numpy as np
import threading
import trimesh  # Add this for loading arbitrary meshes

### Renderer import with better error handling:
try:
    from .renderer_pyrd_v1 import Renderer
    RENDERER_AVAILABLE = True
    print("✓ Renderer successfully imported")
except ImportError as e:
    print(f"✗ Could not import Renderer: {e}")
    RENDERER_AVAILABLE = False
    Renderer = None


# Path to the tool directory shape
ROOT = "/Users/riddhi/Code/Graduate Research/SMPL_Tools/human_model_viewer/"
SEQUENCE = "gates-foyer-2019-01-17_0" # human_model_viewer/sequences/gates-foyer-2019-01-17_0@cam_

model_type_list = ['smpl', 'smplx', 'flame']

FRAME_FREQ = 1

class SimpleObjectModel:
    """Simple wrapper for arbitrary 3D object meshes to work with opendr"""
    def __init__(self, vertices, faces, original_trans=None, object_name=None):
        self.v_original = vertices.copy()  # Store ORIGINAL file vertices (never changes)
        self.v = vertices.copy()  # Working vertices (translated for display)
        self.f = faces
        # Store the original location from the loaded object (this never changes)
        self.original_trans = original_trans if original_trans is not None else np.zeros(3)
        # Calculate object center for proper rotation/scaling
        self.object_center = np.mean(self.v, axis=0)
        # Temporary centering offset (applied automatically when loading)
        self.centering_offset = np.zeros(3)
        # User adjustments for alignment purposes (not saved)
        self.user_trans = np.zeros(3)
        # These get saved
        self.rotation = np.zeros(3)  # Euler angles - SAVED
        self.scale = np.ones(3)      # Scale factors - SAVED
        # Store object name for scale syncing
        self.object_name = object_name  # NEW: Store object name
        
    @property
    def r(self):
        """Get transformed vertices for display - rotate/scale around object center"""
        # Start with working vertices (already positioned)
        vertices = self.v.copy()
        
        # Calculate current object center
        current_center = self.object_center
        
        # Apply scale relative to object center
        if not np.allclose(self.scale, 1.0):
            centered_v = vertices - current_center
            scaled_v = centered_v * self.scale
            vertices = scaled_v + current_center
        
        # Apply rotation relative to object center
        if np.any(self.rotation != 0):
            centered_v = vertices - current_center
            R = cv2.Rodrigues(self.rotation)[0]
            rotated_v = centered_v.dot(R.T)
            vertices = rotated_v + current_center
        
        # Apply user translation adjustments
        transformed_v = vertices + self.user_trans
        
        return transformed_v
    
    def get_original_transformed_vertices(self):
        """Get original vertices with only rotation and scale applied (for saving)"""
        vertices = self.v_original.copy()
        original_center = np.mean(vertices, axis=0)
        
        # Apply scale relative to original center
        if not np.allclose(self.scale, 1.0):
            centered_v = vertices - original_center
            scaled_v = centered_v * self.scale
            vertices = scaled_v + original_center
        
        # Apply rotation relative to original center
        if np.any(self.rotation != 0):
            centered_v = vertices - original_center
            R = cv2.Rodrigues(self.rotation)[0]
            rotated_v = centered_v.dot(R.T)
            vertices = rotated_v + original_center
            
        return vertices

    def get_saved_transform_data(self):
        """Return only the data that should be saved (rotation and scale only)"""
        return {
            'original_trans': self.original_trans.copy(),
            'rotation': self.rotation.copy(), 
            'scale': self.scale.copy(),
            'vertices': self.v_original.copy(),
            'faces': self.f.copy()
        }
    
    def get_original_transformed_vertices_with_position(self):
        """Get original vertices with rotation, scale, AND position applied (for saving)"""
        # Apply user position to original JRDB coordinates first
        vertices = self.v_original + self.user_trans
        adjusted_center = np.mean(vertices, axis=0)
        
        # Apply scale relative to the adjusted center
        if not np.allclose(self.scale, 1.0):
            centered_v = vertices - adjusted_center
            scaled_v = centered_v * self.scale
            vertices = scaled_v + adjusted_center
        
        # Apply rotation relative to the adjusted center
        if np.any(self.rotation != 0):
            centered_v = vertices - adjusted_center
            R = cv2.Rodrigues(self.rotation)[0]
            rotated_v = centered_v.dot(R.T)
            vertices = rotated_v + adjusted_center
            
        return vertices
    

class ReturnableThread(threading.Thread):
    def __init__(self, target, *args, **kwargs):
        threading.Thread.__init__(self)
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.result = None
    
    def run(self):
        self.result = self.target(*self.args, **self.kwargs)


class Ui_MainWindow(QtWidgets.QMainWindow, Ui_MainWindow_Base):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        with open(ROOT + '/camera-calibrations/lidars.yaml', 'r') as file:
            self.camera_config = yaml.safe_load(file)
        
        self.show_pcd = False
        
        # Object mesh related variables
        self.object_model = None
        self.object_rn = None
        self.object_light = None
        self.show_object = False
        self.object_camera = None
        self.object_joints2d = None
        
        # NEW: Store scale values for syncing across track (ONLY ADDITION FOR SCALE SYNC)
        self.stored_scale_for_sync = None
        
        # ADD: Flags to prevent recursive calls and manage frame changes
        self._rendering_in_progress = False
        self._frame_changing = False
        
        self.sensor = 0
        self.all_sensors = [0, 2, 4, 6, 8]
        
        self.track = None
        self.all_tracks = []
        
        self.current_frame = None
        self.all_frames = []
        
        self.sensor_choose.currentIndexChanged[int].connect(
            lambda val: self._update_sensor(val))
        
        self.track_choose.currentIndexChanged[int].connect(
            lambda val: self._update_tracks(val))
        
        self.frame_choose.currentIndexChanged[int].connect(
            lambda val: self._update_frames(val))

        self.focal_length_jrdb = []
        self.camera_center_jrdb = []

        self.jrdb_to_op = [0, 4, 3, 6, 9, 5, 7,
                           12, 8, 10, 13, 15, 11, 14, 16, 1, 2]

        self.orig_back = np.zeros((480,752,3))
        self.pose_transl = np.zeros(3)
        
        self.gender_info = np.load(ROOT + '/pkl_file_jrdb/' + 'gates-foyer-2019-01-17_0' +'.npy',allow_pickle=True).item()

        # Initializing variables that control the state of user interactions
        self._moving = False
        self._rotating = False
        self._mouse_begin_pos = None
        self._update_canvas = False

        # Initializing the camera and related parameters for projecting 3D points to 2D
        self.camera = ProjectPoints(rt=np.zeros(3), t=np.zeros(3))
        self.joints2d = ProjectPoints(rt=np.zeros(3), t=np.zeros(3))
        self.pcd = ProjectPoints(rt=np.zeros(3), t=np.zeros(3))
        self.frustum = {'near': 0.1, 'far': 1000., 'width': 100, 'height': 30}

        # Initializing the lighting model for the scene
        self.light = LambertianPointLight(vc=np.array(
            [0.98, 0.98, 0.98]), light_color=np.array([1., 1., 1.]))

        # Initializing the renderer for drawing the 3D model.
        self.rn = ColoredRenderer()
        self.rn.set(glMode='glfw', bgcolor=np.ones(3), frustum=self.frustum, camera=self.camera, vc=self.light,
                    overdraw=False)
        self.rn.overdraw = True
        self.rn.nsamples = 8
        self.rn.msaa = False
        self.rn.initGL()
        self.rn.debug = False

        # Initializing the model type, gender, and loads the 3D model
        self.model_type = 'smpl'
        self.model_gender = 'm'
        self.model = None
        self._init_model()
        self.model.pose[0] = np.pi

        # Initializing the camera control widget and connects it to the camera button
        self.camera_widget = Ui_CameraWidget(
            self.camera, self.frustum, self.draw)

        # Add UI elements for object manipulation
        self._add_object_controls()

        # Connecting the shape, expression, and pose sliders to their respective update methods
        for key, shape in self._shapes():
            shape.valueChanged[int].connect(
                lambda val, k=key: self._update_shape(k, val))
        for key, exp in self._expressions():
            exp.valueChanged[int].connect(
                lambda val, k=key: self._update_exp(k, val))
        for key, pose in self._poses():
            pose.valueChanged[float].connect(
                lambda val, k=key: self._update_pose(k, val))

        # Connecting the position sliders to the method that updates the model's position (x, y, z)
        self.pos_0.valueChanged[float].connect(
            lambda val: self._update_position(0, val))
        self.pos_1.valueChanged[float].connect(
            lambda val: self._update_position(1, val))
        self.pos_2.valueChanged[float].connect(
            lambda val: self._update_position(2, val))

        # Setting up the UI elements for selecting the model's gender and type
        self.model_choose.currentIndexChanged[int].connect(
            lambda val: self._update_model(val))

        # Connecting the reset buttons to their respective reset methods
        self.reset_pose.clicked.connect(self._reset_pose)
        self.reset_shape.clicked.connect(self._reset_shape)
        self.reset_expression.clicked.connect(self._reset_expression)
        self.reset_postion.clicked.connect(self._reset_position)
        
        self.btn_load_point_cloud.clicked.connect(self._toggle_point_cloud_visibility)

        # Connecting mouse and scroll events to methods that handle zooming and rotating the model
        self.canvas.wheelEvent = self._zoom
        self.canvas.mousePressEvent = self._mouse_begin
        self.canvas.mouseMoveEvent = self._move
        self.canvas.mouseReleaseEvent = self._mouse_end

        # Handle saving the files
        first_frame_name = self.first_name()

        self.action_save_screenshot.triggered.connect(
            self._save_screenshot_dialog)
        self.action_save_mesh.triggered.connect(self.save_skeleton)

        self.action_save_object_mesh = QtWidgets.QAction("Save Object Mesh", self)
        self.action_save_object_mesh.triggered.connect(self._save_object_mesh_dialog)

        self.view_joints.triggered.connect(self.draw)
        self.view_joint_ids.triggered.connect(self.draw)
        self.view_bones.triggered.connect(self.draw)
        self.view_skeleton.triggered.connect(self.draw)
        self.view_rendered_mesh.triggered.connect(self.draw)
        
        self.mesh = np.array([])
        
        self.can_render_mesh = True

        self._update_canvas = True

    def _cleanup_object_renderer(self):
        """Safely cleanup object renderer to prevent context conflicts"""
        if hasattr(self, 'object_rn') and self.object_rn is not None:
            try:
                print("Cleaning up object renderer...")
                
                # Add pyglet-specific cleanup
                import gc
                
                # Clear the renderer properly
                if hasattr(self.object_rn, 'delete') and callable(self.object_rn.delete):
                    self.object_rn.delete()
                elif hasattr(self.object_rn, 'cleanup') and callable(self.object_rn.cleanup):
                    self.object_rn.cleanup()
                    
                # Clear references
                self.object_rn = None
                self.object_light = None
                self.object_camera = None
                
                # Force garbage collection to help with pyglet cleanup
                gc.collect()
                
                print("Object renderer cleaned up successfully")
                
            except Exception as e:
                print(f"Warning: Error during object renderer cleanup: {e}")
                # Force clear references even if cleanup failed
                self.object_rn = None
                self.object_light = None
                self.object_camera = None

    def _frame_change_cleanup(self):
        """Clean up resources when changing frames to prevent context conflicts"""
        print("Performing frame change cleanup...")
        
        # Store object state if present
        object_state = None
        if self.object_model is not None:
            object_state = {
                'vertices': self.object_model.v_original.copy(),
                'faces': self.object_model.f.copy(),
                'rotation': self.object_model.rotation.copy(),
                'scale': self.object_model.scale.copy(),
                'original_trans': self.object_model.original_trans.copy(),
                'user_trans': self.object_model.user_trans.copy(),
                'object_name': self.object_model.object_name  # NEW: Preserve object name
            }
            
        # Clean up object renderer
        self._cleanup_object_renderer()
        
        # Clear object model temporarily
        self.object_model = None
        self.show_object = False
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("Frame change cleanup completed")
        return object_state

    def _restore_object_after_frame_change(self, object_state):
        """Restore object after frame change with new context"""
        if object_state is None:
            return
            
        try:
            print("Restoring object after frame change...")
            
            # Recreate object model with correct transformation
            vertices = object_state['vertices']
            faces = object_state['faces']
            object_name = object_state.get('object_name', 'unknown')  # NEW: Restore object name
            
            # Apply the correct transformation for current human position
            if self.model is not None and hasattr(self, 'pose_transl'):
                transformation = -self.pose_transl
                transformed_vertices = vertices + transformation
            else:
                transformed_vertices = vertices
                transformation = np.zeros(3)
            
            # Recreate object model
            self.object_model = SimpleObjectModel(vertices, faces, 
                                                original_trans=np.mean(vertices, axis=0),
                                                object_name=object_name)  # NEW: Pass object name
            
            # Set working vertices and transformations
            self.object_model.v = transformed_vertices
            self.object_model.centering_offset = transformation
            self.object_model.object_center = np.mean(transformed_vertices, axis=0)
            
            # Restore saved transformations
            self.object_model.rotation = object_state['rotation']
            self.object_model.scale = object_state['scale']
            self.object_model.user_trans = object_state['user_trans']
            
            # Update UI controls
            self.obj_pos_x.setValue(self.object_model.user_trans[0])
            self.obj_pos_y.setValue(self.object_model.user_trans[1])
            self.obj_pos_z.setValue(self.object_model.user_trans[2])
            
            self.obj_rot_x.setValue(np.degrees(self.object_model.rotation[0]))
            self.obj_rot_y.setValue(np.degrees(self.object_model.rotation[1]))
            self.obj_rot_z.setValue(np.degrees(self.object_model.rotation[2]))
            
            self.obj_scale_x.setValue(self.object_model.scale[0])
            self.obj_scale_y.setValue(self.object_model.scale[1])
            self.obj_scale_z.setValue(self.object_model.scale[2])
            
            # Reinitialize renderer with proper context handling
            self._init_object_renderer()
            
            # Enable visibility
            self.chk_show_object.setChecked(True)
            self.show_object = True
            
            print("Object restored successfully after frame change")
            
        except Exception as e:
            print(f"Error restoring object after frame change: {e}")
            self.object_model = None
            self.show_object = False

    def _init_model(self, gender=None):
        pose = None
        betas = None
        trans = None

        # Saving current model parameters
        if self.model is not None:
            pose = self.model.pose.r
            betas = self.model.betas.r
            trans = self.model.trans.r

        if gender == None:
            gender = self.model_gender
        else:
            self.model_gender = gender

        self.model = load_model(
            model_type=self.model_type, gender=self.model_gender)

        # Setting model's parameters
        if pose is not None:
            self.model.pose[:] = pose
            self.model.betas[:] = betas
            self.model.trans[:] = trans

        # Updating Lighting and Renderer with the New Model
        self.light.set(v=self.model, f=self.model.f, num_verts=len(self.model))
        self.rn.set(v=self.model, f=self.model.f)
            

        # Updating Camera and Joint Projections
        self.camera.set(v=self.model)
        self.joints2d.set(v=self.model.J_transformed)

        self.draw()

    def _init_camera(self, update_camera=False):
        # Getting canvas dimension
        w = self.canvas.width()
        h = self.canvas.height()

        if update_camera or w != self.frustum['width'] and h != self.frustum['height']:
            # Setting Camera Parameters
            self.camera.set(rt=np.array([self.camera_widget.rot_0.value(), 
                                         self.camera_widget.rot_1.value(), 
                                         self.camera_widget.rot_2.value()]),
                            
                            t=np.array([self.camera_widget.pos_0.value(), 
                                        self.camera_widget.pos_1.value(),
                                        self.camera_widget.pos_2.value()]),
                            
                            f=np.array([w, w]) * self.camera_widget.focal_len.value(),
                            
                            c=np.array([w, h]) / 2.,
                            
                            k=np.array([self.camera_widget.dist_0.value(), 
                                        self.camera_widget.dist_1.value(),
                                        self.camera_widget.dist_2.value(), 
                                        self.camera_widget.dist_3.value(),
                                        self.camera_widget.dist_4.value()]))

            # Updating the Viewing Frustum
            self.frustum['width'] = w
            self.frustum['height'] = h

            # Setting Lighting Position
            self.light.set(light_pos=Rodrigues(
                self.camera.rt).T.dot(self.camera.t) * -10.)
            self.rn.set(frustum=self.frustum, camera=self.camera)
            
            # Adjusting the OpenGL Matrix for Flipped Coordinates
            flipXRotation = np.array([[1.0, 0.0, 0.0, 0.0],
                                      [0.0, -1.0, 0., 0.0],
                                      [0.0, 0., -1.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0]])
            self.rn.camera.openglMat = flipXRotation
            
            # Setting Additional Renderer Parameters
            self.rn.glMode = 'glfw'
            self.rn.sharedWin = None
            self.rn.overdraw = True
            self.rn.nsamples = 8
            self.rn.msaa = False
            self.rn.initGL()
            self.rn.debug = False
            
            # Update object renderer camera if it exists
            if hasattr(self, 'object_rn') and self.object_rn is not None:
                try:
                    self.object_rn.camera = self.camera
                    self.object_rn.frustum = self.frustum
                    if hasattr(self.rn.camera, 'openglMat'):
                        self.object_rn.camera.openglMat = self.rn.camera.openglMat
                except Exception as e:
                    print(f"Warning: Could not update object renderer camera: {e}")

            self.draw()

    #  Adding visual annotations (such as bones, joints, and joint IDs) to the rendered image
    # zero center mesh to align with human
    def _draw_annotations(self, img):
        orig_img = None
        
        # Prevent recursive calls that could cause event loop issues
        if self._rendering_in_progress or self._frame_changing:
            return img, orig_img
        
        # Setting the camera parameters for the joints2d projection
        self.joints2d.set(t=self.camera.t, rt=self.camera.rt,
                          f=self.camera.f, c=self.camera.c, k=self.camera.k)
        
        self.pcd.set(t=self.camera.t, rt=self.camera.rt,
                          f=self.camera.f, c=self.camera.c, k=self.camera.k)

        # Drawing bones
        height = self.canvas.height()
        if self.view_bones.isChecked():
            kintree = self.model.kintree_table[:, 1:]
            for k in range(kintree.shape[1]):
                cv2.line(img, (int(self.joints2d.r[kintree[0, k], 0]), int(height - self.joints2d.r[kintree[0, k], 1])),
                         (int(self.joints2d.r[kintree[1, k], 0]), int(
                             height - self.joints2d.r[kintree[1, k], 1])),
                         (0.98, 0.98, 0.98), 3)

        # Drawing joints
        if self.view_joints.isChecked():
            for j in self.joints2d.r:
                jj = height - j[1]  # for opengl: flipx
                cv2.circle(img, (int(j[0]), int(jj)),
                           5, (0.38, 0.68, 0.15), -1)

        # Drawing joint IDs
        if self.view_joint_ids.isChecked():
            for k, j in enumerate(self.joints2d.r):
                jj = height - j[1]
                cv2.putText(img, str(k), (int(j[0]), int(
                    jj)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0.3, 0.23, 0.9), 2)
        
        # Drawing PointCloud
        if hasattr(self, 'point_cloud') and self.show_pcd:
            for p in self.pcd.r:
                pp = height - p[1]
                cv2.circle(img, (int(p[0]), int(pp)),
                            5, (0.38, 0.68, 0.15), -1)
                
        
                
        # Drawing JRDB annotation along with current annotation
        if hasattr(self, 'orig_img_path'):
            # Draw projected joints on actual img
            joints_3d = self.model.J_transformed + self.pose_transl
            joints_3d_proj = self.jrdb_projection(joints_3d[None])[0]
            
            min_x1, min_x2 = np.min(joints_3d_proj, 0)
            max_x1, max_x2 = np.max(joints_3d_proj, 0)
            
            min_x1 = max(0, min_x1-CROP_OFFSET)
            min_x2 = max(0, min_x2-CROP_OFFSET)
            max_x1 = min(752, max_x1+CROP_OFFSET)
            max_x2 = min(480, max_x2+CROP_OFFSET)
            
            x1 = min_x1
            y1 = min_x2
            x2 = max_x1
            y2 = max_x2
                        

            orig_img = cv2.imread(self.orig_img_path, cv2.IMREAD_UNCHANGED)
            if self.view_skeleton.isChecked() and hasattr(self, 'has_keypoints') and self.has_keypoints:
                orig_img = draw_pose_2d(orig_img, self.jrdb_2d_pose, None, None,
                            (255, 0, 0), if_jrdb=True)
                for j in joints_3d_proj:
                    cv2.circle(orig_img, (int(j[0]), int(j[1])),
                                3, (0, 0, 255), -1)
                    kintree = self.model.kintree_table[:, 1:]
                    for k in range(kintree.shape[1]):
                        cv2.line(orig_img, (int(joints_3d_proj[kintree[0, k], 0]), int(joints_3d_proj[kintree[0, k], 1])),
                                (int(joints_3d_proj[kintree[1, k], 0]), int(
                                    joints_3d_proj[kintree[1, k], 1])),
                                (0, 0, 255), 2)
            orig_img = orig_img[int(y1):int(y2), int(x1):int(x2)]
            # Rendering mesh
            if self.view_rendered_mesh.isChecked() and self.can_render_mesh:
                try:
                    self._rendering_in_progress = True
                    self.render_mesh_on_image()
                except Exception as e:
                    print(f"Error in mesh rendering: {e}")
                finally:
                    self._rendering_in_progress = False

        return img, orig_img
    
    def _export_complete_scene(self):
        """Export both human and object meshes as a single .obj file"""
        if self.model is None:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No human model loaded.")
            return
            
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export Complete Scene',
            ROOT + "output/complete_scene.obj",
            'Mesh (*.obj)')
            
        if filename:
            try:
                vertex_offset = 0
                
                with open(filename, 'w') as fp:
                    # Write human mesh
                    fp.write("# Human mesh\n")
                    human_verts = self.model.r + self.pose_transl
                    for v in human_verts:
                        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                    
                    vertex_offset = len(human_verts)
                    
                    # Write object mesh if present
                    if self.object_model is not None:
                        fp.write("\n# Object mesh\n")
                        object_verts = self.object_model.r
                        for v in object_verts:
                            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                    
                    # Write human faces
                    fp.write("\n# Human faces\n")
                    for f in self.model.f + 1:
                        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
                    
                    # Write object faces if present
                    if self.object_model is not None:
                        fp.write("\n# Object faces\n")
                        for f in self.object_model.f + vertex_offset + 1:
                            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
                
                print(f"Complete scene saved to: {filename}")
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Complete scene exported successfully to:\n{filename}")
                    
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Error", f"Failed to export scene: {str(e)}")
                

    def _add_object_controls(self):
        """Add UI controls for object manipulation"""
        # Add a groupbox for object controls - UPDATED HEIGHT FOR SYNC BUTTON
        self.object_group = QtWidgets.QGroupBox("Object Controls", self)
        self.object_group.setGeometry(10, 500, 350, 400)  # Increased height for new sync button
        
        # Load object button
        self.btn_load_object = QtWidgets.QPushButton("Load Object Mesh", self.object_group)
        self.btn_load_object.setGeometry(10, 20, 150, 30)
        self.btn_load_object.clicked.connect(self._load_object_mesh)
        
        # Show/Hide object checkbox
        self.chk_show_object = QtWidgets.QCheckBox("Show Object", self.object_group)
        self.chk_show_object.setGeometry(170, 20, 100, 30)
        self.chk_show_object.stateChanged.connect(self._toggle_object_visibility)
        
        # Object position controls (for alignment only - NOT saved)
        self.lbl_obj_pos = QtWidgets.QLabel("Position (alignment only):", self.object_group)
        self.lbl_obj_pos.setGeometry(10, 60, 150, 20)
        
        self.obj_pos_x = QtWidgets.QDoubleSpinBox(self.object_group)
        self.obj_pos_x.setGeometry(10, 80, 100, 30)
        self.obj_pos_x.setRange(-10.0, 10.0)
        self.obj_pos_x.setSingleStep(0.01)
        self.obj_pos_x.valueChanged.connect(lambda v: self._update_object_transform())
        
        self.obj_pos_y = QtWidgets.QDoubleSpinBox(self.object_group)
        self.obj_pos_y.setGeometry(120, 80, 100, 30)
        self.obj_pos_y.setRange(-10.0, 10.0)
        self.obj_pos_y.setSingleStep(0.01)
        self.obj_pos_y.valueChanged.connect(lambda v: self._update_object_transform())
        
        self.obj_pos_z = QtWidgets.QDoubleSpinBox(self.object_group)
        self.obj_pos_z.setGeometry(230, 80, 100, 30)
        self.obj_pos_z.setRange(-10.0, 10.0)
        self.obj_pos_z.setSingleStep(0.01)
        self.obj_pos_z.valueChanged.connect(lambda v: self._update_object_transform())
        
        # Object rotation controls (SAVED)
        self.lbl_obj_rot = QtWidgets.QLabel("Rotation (saved):", self.object_group)
        self.lbl_obj_rot.setGeometry(10, 120, 100, 20)
        
        self.obj_rot_x = QtWidgets.QDoubleSpinBox(self.object_group)
        self.obj_rot_x.setGeometry(10, 140, 100, 30)
        self.obj_rot_x.setRange(-180.0, 180.0)
        self.obj_rot_x.setSingleStep(1.0)
        self.obj_rot_x.valueChanged.connect(lambda v: self._update_object_transform())
        
        self.obj_rot_y = QtWidgets.QDoubleSpinBox(self.object_group)
        self.obj_rot_y.setGeometry(120, 140, 100, 30)
        self.obj_rot_y.setRange(-180.0, 180.0)
        self.obj_rot_y.setSingleStep(1.0)
        self.obj_rot_y.valueChanged.connect(lambda v: self._update_object_transform())
        
        self.obj_rot_z = QtWidgets.QDoubleSpinBox(self.object_group)
        self.obj_rot_z.setGeometry(230, 140, 100, 30)
        self.obj_rot_z.setRange(-180.0, 180.0)
        self.obj_rot_z.setSingleStep(1.0)
        self.obj_rot_z.valueChanged.connect(lambda v: self._update_object_transform())
        
        # Object scale controls (SAVED) - Now separate X, Y, Z
        self.lbl_obj_scale = QtWidgets.QLabel("Scale X,Y,Z (saved):", self.object_group)
        self.lbl_obj_scale.setGeometry(10, 180, 130, 20)
        
        # Scale labels
        self.lbl_scale_x = QtWidgets.QLabel("X:", self.object_group)
        self.lbl_scale_x.setGeometry(35, 200, 15, 30)
        self.lbl_scale_y = QtWidgets.QLabel("Y:", self.object_group)
        self.lbl_scale_y.setGeometry(145, 200, 15, 30)
        self.lbl_scale_z = QtWidgets.QLabel("Z:", self.object_group)
        self.lbl_scale_z.setGeometry(255, 200, 15, 30)
        
        # Scale X
        self.obj_scale_x = QtWidgets.QDoubleSpinBox(self.object_group)
        self.obj_scale_x.setGeometry(10, 220, 100, 25)
        self.obj_scale_x.setRange(0.001, 10.0)
        self.obj_scale_x.setSingleStep(0.01)
        self.obj_scale_x.setValue(1.0)
        self.obj_scale_x.valueChanged.connect(lambda v: self._update_object_transform())
        
        # Scale Y
        self.obj_scale_y = QtWidgets.QDoubleSpinBox(self.object_group)
        self.obj_scale_y.setGeometry(120, 220, 100, 25)
        self.obj_scale_y.setRange(0.001, 10.0)
        self.obj_scale_y.setSingleStep(0.01)
        self.obj_scale_y.setValue(1.0)
        self.obj_scale_y.valueChanged.connect(lambda v: self._update_object_transform())
        
        # Scale Z
        self.obj_scale_z = QtWidgets.QDoubleSpinBox(self.object_group)
        self.obj_scale_z.setGeometry(230, 220, 100, 25)
        self.obj_scale_z.setRange(0.001, 10.0)
        self.obj_scale_z.setSingleStep(0.01)
        self.obj_scale_z.setValue(1.0)
        self.obj_scale_z.valueChanged.connect(lambda v: self._update_object_transform())
        
        # Reset object button
        self.btn_reset_object = QtWidgets.QPushButton("Reset", self.object_group)
        self.btn_reset_object.setGeometry(10, 260, 80, 30)
        self.btn_reset_object.clicked.connect(self._reset_object_transform)
        
        # Save object mesh button
        self.btn_save_object = QtWidgets.QPushButton("Save Object Mesh", self.object_group)
        self.btn_save_object.setGeometry(100, 260, 150, 30)
        self.btn_save_object.clicked.connect(self._save_object_mesh_dialog)
        
        # NEW: Sync object scale button (SCALE SYNC ADDITION)
        self.btn_sync_scale = QtWidgets.QPushButton("Sync Scale Across Track", self.object_group)
        self.btn_sync_scale.setGeometry(10, 300, 200, 30)
        self.btn_sync_scale.setStyleSheet("background-color: #90EE90;")  # Light green
        self.btn_sync_scale.clicked.connect(self._sync_object_scale_across_track)
        

    def _load_object_mesh(self):
        """Load object from prealigned_object_meshes folder structure"""
        
        # Construct the base directory path for current session
        base_dir = os.path.join(
            ROOT, 
            "sequences",
            f"{SEQUENCE}@cam_",
            f"{SEQUENCE}@cam_{self.sensor}",
            self.track,
            self.current_frame
        )
        
        if not os.path.exists(base_dir):
            QtWidgets.QMessageBox.information(
                self, "No Objects", 
                f"No prealigned objects found for:\n{self.track}/{self.current_frame}")
            
            # Fall back to file dialog
            self._load_object_mesh_from_dialog()
            return
        
        # Find all object folders in the frame directory
        object_folders = []
        try:
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isdir(item_path):
                    # Look for the corresponding _posed.obj file
                    posed_file = os.path.join(item_path, f"{item}_posed.obj")
                    if os.path.exists(posed_file):
                        object_folders.append((item, posed_file))
            
            # Sort alphabetically (top to bottom)
            object_folders.sort(key=lambda x: x[0])
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error", f"Error scanning object directory: {str(e)}")
            return
        
        if not object_folders:
            QtWidgets.QMessageBox.information(
                self, "No Objects", 
                f"No valid objects found in:\n{base_dir}")
            
            # Fall back to file dialog
            self._load_object_mesh_from_dialog()
            return
        
        # If multiple objects, let user choose
        if len(object_folders) == 1:
            object_name, file_path = object_folders[0]
            self._load_specific_object_mesh(file_path, object_name)
        else:
            # Present choice dialog
            object_names = [obj[0] for obj in object_folders]
            object_name, ok = QtWidgets.QInputDialog.getItem(
                self, "Select Object", 
                f"Multiple objects found for {self.track}/{self.current_frame}.\nSelect one to load:",
                object_names, 0, False)
            
            if ok:
                # Find the selected object's file path
                for name, file_path in object_folders:
                    if name == object_name:
                        self._load_specific_object_mesh(file_path, object_name)
                        break

    def _load_object_mesh_from_dialog(self):
        """Fallback method to load object from file dialog (original method)"""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load Object Mesh', '', 
            'Mesh Files (*.obj *.ply *.stl *.off);;All Files (*.*)')
        
        if filename:
            object_name = os.path.splitext(os.path.basename(filename))[0]
            self._load_specific_object_mesh(filename, object_name)

    def _load_specific_object_mesh(self, filename, object_name):
        """Load a specific object mesh file"""
        try:
            # Load mesh using trimesh
            mesh = trimesh.load(filename)
            
            # Convert to numpy arrays
            vertices = np.array(mesh.vertices, dtype=np.float64)
            faces = np.array(mesh.faces, dtype=np.uint32)
            
            original_object_center = np.mean(vertices, axis=0)
            print(f"Loading {object_name} from: {filename}")
            print(f"Object center: {original_object_center}")
            
            # Calculate the correct transformation
            if self.model is not None and hasattr(self, 'pose_transl'):
                # In MeshLab/JRDB: Human is at (self.model.r + self.pose_transl)
                # In Tool: Human is at self.model.r
                # So transformation needed = -self.pose_transl
                
                human_jrdb_position = np.mean(self.model.r + self.pose_transl, axis=0)
                human_tool_position = np.mean(self.model.r, axis=0)
                
                # The transformation is simply the negative of pose_transl
                transformation = -self.pose_transl
                
                print(f"Human in JRDB coordinates: {human_jrdb_position}")
                print(f"Human in tool coordinates: {human_tool_position}")
                print(f"Transformation to apply: {transformation}")
                
                # Apply the same transformation to the object
                transformed_vertices = vertices + transformation
                
                new_object_center = np.mean(transformed_vertices, axis=0)
                print(f"Object transformed to tool position: {new_object_center}")
                
                # Verify relative positioning is preserved
                relative_in_jrdb = original_object_center - human_jrdb_position
                relative_in_tool = new_object_center - human_tool_position
                print(f"Relative position in JRDB: {relative_in_jrdb}")
                print(f"Relative position in tool: {relative_in_tool}")
                print(f"Relative positioning preserved: {np.allclose(relative_in_jrdb, relative_in_tool, atol=1e-6)}")
                
            else:
                print("No human model or JRDB translation found, loading at original position")
                transformed_vertices = vertices
                transformation = np.zeros(3)
            
            # Create object model with original vertices preserved
            self.object_model = SimpleObjectModel(vertices, faces, 
                                                original_trans=original_object_center,
                                                object_name=object_name)  # NEW: Pass object name
            
            # Set the working vertices to the transformed ones
            self.object_model.v = transformed_vertices
            self.object_model.centering_offset = transformation
            # Set rotation/scale center to the transformed position
            self.object_model.object_center = np.mean(transformed_vertices, axis=0)
            
            print(f"SUCCESS: {object_name} loaded with correct JRDB transformation!")
            print(f"Object will rotate/scale around: {self.object_model.object_center}")
            
            # Initialize object renderer
            self._init_object_renderer()
            if self.object_rn is not None:
                # Enable object visibility
                self.chk_show_object.setChecked(True)
                self.show_object = True
                
                # Set UI controls to zero/default
                self.obj_pos_x.setValue(0.0)
                self.obj_pos_y.setValue(0.0)
                self.obj_pos_z.setValue(0.0)
                self.obj_rot_x.setValue(0.0)
                self.obj_rot_y.setValue(0.0)
                self.obj_rot_z.setValue(0.0)
                self.obj_scale_x.setValue(1.0)
                self.obj_scale_y.setValue(1.0)
                self.obj_scale_z.setValue(1.0)
                
                # Reset transformations
                self.object_model.user_trans = np.zeros(3)
                self.object_model.rotation = np.zeros(3)
                self.object_model.scale = np.ones(3)
                
                self.draw()
                
                QtWidgets.QMessageBox.information(
                    self, "Success", f"Loaded {object_name} successfully!")
                
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Warning", "Could not initialize renderer for object mesh.")
                self.object_model = None
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error", f"Failed to load {object_name}: {str(e)}")
            self.object_model = None
            self.object_rn = None


    def _init_object_renderer(self):
        """Initialize renderer for object mesh with enhanced context management - ORIGINAL VERSION"""
        if self.object_model is None:
            return
            
        try:
            print("Initializing object renderer with enhanced context management...")
            
            # Get the transformed vertices
            v_object = self.object_model.r
            f_object = self.object_model.f
            
            # Create vertex colors (uniform gray color for all vertices)
            vc_object = np.ones((len(v_object), 3)) * 0.7  # Gray color
            
            # Ensure main renderer has proper window reference
            if not hasattr(self.rn, 'glfw_window') or self.rn.glfw_window is None:
                print("Main renderer missing window, reinitializing...")
                self.rn.initGL()
                
            # Force render once to ensure window creation
            try:
                _ = self.rn.r
                print("Main renderer window confirmed active")
            except Exception as e:
                print(f"Main renderer needs reinitialization: {e}")
                self.rn.initGL()
                _ = self.rn.r
            
            # Create the object renderer with enhanced sharing attempts
            self.object_rn = ColoredRenderer()
            
            # Create lighting for the object
            self.object_light = LambertianPointLight(
                vc=vc_object,
                light_color=np.array([1., 1., 1.])
            )
            
            # Set lighting
            self.object_light.set(
                v=v_object, 
                f=f_object, 
                num_verts=len(v_object),
                light_pos=self.light.light_pos
            )
            
            # Set basic parameters
            self.object_rn.set(
                v=v_object,
                f=f_object,
                vc=self.object_light,
                bgcolor=np.ones(3),
                camera=self.camera,
                frustum=self.frustum
            )
            
            # Set OpenGL parameters
            self.object_rn.glMode = 'glfw'
            self.object_rn.overdraw = True
            self.object_rn.nsamples = 8
            self.object_rn.msaa = False
            
            # Enhanced context sharing with multiple fallback methods
            shared_context_found = False
            
            # Method 1: Try glfw_window
            if hasattr(self.rn, 'glfw_window') and self.rn.glfw_window is not None:
                try:
                    self.object_rn.sharedWin = self.rn.glfw_window
                    shared_context_found = True
                    print("✓ Context sharing method 1: glfw_window")
                except:
                    pass
            
            # Method 2: Try window attribute
            if not shared_context_found and hasattr(self.rn, 'window') and self.rn.window is not None:
                try:
                    self.object_rn.sharedWin = self.rn.window
                    shared_context_found = True
                    print("✓ Context sharing method 2: window")
                except:
                    pass
            
            # Method 3: Try to get context through OpenGL directly
            if not shared_context_found:
                try:
                    import glfw
                    current_context = glfw.get_current_context()
                    if current_context:
                        self.object_rn.sharedWin = current_context
                        shared_context_found = True
                        print("✓ Context sharing method 3: current_context")
                except:
                    pass
            
            # Method 4: Independent context as fallback
            if not shared_context_found:
                print("⚠ No shared context available, using independent context")
                self.object_rn.sharedWin = None
            
            # Initialize OpenGL
            try:
                self.object_rn.initGL()
                self.object_rn.debug = False
                
                # Test render to verify initialization
                _ = self.object_rn.r
                
                print("✓ Object renderer initialized successfully")
                
            except Exception as init_error:
                print(f"✗ Object renderer initialization failed: {init_error}")
                
                # Fallback: try without any context sharing
                print("Attempting fallback initialization...")
                self.object_rn.sharedWin = None
                try:
                    self.object_rn.initGL()
                    _ = self.object_rn.r
                    print("✓ Object renderer initialized with fallback method")
                except Exception as fallback_error:
                    print(f"✗ Fallback initialization also failed: {fallback_error}")
                    self.object_rn = None
                    return
            
            # Create camera for object projections
            self.object_camera = ProjectPoints(rt=np.zeros(3), t=np.zeros(3))
            self.object_camera.set(v=v_object)
            
            print("Object renderer setup completed successfully")
            
        except Exception as e:
            print(f"Error in _init_object_renderer: {e}")
            import traceback
            traceback.print_exc()
            self.object_rn = None
            self.object_light = None
            self.object_camera = None

    def _compute_face_normals(self, vertices, faces):
        """Compute face normals for simple shading"""
        normals = []
        for face in faces:
            v0, v1, v2 = vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            normals.append(normal)
        return np.array(normals)
    

    def _save_object_mesh_dialog(self):
        """Save the object mesh to the prealigned_object_meshes folder structure"""
        if self.object_model is None:
            QtWidgets.QMessageBox.warning(
                self, "Warning", "No object mesh loaded to save.")
            return
        
        # Ensure object model is synchronized with current UI slider values
        #self._update_object_transform()
        
        # Use the stored object name if available, otherwise ask user
        if self.object_model.object_name:
            object_name = self.object_model.object_name
        else:
            object_name, ok = QtWidgets.QInputDialog.getText(
                self, 'Object Name', 'Enter object name (e.g., "controller"):')
            
            if not ok or not object_name.strip():
                return
            
            object_name = object_name.strip()
            # Update the object model with the name
            self.object_model.object_name = object_name
        
        # Construct the save path based on current session parameters
        save_dir = os.path.join(
            ROOT, 
            "sequences",
            f"{SEQUENCE}@cam_",
            f"{SEQUENCE}@cam_{self.sensor}",
            self.track,
            self.current_frame,
            object_name
        )
        
        # Create directories if they don't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Construct the filename
        filename = os.path.join(save_dir, f"{object_name}_posed.obj")
        
        try:
            # Get original vertices WITH user adjustments applied to JRDB position
            transformed_vertices = self.object_model.get_original_transformed_vertices_with_position()
            faces = self.object_model.f
            
            print(f"Saving object mesh to: {filename}")
            print(f"  Applied rotation: {np.degrees(self.object_model.rotation)} degrees")
            print(f"  Applied scale X,Y,Z: {self.object_model.scale}")
            
            # Save as .obj file
            with open(filename, 'w') as fp:
                # Write vertices
                for v in transformed_vertices:
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                
                # Write faces (1-indexed for .obj format)
                for f in faces + 1:
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
            
            print(f"Object mesh saved to: {filename}")
            
            # NEW: Update transformation_info.json with composed transformations
            self._update_transformation_info(save_dir, object_name)
            
            # NEW: Update sequence progress tracking
            self._update_sequence_progress(object_name)
            
            # Store scale RATIOS for potential syncing (FIXED APPROACH)
            if self.object_model is not None:
                # Store the ratio of current scale to original scale (1.0, 1.0, 1.0)
                original_scale = np.ones(3)  # All objects start at scale 1.0
                self.stored_scale_for_sync = {
                    'scale_ratios': self.object_model.scale / original_scale,
                    'original_scale': original_scale.copy(),
                    'final_scale': self.object_model.scale.copy()
                }
                print(f"Stored scale ratios for sync: {self.stored_scale_for_sync['scale_ratios']}")
                print(f"  (These ratios will be applied to each object's current scale)")
            
            QtWidgets.QMessageBox.information(
                self, "Success", 
                f"Object mesh saved successfully to:\n{filename}\n\n"
                f"Applied transformations:\n"
                f"• Rotation: {np.degrees(self.object_model.rotation)}\n"
                f"• Scale X,Y,Z: {self.object_model.scale}")
                
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error", f"Failed to save object mesh: {str(e)}")

    def _update_transformation_info(self, save_dir, object_name):
        """Update transformation_info.json with composed transformations from user edits"""
        import json
        
        # Path to transformation_info.json
        transformation_path = os.path.join(save_dir, "transformation_info.json")
        
        # Get current user transformations
        user_translation = self.object_model.user_trans.copy()
        user_rotation = self.object_model.rotation.copy()  # Euler angles
        user_scale = self.object_model.scale.copy()
        
        print(f"\nUpdating transformation_info.json:")
        print(f"  User translation: {user_translation}")
        print(f"  User rotation (degrees): {np.degrees(user_rotation)}")
        print(f"  User scale: {user_scale}")
        
        try:
            # Load existing transformation if it exists
            if os.path.exists(transformation_path):
                print(f"  Loading existing transformation from: {transformation_path}")
                with open(transformation_path, 'r') as f:
                    existing_data = json.load(f)
                
                # Extract existing transformation parameters
                existing_translation = np.array(existing_data.get('translation', [0.0, 0.0, 0.0]))
                existing_rodrigues = np.array(existing_data.get('rotation_rodrigues', [0.0, 0.0, 0.0]))
                existing_scale = np.array(existing_data.get('scale', [1.0, 1.0, 1.0]))
                
                print(f"  Existing translation: {existing_translation}")
                print(f"  Existing rotation (rodrigues): {existing_rodrigues}")
                print(f"  Existing scale: {existing_scale}")
                
            else:
                print(f"  No existing transformation found, creating new one")
                existing_data = {}
                existing_translation = np.zeros(3)
                existing_rodrigues = np.zeros(3)
                existing_scale = np.ones(3)
            
            # Compose transformations: new_transform = user_transform * existing_transform
            
            # 1. Compose Translation: simply add translations
            new_translation = existing_translation + user_translation
            
            # 2. Compose Scale: multiply scale factors
            new_scale = existing_scale * user_scale
            
            # 3. Compose Rotation: multiply rotation matrices
            if np.allclose(existing_rodrigues, 0.0) and np.allclose(user_rotation, 0.0):
                # Both rotations are zero
                new_rodrigues = np.zeros(3)
            elif np.allclose(existing_rodrigues, 0.0):
                # Only user rotation exists (user_rotation is euler angles in radians)
                # Convert euler to rodrigues - user_rotation is actually used as rodrigues in UI code
                new_rodrigues = user_rotation.copy()
            elif np.allclose(user_rotation, 0.0):
                # Only existing rotation exists
                new_rodrigues = existing_rodrigues.copy()
            else:
                # Both rotations exist - compose them
                # existing_rodrigues is rodrigues vector, user_rotation is treated as rodrigues in UI
                R_existing = cv2.Rodrigues(existing_rodrigues)[0]
                R_user = cv2.Rodrigues(user_rotation)[0] 
                R_combined = R_user @ R_existing
                # Convert combined rotation matrix back to rodrigues vector
                new_rodrigues = cv2.Rodrigues(R_combined)[0].flatten()
            
            # Convert rodrigues to euler for display
            if not np.allclose(new_rodrigues, 0.0):
                R_combined_matrix = cv2.Rodrigues(new_rodrigues)[0]
                # Extract euler angles from rotation matrix (ZYX order)
                sy = np.sqrt(R_combined_matrix[0,0] * R_combined_matrix[0,0] + R_combined_matrix[1,0] * R_combined_matrix[1,0])
                singular = sy < 1e-6
                if not singular:
                    x = np.arctan2(R_combined_matrix[2,1], R_combined_matrix[2,2])
                    y = np.arctan2(-R_combined_matrix[2,0], sy)
                    z = np.arctan2(R_combined_matrix[1,0], R_combined_matrix[0,0])
                else:
                    x = np.arctan2(-R_combined_matrix[1,2], R_combined_matrix[1,1])
                    y = np.arctan2(-R_combined_matrix[2,0], sy)
                    z = 0
                new_euler_degrees = np.degrees([x, y, z])
            else:
                new_euler_degrees = np.zeros(3)
            
            print(f"  Composed translation: {new_translation}")
            print(f"  Composed rotation (degrees): {new_euler_degrees}")
            print(f"  Composed scale: {new_scale}")
            
            # Update the transformation data
            existing_data.update({
                'translation': new_translation.tolist(),
                'rotation_euler_degrees': new_euler_degrees.tolist(),
                'rotation_rodrigues': new_rodrigues.tolist(),
                'scale': new_scale.tolist()
            })
            
            # Update metadata if it doesn't exist
            if 'metadata' not in existing_data:
                existing_data['metadata'] = {}
            
            existing_data['metadata'].update({
                'sensor_id': self.sensor,
                'track_id': int(self.track.replace('track_', '')),
                'frame_id': int(self.current_frame),
                'object_name': object_name,
                'alignment_name': object_name
            })
            
            # Save the updated transformation
            with open(transformation_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            print(f"  ✅ Updated transformation_info.json: {transformation_path}")
            
        except Exception as e:
            print(f"  ❌ Error updating transformation_info.json: {e}")
            import traceback
            traceback.print_exc()

    def _update_sequence_progress(self, object_name):
        """Update sequence progress JSON to track edited meshes"""
        import json
        
        # Construct progress file path: human_model_viewer/sequences/{SEQUENCE}@cam_/progress.json
        sequences_dir = os.path.join(ROOT, "sequences", f"{SEQUENCE}@cam_")
        progress_path = os.path.join(sequences_dir, "progress.json")
        
        # Create sequences directory if it doesn't exist
        os.makedirs(sequences_dir, exist_ok=True)
        
        # Current session info
        sensor_key = f"sensor_{self.sensor}"
        track_key = self.track  # e.g., "track_9" 
        current_frame = self.current_frame  # e.g., "000000"
        
        print(f"\nUpdating sequence progress:")
        print(f"  Sequence: {SEQUENCE}@cam_")
        print(f"  Sensor: {sensor_key}")
        print(f"  Track: {track_key}")
        print(f"  Object: {object_name}")
        print(f"  Frame: {current_frame}")
        
        try:
            # Load existing progress data
            if os.path.exists(progress_path):
                with open(progress_path, 'r') as f:
                    progress_data = json.load(f)
                print(f"  Loaded existing progress from: {progress_path}")
            else:
                progress_data = {}
                print(f"  Creating new progress file: {progress_path}")
            
            # Initialize sensor if doesn't exist
            if sensor_key not in progress_data:
                progress_data[sensor_key] = {}
            
            # Initialize track if doesn't exist
            if track_key not in progress_data[sensor_key]:
                progress_data[sensor_key][track_key] = []
            
            # Find existing alignment entry for this object
            alignment_entry = None
            for entry in progress_data[sensor_key][track_key]:
                if entry.get('alignment') == object_name:
                    alignment_entry = entry
                    break
            
            # Create new alignment entry if doesn't exist
            if alignment_entry is None:
                alignment_entry = {
                    'alignment': object_name,
                    'frames': []
                }
                progress_data[sensor_key][track_key].append(alignment_entry)
                print(f"  Created new alignment entry for: {object_name}")
            else:
                print(f"  Found existing alignment entry for: {object_name}")
            
            # Add current frame if not already present
            if current_frame not in alignment_entry['frames']:
                alignment_entry['frames'].append(current_frame)
                alignment_entry['frames'].sort()  # Keep frames sorted
                print(f"  Added frame {current_frame} to {object_name}")
            else:
                print(f"  Frame {current_frame} already exists for {object_name}")
            
            # Save updated progress
            with open(progress_path, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            print(f"  ✅ Updated sequence progress: {progress_path}")
            print(f"  Current frames for {object_name}: {alignment_entry['frames']}")
            
        except Exception as e:
            print(f"  ❌ Error updating sequence progress: {e}")
            import traceback
            traceback.print_exc()

    # UPDATED: Scale sync method with relative scaling (FIXED APPROACH)
    def _sync_object_scale_across_track(self):
        """Apply stored scale ratios to all other instances of the same object in the current track"""
        if (not hasattr(self, 'stored_scale_for_sync') or 
            self.stored_scale_for_sync is None or 
            'scale_ratios' not in self.stored_scale_for_sync):
            QtWidgets.QMessageBox.warning(
                self, "No Stored Scale", 
                "No scale data stored. Please save an object mesh first.")
            return
        
        if self.object_model is None:
            QtWidgets.QMessageBox.warning(
                self, "No Object", "No object currently loaded.")
            return
        
        object_name = self.object_model.object_name
        if not object_name:
            QtWidgets.QMessageBox.warning(
                self, "No Object Name", "Current object has no name.")
            return
        
        # Base directory for the current track
        track_base_dir = os.path.join(
            ROOT, 
            "prealigned_object_meshes",
            f"{SEQUENCE}@cam_",
            f"{SEQUENCE}@cam_{self.sensor}",
            self.track
        )
        
        if not os.path.exists(track_base_dir):
            QtWidgets.QMessageBox.warning(
                self, "Track Not Found", f"Track directory not found: {track_base_dir}")
            return
        
        # Store scale ratios before clearing for message
        applied_ratios = self.stored_scale_for_sync['scale_ratios'].copy()
        
        # Find all frames and apply scale ratios
        frames_processed = 0
        frames_skipped = 0
        
        try:
            for frame_dir in os.listdir(track_base_dir):
                frame_path = os.path.join(track_base_dir, frame_dir)
                if not os.path.isdir(frame_path):
                    continue
                
                # Skip current frame to avoid overwriting the object we just saved
                if frame_dir == self.current_frame:
                    frames_skipped += 1
                    continue
                
                object_dir = os.path.join(frame_path, object_name)
                object_file = os.path.join(object_dir, f"{object_name}_posed.obj")
                
                if not os.path.exists(object_file):
                    frames_skipped += 1
                    continue
                
                # Apply scale ratios to this object
                if self._apply_scale_ratios_to_object_file(object_file, self.stored_scale_for_sync['scale_ratios']):
                    frames_processed += 1
                else:
                    frames_skipped += 1
        
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error", f"Error during sync: {str(e)}")
            return
        
        # Clear stored scale after successful sync
        self.stored_scale_for_sync = None
        
        QtWidgets.QMessageBox.information(
            self, "Sync Complete", 
            f"Scale sync completed!\n"
            f"Processed: {frames_processed} frames\n" 
            f"Skipped: {frames_skipped} frames\n"
            f"Applied scale ratios X,Y,Z: {applied_ratios}\n"
            f"(Each object's scale was multiplied by these ratios)")

    # UPDATED: Helper method for relative scaling (FIXED APPROACH)
    def _apply_scale_ratios_to_object_file(self, obj_file_path, scale_ratios):
        """Apply scale ratios to an .obj file (multiply current scale by ratios)"""
        try:
            # Load mesh using trimesh
            mesh = trimesh.load(obj_file_path)
            vertices = np.array(mesh.vertices, dtype=np.float64)
            faces = np.array(mesh.faces)
            
            # Calculate center of vertices
            center = np.mean(vertices, axis=0)
            
            # Apply scale ratios relative to center
            # This multiplies the current scale by the ratios, preserving coordinate system
            centered_vertices = vertices - center
            scaled_vertices = centered_vertices * scale_ratios  # Element-wise multiplication
            final_vertices = scaled_vertices + center
            
            # Save back to .obj file
            with open(obj_file_path, 'w') as fp:
                # Write vertices
                for v in final_vertices:
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                
                # Write faces (1-indexed for .obj format)
                for f in faces + 1:
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
            
            print(f"Applied scale ratios {scale_ratios} to {obj_file_path}")
            return True
            
        except Exception as e:
            print(f"Error applying scale ratios to {obj_file_path}: {e}")
            return False

    def _toggle_object_visibility(self, state):
        """Toggle object mesh visibility"""
        self.show_object = (state == 2)  # Qt.Checked = 2
        self.draw()

    def _update_object_transform(self):
        """Update object transformation (user adjustments for alignment only)"""
        if self.object_model is None:
            return
            
        # Update user translation adjustments (for alignment, not saved)
        self.object_model.user_trans[0] = self.obj_pos_x.value()
        self.object_model.user_trans[1] = self.obj_pos_y.value()
        self.object_model.user_trans[2] = self.obj_pos_z.value()
        
        # Update rotation (convert degrees to radians) - SAVED
        self.object_model.rotation[0] = np.radians(self.obj_rot_x.value())
        self.object_model.rotation[1] = np.radians(self.obj_rot_y.value())
        self.object_model.rotation[2] = np.radians(self.obj_rot_z.value())
        
        # Update scale (separate X, Y, Z) - SAVED
        self.object_model.scale[0] = self.obj_scale_x.value()
        self.object_model.scale[1] = self.obj_scale_y.value()
        self.object_model.scale[2] = self.obj_scale_z.value()
        
        # Update renderer with new vertices and lighting
        if self.object_rn is not None:
            try:
                v_object = self.object_model.r
                
                # Update vertex colors
                vc_object = np.ones((len(v_object), 3)) * 0.7
                
                # Update lighting with new vertex positions
                if hasattr(self, 'object_light') and self.object_light is not None:
                    self.object_light.set(
                        v=v_object, 
                        f=self.object_model.f, 
                        num_verts=len(v_object),
                        light_pos=self.light.light_pos
                    )
                    
                    self.object_rn.set(
                        v=v_object,
                        f=self.object_model.f,
                        vc=self.object_light
                    )
                else:
                    self.object_rn.v = v_object
                    self.object_rn.vc = vc_object
                
                # Update camera if needed
                if self.object_camera is not None:
                    self.object_camera.set(v=v_object)
                    
            except Exception as e:
                print(f"Error updating object transform: {e}")
        
        self.draw()

    def _reset_object_transform(self):
        """Reset object transformation to default"""
        self.obj_pos_x.setValue(0.0)
        self.obj_pos_y.setValue(0.0)
        self.obj_pos_z.setValue(0.0)
        self.obj_rot_x.setValue(0.0)
        self.obj_rot_y.setValue(0.0)
        self.obj_rot_z.setValue(0.0)
        self.obj_scale_x.setValue(1.0)
        self.obj_scale_y.setValue(1.0)
        self.obj_scale_z.setValue(1.0)
        self._update_object_transform()

    def _load_saved_object_transform(self, npy_data):
        """Load saved object with correct JRDB transformation"""
        if 'has_object' in npy_data and npy_data['has_object']:
            try:
                # Get saved data
                original_file_trans = npy_data.get('object_original_trans', np.zeros(3))
                original_vertices = npy_data['object_vertices']
                
                # Try to get object name if stored (may not exist in older saves)
                object_name = npy_data.get('object_name', 'unknown')  # NEW: Load object name if available
                
                # Apply the correct transformation
                if self.model is not None and hasattr(self, 'pose_transl'):
                    # Same transformation as fresh loading: -self.pose_transl
                    transformation = -self.pose_transl
                    transformed_vertices = original_vertices + transformation
                    
                    print(f"Loading saved object with transformation: {transformation}")
                else:
                    transformed_vertices = original_vertices
                    transformation = np.zeros(3)
                
                # Create object model preserving original vertices
                self.object_model = SimpleObjectModel(original_vertices, npy_data['object_faces'], 
                                                    original_trans=np.mean(original_vertices, axis=0),
                                                    object_name=object_name)  # NEW: Pass object name
                
                # Set working vertices to transformed position
                self.object_model.v = transformed_vertices
                self.object_model.centering_offset = transformation
                # Set rotation/scale center to transformed position
                self.object_model.object_center = np.mean(transformed_vertices, axis=0)
                
                # Apply saved transformations (rotation and scale only)
                self.object_model.rotation = npy_data['object_rotation']
                self.object_model.scale = npy_data['object_scale']
                self.object_model.user_trans = np.zeros(3)
                
                # Update UI controls to show saved rotation and scale
                self.obj_pos_x.setValue(0.0)
                self.obj_pos_y.setValue(0.0)
                self.obj_pos_z.setValue(0.0)
                
                self.obj_rot_x.setValue(np.degrees(self.object_model.rotation[0]))
                self.obj_rot_y.setValue(np.degrees(self.object_model.rotation[1]))
                self.obj_rot_z.setValue(np.degrees(self.object_model.rotation[2]))
                
                # Update scale UI controls (now separate X, Y, Z)
                self.obj_scale_x.setValue(self.object_model.scale[0])
                self.obj_scale_y.setValue(self.object_model.scale[1])
                self.obj_scale_z.setValue(self.object_model.scale[2])
                
                # Initialize renderer
                self._init_object_renderer()
                
                # Enable visibility
                self.chk_show_object.setChecked(True)
                self.show_object = True
                
                print(f"Loaded saved object with correct JRDB transformation and scale: {self.object_model.scale}")
                
            except Exception as e:
                print(f"Warning: Could not load saved object: {e}")

    def draw(self):
        """Modified draw method with proper context management and error handling - ORIGINAL VERSION"""
        if not self._update_canvas or self._rendering_in_progress or self._frame_changing:
            return
            
        try:
            # Render the Human Mesh
            if self.model is not None:
                # Try to get the rendered image with error handling
                try:
                    img = np.array(self.rn.r)
                except Exception as gl_error:
                    print(f"OpenGL rendering error: {gl_error}")
                    # Try to reinitialize the renderer
                    try:
                        print("Attempting to recover OpenGL context...")
                        self.rn.initGL()
                        img = np.array(self.rn.r)
                        print("✓ OpenGL context recovered")
                    except Exception as recover_error:
                        print(f"✗ Failed to recover OpenGL context: {recover_error}")
                        return  # Skip this frame

                # Render object mesh if visible
                if self.show_object and self.object_model is not None and self.object_rn is not None:
                    try:
                        # Ensure object renderer uses same camera settings
                        self.object_rn.camera = self.camera
                        self.object_rn.frustum = self.frustum
                        
                        # Update object lighting position to match main lighting
                        if hasattr(self, 'object_light') and self.object_light is not None:
                            self.object_light.set(light_pos=self.light.light_pos)
                        
                        # Also update the OpenGL matrix to match
                        if hasattr(self.rn, 'camera') and hasattr(self.rn.camera, 'openglMat'):
                            self.object_rn.camera.openglMat = self.rn.camera.openglMat
                        
                        # Render object with error handling
                        try:
                            object_img = np.array(self.object_rn.r)
                            
                            # Simple compositing - overlay object on human
                            # Only copy pixels that aren't background (white)
                            mask = np.any(object_img != 1.0, axis=2)
                            img[mask] = object_img[mask]
                            
                        except Exception as render_error:
                            print(f"Object render error: {render_error}")
                            # Continue with human-only rendering
                            
                    except Exception as obj_error:
                        print(f"Error in object rendering setup: {obj_error}")
                        # Continue without object if there's an error

                # Draw annotations and update UI
                img, jrdb_img = self._draw_annotations(img)

                self.canvas.setScaledContents(False)
                self.canvas.setPixmap(self._to_pixmap(img))
                
                if jrdb_img is not None:
                    self.jrdb_annot.setScaledContents(False)
                    self.jrdb_annot.setPixmap(self._to_pixmap(jrdb_img))
                
                if len(self.mesh) > 0:
                    self.jrdb_mesh.setScaledContents(False)
                    self.jrdb_mesh.setPixmap(self._to_pixmap(self.mesh))
                    
        except Exception as e:
            print(f"Critical error in draw(): {e}")
            # Don't crash the application, just skip this frame

    def save_skeleton(self):
        """Modified to save only rotation and scale for objects, preserve original location"""
        main_smpl_pose = []
        for _, pose in self._poses():
            main_smpl_pose.append(pose.value())
        main_smpl_pose = np.array(main_smpl_pose)
        
        main_smpl_shape = []
        for _, shape in self._shapes():
            main_smpl_shape.append(shape.value())
        main_smpl_shape = np.array(main_smpl_shape)

        print(self.model_gender)
        print(main_smpl_shape)
        
        info = {}
        info['smpl_pose'] = main_smpl_pose.astype(np.float64)[None]
        info['smpl_shape'] = ((main_smpl_shape - 50000.0) / 50000.0 * 5.0).astype(np.float64)[None]
        info['transl'] = self.main_smpl_trans + np.array([self.pos_0.value(), self.pos_1.value(), self.pos_2.value()])[None]
        info['bbox_3d'] = self.main_bbox_3d
        info['pc_inside_bbox'] = self.main_track_pc
        info['gender'] = self.model_gender
        
        # Add object information if present - ONLY save rotation and scale
        if self.object_model is not None:
            info['has_object'] = True
            # Save ORIGINAL FILE location (unchanged) if available, otherwise use original_trans
            original_location = getattr(self.object_model, 'original_file_trans', self.object_model.original_trans)
            info['object_original_trans'] = original_location.copy()
            # Save only rotation and scale (the changes we want to keep)
            info['object_rotation'] = self.object_model.rotation.copy()
            info['object_scale'] = self.object_model.scale.copy()  # This now saves 3-component scale
            # Save mesh data
            # Instead of saving: self.object_model.v.copy()
            # Save: self.object_model.v_original + self.object_model.user_trans
            info['object_vertices'] = self.object_model.v_original + self.object_model.user_trans
            info['object_faces'] = self.object_model.f.copy()
            # NEW: Save object name for future reference
            info['object_name'] = self.object_model.object_name
            
            print(f"Saving object with:")
            print(f"  Original location: {original_location}")
            print(f"  Rotation: {np.degrees(self.object_model.rotation)} degrees")
            print(f"  Scale X,Y,Z: {self.object_model.scale}")
            print(f"  Object name: {self.object_model.object_name}")
            print(f"  User translation adjustments (NOT saved): {self.object_model.user_trans}")
        else:
            info['has_object'] = False
        
        save_path = ROOT + "undistorted_videos/"+ SEQUENCE + "@cam_/" + SEQUENCE + "@cam_" + str(self.sensor) + "/per_frame_annot_fixedshape/" + self.track
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        save_path = os.path.join(save_path, self.current_frame + ".npy")
        print("saving to ", save_path)
        np.save(save_path, info)

    def jrdb_projection(self, points):
        x, y = points[:, :, 0]/points[:, :, 2], points[:, :, 1]/points[:, :, 2]
        # Get intrinsics params
        fx, fy = self.focal_length_jrdb
        cx, cy = self.camera_center_jrdb
        # Apply radial distortion and intrinsic parameters
        xd = fx * x + cx
        yd = fy * y + cy

        pts_2d = np.stack([xd, yd], -1)
        return pts_2d

    def showEvent(self, event):
        self._init_camera()
        super(self.__class__, self).showEvent(event)

    def resizeEvent(self, event):
        self._init_camera()
        super(self.__class__, self).resizeEvent(event)

    def closeEvent(self, event):
        # Clean up renderers before closing
        self._cleanup_object_renderer()
        self.camera_widget.close()
        super(self.__class__, self).closeEvent(event)

    def _open_config_dialog(self):
        # Set frame changing flag to prevent conflicts
        self._frame_changing = True
        
        # Store object state before frame change
        object_state = self._frame_change_cleanup()
        
        jrdb_dir = ROOT + "undistorted_videos/"+ SEQUENCE + "@cam_/" + SEQUENCE + "@cam_" + str(self.sensor) + "/per_frame_annot_fixedshape/" + self.track + "/" + self.current_frame
        jrdb_npy = np.load(jrdb_dir + ".npy", allow_pickle=True).item()
        
        # DEBUG: Print the structure of the NPY file
        print(f"DEBUG: NPY file keys: {list(jrdb_npy.keys())}")
        if 'smpl_shape' in jrdb_npy:
            print(f"DEBUG: smpl_shape type: {type(jrdb_npy['smpl_shape'])}")
            print(f"DEBUG: smpl_shape shape: {np.array(jrdb_npy['smpl_shape']).shape}")
            print(f"DEBUG: smpl_shape value: {jrdb_npy['smpl_shape']}")
        if 'smpl_pose' in jrdb_npy:
            print(f"DEBUG: smpl_pose type: {type(jrdb_npy['smpl_pose'])}")
            print(f"DEBUG: smpl_pose shape: {np.array(jrdb_npy['smpl_pose']).shape}")
        
        # SAFE GENDER LOOKUP WITH ERROR HANDLING
        track_id = self.track.split('_')[-1]
        print(f"Looking up gender info for track ID: {track_id}")
        
        if track_id in self.gender_info:
            gender_str = self.gender_info[track_id]
            print(f"Found gender info: {gender_str}")
            gender = 'm' if gender_str == 'Male' else 'f'
        else:
            print(f"WARNING: No gender info found for track {track_id}, defaulting to male")
            # Default to male if no gender info available
            gender = 'm'
            
            # Optional: Show available track IDs for debugging
            available_tracks = list(self.gender_info.keys())
            print(f"Available track IDs in gender_info: {available_tracks}")
        
        self.model_gender = gender
        print('track', track_id, 'gender:', self.model_gender)        
        
        self.orig_img_path = ROOT + "undistorted_images/image_" + str(self.sensor) + "/" + SEQUENCE + "/" + self.current_frame + ".jpg"        
        jrdb_keypoint_dir = ROOT + "pose_visualization_keypoints_2D_undistorted/" + SEQUENCE + "/" + "sensor_" + str(self.sensor) + "/" + self.track + "/" + str(int(self.current_frame)) + ".txt"
                                
        try:
            self.jrdb_2d_pose = np.loadtxt(jrdb_keypoint_dir)
            self.jrdb_2d_pose = self.jrdb_2d_pose[self.jrdb_to_op, :]
            self.has_keypoints = True
        except FileNotFoundError:
            print(f"Warning: Keypoint file not found: {jrdb_keypoint_dir}")
            print("2D pose skeleton overlay will be disabled.")
            self.jrdb_2d_pose = None
            self.has_keypoints = False

        self._update_canvas = False
        self._init_model(self.model_gender)
        
        # SAFE HANDLING OF SMPL SHAPE AND POSE DATA
        try:
            # Handle smpl_shape
            if 'smpl_shape' in jrdb_npy:
                shapes_raw = jrdb_npy['smpl_shape']
                
                # Convert to numpy array and ensure it's the right shape
                shapes_array = np.array(shapes_raw)
                
                # Handle different possible shapes
                if shapes_array.ndim == 0:
                    # It's a scalar, create default shape parameters
                    print("WARNING: smpl_shape is a scalar, using default shape parameters")
                    shapes = np.zeros(10)  # Default shape parameters
                elif shapes_array.ndim == 1:
                    # It's a 1D array, use as-is
                    shapes = shapes_array
                elif shapes_array.ndim == 2:
                    # It's a 2D array, take the first row
                    shapes = shapes_array[0]
                else:
                    print(f"WARNING: Unexpected smpl_shape dimensions: {shapes_array.ndim}, using defaults")
                    shapes = np.zeros(10)
                    
                # Ensure we have exactly 10 shape parameters
                if len(shapes) < 10:
                    print(f"WARNING: smpl_shape has {len(shapes)} parameters, padding to 10")
                    shapes = np.pad(shapes, (0, 10 - len(shapes)), 'constant')
                elif len(shapes) > 10:
                    print(f"WARNING: smpl_shape has {len(shapes)} parameters, truncating to 10")
                    shapes = shapes[:10]
                    
            else:
                print("WARNING: smpl_shape not found in NPY file, using default parameters")
                shapes = np.zeros(10)
                
            # Handle smpl_pose
            if 'smpl_pose' in jrdb_npy:
                poses_raw = jrdb_npy['smpl_pose']
                
                # Convert to numpy array and ensure it's the right shape
                poses_array = np.array(poses_raw)
                
                # Handle different possible shapes
                if poses_array.ndim == 0:
                    # It's a scalar, create default pose parameters
                    print("WARNING: smpl_pose is a scalar, using default pose parameters")
                    poses = np.zeros(72)  # Default pose parameters
                elif poses_array.ndim == 1:
                    # It's a 1D array, use as-is
                    poses = poses_array
                elif poses_array.ndim == 2:
                    # It's a 2D array, take the first row
                    poses = poses_array[0]
                else:
                    print(f"WARNING: Unexpected smpl_pose dimensions: {poses_array.ndim}, using defaults")
                    poses = np.zeros(72)
                    
                # Ensure we have exactly 72 pose parameters
                if len(poses) < 72:
                    print(f"WARNING: smpl_pose has {len(poses)} parameters, padding to 72")
                    poses = np.pad(poses, (0, 72 - len(poses)), 'constant')
                elif len(poses) > 72:
                    print(f"WARNING: smpl_pose has {len(poses)} parameters, truncating to 72")
                    poses = poses[:72]
                    
            else:
                print("WARNING: smpl_pose not found in NPY file, using default parameters")
                poses = np.zeros(72)
                
            print(f"Final shapes array: {shapes.shape} = {shapes}")
            print(f"Final poses array: {poses.shape}")
            
        except Exception as e:
            print(f"ERROR loading shape/pose data: {e}")
            print("Using default parameters")
            shapes = np.zeros(10)
            poses = np.zeros(72)
        
        # Handle position and translation
        position = np.array([0, 0, 0])
        if 'transl' in jrdb_npy:
            transl_raw = jrdb_npy['transl']
            transl_array = np.array(transl_raw)
            if transl_array.ndim == 2:
                self.pose_transl = transl_array[0]
                self.main_smpl_trans = transl_array
            elif transl_array.ndim == 1:
                self.pose_transl = transl_array
                self.main_smpl_trans = transl_array[None]  # Add batch dimension
            else:
                print(f"WARNING: Unexpected transl dimensions: {transl_array.ndim}")
                self.pose_transl = np.zeros(3)
                self.main_smpl_trans = np.zeros(3)[None]
        else:
            print("WARNING: transl not found in NPY file")
            self.pose_transl = np.zeros(3)
            self.main_smpl_trans = np.zeros(3)[None]
        
        # SAFE HANDLING OF MISSING KEYS
        # Handle missing bbox_3d
        if 'bbox_3d' in jrdb_npy:
            self.main_bbox_3d = jrdb_npy['bbox_3d']
            print(f"Found bbox_3d: {self.main_bbox_3d}")
        else:
            print(f"WARNING: bbox_3d not found in NPY file, using default empty array")
            self.main_bbox_3d = np.array([])
        
        # Handle missing pc_inside_bbox
        if 'pc_inside_bbox' in jrdb_npy:
            self.main_track_pc = jrdb_npy['pc_inside_bbox']
            print(f"Found pc_inside_bbox with {len(self.main_track_pc)} points")
        else:
            print(f"WARNING: pc_inside_bbox not found in NPY file, using default empty array")
            self.main_track_pc = np.array([])
        
        # Set shape parameters safely
        try:
            for key, shape in self._shapes():
                if key < len(shapes):
                    val = shapes[key] / 5.0 * 50000.0 + 50000.0
                    shape.setValue(val)
                else:
                    print(f"WARNING: Missing shape parameter {key}, using default")
                    shape.setValue(50000.0)  # Default value
        except Exception as e:
            print(f"ERROR setting shape parameters: {e}")
            # Set all to default
            for key, shape in self._shapes():
                shape.setValue(50000.0)
        
        # Set pose parameters safely
        try:
            for key, pose in self._poses():
                if key < len(poses):
                    val = poses[key]
                    pose.setValue(val)
                else:
                    print(f"WARNING: Missing pose parameter {key}, using default")
                    pose.setValue(0.0)  # Default value
        except Exception as e:
            print(f"ERROR setting pose parameters: {e}")
            # Set all to default
            for key, pose in self._poses():
                pose.setValue(0.0)

        self.pos_0.setValue(position[0])
        self.pos_1.setValue(position[1])
        self.pos_2.setValue(position[2])
        
        self._update_canvas = True
        
        # Restore object after human model is loaded
        if object_state is not None:
            self._restore_object_after_frame_change(object_state)
        else:
            # Check if there's a saved object transformation
            saved_path = ROOT + "output"+"/"+ SEQUENCE + "@cam_" + str(self.sensor) + "/interacting_objects/" + self.track + "/" + self.current_frame + ".npy"
            if os.path.exists(saved_path):
                saved_data = np.load(saved_path, allow_pickle=True).item()
                self._load_saved_object_transform(saved_data)
        
        # Load point cloud data but don't show it automatically
        self._load_point_cloud_dialog()
        
        # Clear frame changing flag
        self._frame_changing = False
        
        self._update_canvas = True
        self.draw()

    # saving a screenshot of the current 3D model view, including any annotations (like joints, bones, and joint IDs)
    def _save_screenshot_dialog(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save screenshot',
                                                            ROOT + "spin/out_pic/%s" % self.back_name.toPlainText(),
                                                            'Images (*.png *.jpg *.ppm)')
        if filename:
            img = np.array(self.rn.r)
            # Only add annotations if visualization options are enabled
            view_any = False
            if hasattr(self, 'view_joints') and self.view_joints.isChecked():
                view_any = True
            if hasattr(self, 'view_joint_ids') and self.view_joint_ids.isChecked():
                view_any = True
            if hasattr(self, 'view_bones') and self.view_bones.isChecked():
                view_any = True
                
            if view_any:
                img, _ = self._draw_annotations(img)
            cv2.imwrite(str(filename), np.uint8(img * 255))

    # saving the current 3D model as a mesh file in the ".obj" format
    def _save_mesh_dialog(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save mesh',
                                                            ROOT + "spin/out_mesh/%s" % self.back_name.toPlainText(),
                                                            'Mesh (*.obj)')
        if filename:
            with open(filename, 'w') as fp:
                for v in self.model.r:
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

                for f in self.model.f + 1:
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def _zoom(self, event):
        delta = -event.angleDelta().y() / 1200.0
        self.camera_widget.pos_2.setValue(
            self.camera_widget.pos_2.value() + delta)

    def _mouse_begin(self, event):
        self.can_render_mesh = False
        if event.button() == 4:  # middle
            self._moving = True
        elif event.button() == 1:  # left
            self._rotating = True
        self._mouse_begin_pos = event.pos()
        self.can_render_mesh = True

    def _mouse_end(self, event):
        self._moving = False
        self._rotating = False

    # Handling mouse drag and drop event
    def _move(self, event):
        self.can_render_mesh = False
        if self._moving:
            delta = event.pos() - self._mouse_begin_pos
            # Updating the Camera's Position
            self.camera_widget.pos_0.setValue(
                self.camera_widget.pos_0.value() + delta.x() / 1000.)
            self.camera_widget.pos_1.setValue(
                self.camera_widget.pos_1.value() + delta.y() / 1000.)
            self._mouse_begin_pos = event.pos()
            
        
        elif self._rotating:
            delta = event.pos() - self._mouse_begin_pos
            # Updating the Camera's Position
            self.camera_widget.rot_0.setValue(
                self.camera_widget.rot_0.value() + delta.y() / 300.)
            self.camera_widget.rot_1.setValue(
                self.camera_widget.rot_1.value() - delta.x() / 300.)
            self._mouse_begin_pos = event.pos()
        self.can_render_mesh = True

    def _show_camera_widget(self):
        self.camera_widget.show()
        self.camera_widget.raise_()

    # Updating the 3D model based on the user's selection
    def _update_model(self, id):
        self.model = None
        self.model_type = model_type_list[id]

        # Disabling Canvas Updates Temporarily
        self._update_canvas = False

        # Re-initializing the Light Source
        self.light = LambertianPointLight(vc=np.array(
            [0.98, 0.98, 0.98]), light_color=np.array([1., 1., 1.]))

        # Setting Renderer Parameters
        self.rn.set(glMode='glfw', bgcolor=np.ones(3), frustum=self.frustum, camera=self.camera, vc=self.light,
                    overdraw=False)

        # Re-initializing the Model
        self._init_model()
        self.model.pose[0] = np.pi
        self._init_camera(update_camera=True)

        # Resetting Model Parameter
        self._reset_shape()
        self._reset_expression()
        self._reset_pose()
        self._reset_position()

        self._update_canvas = True
        self.draw()

    # Updating the shape of the 3D model based on user input
    def _update_shape(self, id, val):
        val = (val - 50000) / 50000.0 * 5.0
        self.model.betas[id] = val
        self.draw()

    # Updating the expression-related parameters of the 3D model
    def _update_exp(self, id, val):
        val = (val - 50000) / 50000.0 * 5.0
        if self.model_type == 'smplx':
            self.model.betas[10 + id] = val
            self.draw()
        elif self.model_type == 'flame':
            self.model.betas[10 + id] = val
            self.draw()

    # Resetting the shape parameters of the 3D model
    def _reset_shape(self):
        jrdb_dir = ROOT + "undistorted_videos/"+ SEQUENCE + "@cam_/" + SEQUENCE + "@cam_" + str(self.sensor) + "/per_frame_annot_fixedshape/" + self.track + "/" + self.current_frame
        jrdb_npy = np.load(jrdb_dir + ".npy", allow_pickle=True).item()

        self._update_canvas = False
        
        shapes = jrdb_npy['smpl_shape'][0]
        #shapes = np.asarray([1.19550167,  0.62523977,  0.04172035, -0.22713715, -0.02500708, -0.05290442, -0.02488141,  0.0140115 ,  0.0112469 ,  0.02695486])

        self.pose_transl = jrdb_npy['transl'][0]
        
        for key, shape in self._shapes():
            val = shapes[key] / 5.0 * 50000.0 + 50000.0
            shape.setValue(val)

        self._update_canvas = True
        self.draw()

    # Resetting expression-related parameters of the 3D model
    def _reset_expression(self):
        if self.model_type != 'smplx' and self.model_type != 'flame':
            return
        self._update_canvas = False
        for _, exp in self._expressions():
            exp.setValue(50000)
        self._update_canvas = True
        self.draw()

    # Updating the pose of 3D model
    def _update_pose(self, id, val):
        # Scaling value to -π to π
        '''val = (val - 50000) / 50000.0 * np.pi

        if self.model_type == 'flame' and id >= 5 * 3:
            return'''

        self.model.pose[id] = val
        self.draw()

    # Resetting the pose value
    def _reset_pose(self):
        jrdb_dir = ROOT + "undistorted_videos/"+ SEQUENCE + "@cam_/" + SEQUENCE + "@cam_" + str(self.sensor) + "/per_frame_annot_fixedshape/" + self.track + "/" + self.current_frame
        jrdb_npy = np.load(jrdb_dir + ".npy", allow_pickle=True).item()
        
        poses = jrdb_npy['smpl_pose'][0]
        for key, pose in self._poses():
            # if key == 0:
            #     val = (poses[key]) / np.pi * 50000.0 + 50000.0
            # else:
            #     val = poses[key] / np.pi * 50000.0 + 50000.0
            val = poses[key]
            pose.setValue(val)

        self._update_canvas = True
        self.draw()

    def _update_object_centering(self):
        """Update object when human moves (maintain relative positioning)"""
        if self.object_model is None:
            return
        
        # Apply the correct transformation for current human position
        if self.model is not None and hasattr(self, 'pose_transl'):
            # Transformation is always -self.pose_transl
            new_transformation = -self.pose_transl
            
            # Apply transformation to object
            self.object_model.v = self.object_model.v_original + new_transformation
            self.object_model.centering_offset = new_transformation
            # Update rotation/scale center
            self.object_model.object_center = np.mean(self.object_model.v, axis=0)
            
            # Update renderer if it exists
            if hasattr(self, 'object_rn') and self.object_rn is not None:
                self._init_object_renderer()
            
            print(f"Object updated with transformation: {new_transformation}")
        
        self.draw()

    def _update_position(self, id, val):
        """Modified to keep object centered when human moves"""
        self.model.trans[id] = val
        
        # If object is loaded, update its centering to follow the human
        if hasattr(self, 'object_model') and self.object_model is not None:
            self._update_object_centering()
        else:
            self.draw()

    def _reset_position(self):
        self._update_canvas = False
        self.pos_0.setValue(0)
        self.pos_1.setValue(0)
        self.pos_2.setValue(0)
        self._update_canvas = True
        self.draw()

    def _poses(self):
        return enumerate([
            self.pose_0,
            self.pose_1,
            self.pose_2,
            self.pose_3,
            self.pose_4,
            self.pose_5,
            self.pose_6,
            self.pose_7,
            self.pose_8,
            self.pose_9,
            self.pose_10,
            self.pose_11,
            self.pose_12,
            self.pose_13,
            self.pose_14,
            self.pose_15,
            self.pose_16,
            self.pose_17,
            self.pose_18,
            self.pose_19,
            self.pose_20,
            self.pose_21,
            self.pose_22,
            self.pose_23,
            self.pose_24,
            self.pose_25,
            self.pose_26,
            self.pose_27,
            self.pose_28,
            self.pose_29,
            self.pose_30,
            self.pose_31,
            self.pose_32,
            self.pose_33,
            self.pose_34,
            self.pose_35,
            self.pose_36,
            self.pose_37,
            self.pose_38,
            self.pose_39,
            self.pose_40,
            self.pose_41,
            self.pose_42,
            self.pose_43,
            self.pose_44,
            self.pose_45,
            self.pose_46,
            self.pose_47,
            self.pose_48,
            self.pose_49,
            self.pose_50,
            self.pose_51,
            self.pose_52,
            self.pose_53,
            self.pose_54,
            self.pose_55,
            self.pose_56,
            self.pose_57,
            self.pose_58,
            self.pose_59,
            self.pose_60,
            self.pose_61,
            self.pose_62,
            self.pose_63,
            self.pose_64,
            self.pose_65,
            self.pose_66,
            self.pose_67,
            self.pose_68,
            self.pose_69,
            self.pose_70,
            self.pose_71,
        ])

    def _shapes(self):
        return enumerate([
            self.shape_0,
            self.shape_1,
            self.shape_2,
            self.shape_3,
            self.shape_4,
            self.shape_5,
            self.shape_6,
            self.shape_7,
            self.shape_8,
            self.shape_9,
        ])

    def _expressions(self):
        return enumerate([
            self.shape_10,
            self.shape_11,
            self.shape_12,
            self.shape_13,
            self.shape_14,
            self.shape_15,
            self.shape_16,
            self.shape_17,
            self.shape_18,
            self.shape_19,
        ])

    @staticmethod
    def _to_pixmap(im):
        if im.dtype == np.float32 or im.dtype == np.float64:
            im = np.uint8(im * 255)

        if len(im.shape) < 3 or im.shape[-1] == 1:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        qimg = QtGui.QImage(
            im, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)

        return QtGui.QPixmap(qimg)

    def first_name(self):
        pkl_folder_name = ROOT + '/spin/pkl_file'
        pkl_files = os.listdir(pkl_folder_name)
        first_pkl = pkl_files[0]
        first_pkl_name = first_pkl.split(".")[0]
        return first_pkl_name

    
    def _update_sensor(self, id):
        self.sensor = self.all_sensors[id - 1]
        
        cam_pos = np.array([0.0, 0.0, 3.0], dtype=np.float64)
        cam_rot = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        cam_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        cam_f = 1.0
        cam_c = np.array([0.5, 0.5])
        
        sns_number = "sensor_" + str(self.sensor)
        K = self.camera_config[sns_number]['undistorted_img_K']
        fx, _, cx = K[0]
        _, fy, cy = K[1]
        self.focal_length_jrdb = np.array([fx, fy], dtype=np.float64)
        self.camera_center_jrdb = np.array([cx, cy], dtype=np.float64)
        
        self.camera_widget.set_values(
            cam_pos, cam_rot, cam_f, cam_c, cam_dist)
        
        tmp = ROOT + "undistorted_videos/"+ SEQUENCE + "@cam_/" + SEQUENCE + "@cam_" + str(self.sensor) + "/per_frame_annot_fixedshape"
        self.all_tracks = os.listdir(tmp)

        # FILTER OUT SYSTEM FILES AND NON-TRACK FOLDERS
        print(f"Found tracks: {self.all_tracks}")
        
        # Filter to only include items that start with "track_"
        filtered_tracks = []
        for track in self.all_tracks:
            if track.startswith("track_") and "_" in track:
                try:
                    # Test if we can extract a valid track number
                    track_num = int(track.split("_")[-1])
                    filtered_tracks.append(track)
                    print(f"Valid track: '{track}', number: {track_num}")
                except ValueError:
                    print(f"Skipping invalid track format: '{track}'")
            else:
                print(f"Skipping non-track item: '{track}'")
        
        self.all_tracks = filtered_tracks
        
        # Convert to integers for sorting
        track_numbers = []
        for track in self.all_tracks:
            track_numbers.append(int(track.split("_")[-1]))
        
        track_numbers = sorted(track_numbers)
        
        # Convert back to track names
        self.all_tracks = [f"track_{num}" for num in track_numbers]
        
        self.track_choose.clear()
        self.track_choose.addItem("track")
        for item in self.all_tracks:
            self.track_choose.addItem(item)
    
    def _update_tracks(self, id):
        self.track = self.all_tracks[id - 1]
        tmp = ROOT + "undistorted_videos/"+ SEQUENCE + "@cam_/" + SEQUENCE + "@cam_" + str(self.sensor) + "/per_frame_annot_fixedshape/" + self.track
        all_files = sorted(os.listdir(tmp))  # Get all .npy files
        
        # Filter to only include frame 0 and multiples of 75
        filtered_frames = []
        for item in all_files:
            if item.endswith('.npy'):
                try:
                    frame_num = int(item[:-4])  # Remove .npy extension and convert to int
                    if frame_num == 0 or frame_num % FRAME_FREQ == 0:
                        filtered_frames.append(item)
                except ValueError:
                    # Skip files that don't have numeric names
                    continue
        
        self.all_frames = filtered_frames
        
        self.frame_choose.clear()
        self.frame_choose.addItem("frame id")
        
        for item in self.all_frames:
            self.frame_choose.addItem(item)
        
    def _update_frames(self, id):
        if len(self.all_frames) > 0 and id > 0:
            self.current_frame = self.all_frames[id - 1][:-4]
            self._open_config_dialog()
            # Point cloud will be loaded automatically in _open_config_dialog() but not displayed
    
    def _load_point_cloud_dialog(self):
        """Load point cloud data (user must explicitly enable visualization)"""
        # Load the point cloud data but don't automatically show it
        if hasattr(self, 'current_frame') and self.current_frame:
            try:
                file_path = ROOT + "pointclouds/upper_velodyne/" + SEQUENCE + "/" + self.current_frame + ".pcd"
                point_cloud_data = np.load(file_path, allow_pickle=True).item()
                self.point_cloud = point_cloud_data['pc_inside_bbox'] - self.pose_transl
                self.pcd.set(v=self.point_cloud)
                print("Point cloud data loaded (use menu to visualize)")
            except Exception as e:
                print(f"Could not load point cloud: {e}")
        
        # Only show if explicitly enabled (don't auto-toggle)
        if self.show_pcd:
            self.draw()

    def _toggle_point_cloud_visibility(self):
        """Toggle point cloud visibility (load data if needed)"""
        self.show_pcd = not self.show_pcd
        
        if self.show_pcd:
            # Load point cloud data if not already loaded
            self._load_point_cloud_dialog()
            print("Point cloud visualization: ON")
        else:
            print("Point cloud visualization: OFF")
            
        self.draw()

    def render_mesh_on_image(self):
        """Main method to render mesh overlay on camera image - using separate rendering like main 3D view"""
        if not RENDERER_AVAILABLE:
            print("Rendered mesh feature not available - Renderer class not found")
            return
            
        back_img = cv2.imread(self.orig_img_path)
        if back_img is None:
            print(f"Could not load image: {self.orig_img_path}")
            return
            
        img_h, img_w, _ = back_img.shape
        back_img = cv2.putText(back_img, 'frame_'+str(int(self.current_frame)),
                            (img_w-150, img_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                            
        joints_3d = self.model.J_transformed + self.pose_transl
        pose_2d = self.jrdb_projection(joints_3d[None])[0]
        min_x1, min_x2 = np.min(pose_2d, 0)
        max_x1, max_x2 = np.max(pose_2d, 0)
        
        min_x1 = max(0, min_x1-CROP_OFFSET)
        min_x2 = max(0, min_x2-CROP_OFFSET)
        max_x1 = min(752, max_x1+CROP_OFFSET)
        max_x2 = min(480, max_x2+CROP_OFFSET)
        
        x1 = min_x1
        y1 = min_x2
        x2 = max_x1
        y2 = max_x2
        
        # Start with background image
        result_img = back_img[:, :, ::-1].copy()  # Convert BGR to RGB
        
        print("Attempting mesh rendering using existing OpenGL context...")
        
        try:
            # Step 1: Render human mesh on camera image
            human_verts = self.model.r + self.pose_transl  # Human vertices in world coordinates
            human_faces = self.model.f
            
            print("Rendering human mesh...")
            human_result = self.render_single_mesh_on_image(
                human_verts, human_faces, result_img.copy(),
                self.focal_length_jrdb, 
                self.camera_center_jrdb[0], 
                self.camera_center_jrdb[1]
            )
            
            if human_result is not None:
                result_img = human_result
                print("Human mesh rendered successfully")
            else:
                print("Human mesh rendering failed")
                
            # Step 2: Render object mesh on top if present
            if (self.show_object and self.object_model is not None):
                print("Rendering object mesh...")
                object_verts = self.object_model.r + self.pose_transl
                object_faces = self.object_model.f
                
                object_result = self.render_single_mesh_on_image(
                    object_verts, object_faces, result_img.copy(),
                    self.focal_length_jrdb, 
                    self.camera_center_jrdb[0], 
                    self.camera_center_jrdb[1],
                    mesh_color=(0.9, 0.5, 0.1)  # Orange color for object
                )
                
                if object_result is not None:
                    # Composite object on top of human+background
                    # Only use object pixels where the object actually rendered
                    human_pixels = result_img
                    object_pixels = object_result
                    
                    # Create mask where object differs from background
                    background_color = np.array([back_img[0, 0, 2], back_img[0, 0, 1], back_img[0, 0, 0]]) / 255.0
                    object_mask = np.any(np.abs(object_pixels - background_color) > 0.1, axis=2)
                    
                    # Apply object where it's different from background
                    result_img[object_mask] = object_pixels[object_mask]
                    print("Object mesh composited successfully")
                else:
                    print("Object mesh rendering failed, showing only human")
            
            if result_img is not None:
                self.mesh = result_img[int(y1):int(y2), int(x1):int(x2)]
                # Convert back to BGR for display
                self.mesh = self.mesh[:, :, ::-1]
                print("Combined mesh rendering successful!")
            else:
                print("Mesh rendering returned None")
                self.mesh = np.array([])
                
        except Exception as e:
            print(f"Error in render_mesh_on_image: {e}")
            import traceback
            traceback.print_exc()
            self.mesh = np.array([])

    def render_single_mesh_on_image(self, verts, faces, background_image, focal_length, cx, cy, mesh_color=None):
        """Render a single mesh on background image"""
        if not RENDERER_AVAILABLE:
            print("Renderer not available")
            return None
        
        try:
            print(f"Rendering single mesh: vertices {verts.shape}, faces {faces.shape}")
            
            # Create renderer
            renderer = Renderer(focal_length=focal_length, img_w=752, img_h=480, cx=cx, cy=cy, faces=faces)
            
            # Wrap vertices in list as expected by renderer
            verts_wrapped = [verts]
            
            # Render the mesh
            if mesh_color is not None:
                # Use custom color if provided
                result = renderer.render_front_view_with_color(verts_wrapped, bg_img_rgb=background_image, mesh_color=mesh_color)
            else:
                # Use default color
                result = renderer.render_front_view(verts_wrapped, bg_img_rgb=background_image)
            
            # Clean up
            renderer.delete()
            
            return result
            
        except Exception as e:
            print(f"Error in render_single_mesh_on_image: {e}")
            import traceback
            traceback.print_exc()
            return None

    def render_mesh_main_thread(self, human_verts, human_faces, object_verts, object_faces, background_image, focal_length, cx, cy):
        """Render human and object meshes separately in the same scene"""
        if not RENDERER_AVAILABLE:
            print("Renderer not available")
            return None
        
        try:
            print(f"Main thread render: focal_length={focal_length}, cx={cx}, cy={cy}")
            print(f"Human vertices: {human_verts.shape}, Human faces: {human_faces.shape}")
            if object_verts is not None:
                print(f"Object vertices: {object_verts.shape}, Object faces: {object_faces.shape}")
            
            # Create renderer - we'll use it to create a scene and render both meshes
            renderer = Renderer(focal_length=focal_length, img_w=752, img_h=480, cx=cx, cy=cy, faces=human_faces)
            
            print("Calling render_scene_with_both_meshes...")
            
            # Render both meshes in the same scene
            front_view_after = renderer.render_scene_with_both_meshes(
                human_verts, human_faces,
                object_verts, object_faces,
                bg_img_rgb=background_image[:, :, ::-1].copy()
            )
            
            # Clean up
            renderer.delete()
            
            print("Main thread rendering completed successfully")
            return front_view_after[:, :, ::-1]
            
        except Exception as e:
            print(f"Error in render_mesh_main_thread: {e}")
            import traceback
            traceback.print_exc()
            return None

    def render_mesh_direct(self, verts, face, background_image, focal_length, cx, cy):
        """Legacy method - now redirects to main thread rendering"""
        return self.render_mesh_main_thread(verts, face, background_image, focal_length, cx, cy)

    def render_mesh_threaded(self, verts, face, background_image, focal_length, cx, cy):
        """Replaced with main thread rendering for Apple Silicon compatibility"""
        return self.render_mesh_main_thread(verts, face, background_image, focal_length, cx, cy)
    

    ### Add this method to check dependencies:
    def check_renderer_dependencies(self):
        """Check if all dependencies for rendered mesh are available"""
        try:
            import pyrender
            import trimesh
            print("✓ pyrender and trimesh available")
            
            if hasattr(self, 'focal_length_jrdb') and hasattr(self, 'camera_center_jrdb'):
                print("✓ Camera parameters available")
            else:
                print("✗ Camera parameters not set")
                
            if hasattr(self, 'orig_img_path') and os.path.exists(self.orig_img_path):
                print(f"✓ Original image available: {self.orig_img_path}")
            else:
                print("✗ Original image path not available")
                
            if RENDERER_AVAILABLE:
                print("✓ Renderer class imported successfully")
            else:
                print("✗ Renderer class not available")
                
        except ImportError as e:
            print(f"✗ Missing dependencies: {e}")

    def test_renderer_setup(self):
        """Test if the renderer is properly set up and working - DISABLED to prevent frame change issues"""
        print("\n=== Renderer Setup Test DISABLED ===")
        print("Skipping renderer test to prevent frame change conflicts")
        return True


def letterbox_image(img, input_dim):
    orig_w, orig_h = img.shape[1], img.shape[0]
    input_w, input_h = input_dim

    min_ratio = min(input_w / orig_w, input_h / orig_h)
    new_w = int(orig_w * min_ratio)
    new_h = int(orig_h * min_ratio)
    resized_image = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((input_dim[1], input_dim[0], 3),
                    128, np.uint8)

    start_h = (input_h - new_h) // 2
    start_w = (input_w - new_w) // 2
    canvas[start_h:start_h + new_h, start_w:start_w + new_w, :] = resized_image

    return canvas