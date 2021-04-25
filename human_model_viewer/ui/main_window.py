import sys
import numpy as np
import cv2
import os
import configparser

from PyQt5 import QtGui, QtWidgets
from opendr.camera import ProjectPoints, Rodrigues
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight

from .gen.main_window import Ui_MainWindow as Ui_MainWindow_Base
from .camera_widget import Ui_CameraWidget
from .util import *

model_type_list = ['smpl','smplx','flame']

class Ui_MainWindow(QtWidgets.QMainWindow, Ui_MainWindow_Base):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self._moving = False
        self._rotating = False
        self._mouse_begin_pos = None
        self._update_canvas = False

        self.camera = ProjectPoints(rt=np.zeros(3), t=np.zeros(3))
        self.joints2d = ProjectPoints(rt=np.zeros(3), t=np.zeros(3))
        self.frustum = {'near': 0.1, 'far': 1000., 'width': 100, 'height': 30}
        self.light = LambertianPointLight(vc=np.array([0.98, 0.98, 0.98]), light_color=np.array([1., 1., 1.]))
        # self.rn = ColoredRenderer(bgcolor=np.ones(3), frustum=self.frustum, camera=self.camera, vc=self.light,
                                  # overdraw=False)
        self.rn = ColoredRenderer()
        self.rn.set(glMode='glfw',bgcolor=np.ones(3), frustum=self.frustum, camera=self.camera, vc=self.light,
                                  overdraw=False)
        self.rn.overdraw = True
        self.rn.nsamples = 8
        self.rn.msaa = True  #Without anti-aliasing optimization often does not work.
        self.rn.initGL()
        self.rn.debug = False

        self.model_type = 'smpl'
        self.model_gender = 'm'
        self.model = None
        self._init_model()
        self.model.pose[0] = np.pi

        self.camera_widget = Ui_CameraWidget(self.camera, self.frustum, self.draw)
        self.btn_camera.clicked.connect(lambda: self._show_camera_widget())

        for key, shape in self._shapes():
            shape.valueChanged[int].connect(lambda val, k=key: self._update_shape(k, val))

        for key, exp in self._expressions():
            exp.valueChanged[int].connect(lambda val, k=key: self._update_exp(k, val))

        for key, pose in self._poses():
            pose.valueChanged[int].connect(lambda val, k=key: self._update_pose(k, val))

        self.pos_0.valueChanged[float].connect(lambda val: self._update_position(0, val))
        self.pos_1.valueChanged[float].connect(lambda val: self._update_position(1, val))
        self.pos_2.valueChanged[float].connect(lambda val: self._update_position(2, val))

        self.radio_f.pressed.connect(lambda: self._init_model('f'))
        self.radio_m.pressed.connect(lambda: self._init_model('m'))

        self.model_choose.currentIndexChanged[int].connect(lambda val: self._update_model(val))

        self.reset_pose.clicked.connect(self._reset_pose)
        self.reset_shape.clicked.connect(self._reset_shape)
        self.reset_expression.clicked.connect(self._reset_expression)
        self.reset_postion.clicked.connect(self._reset_position)

        self.canvas.wheelEvent = self._zoom
        self.canvas.mousePressEvent = self._mouse_begin
        self.canvas.mouseMoveEvent = self._move
        self.canvas.mouseReleaseEvent = self._mouse_end
        first_frame_name = self.first_name()
        self.back_name.setPlainText(first_frame_name)
        self.action_save.triggered.connect(self._save_config_dialog_2)
        self.action_open.triggered.connect(self._open_config_dialog_2)
        self.frame_id.valueChanged.connect(self._save_config_dialog)
        self.frame_id.valueChanged.connect(self._open_config_dialog)

        self.action_save_screenshot.triggered.connect(self._save_screenshot_dialog)
        self.action_save_mesh.triggered.connect(self._save_mesh_dialog)

        self.view_joints.triggered.connect(self.draw)
        self.view_joint_ids.triggered.connect(self.draw)
        self.view_bones.triggered.connect(self.draw)
        self.view_back.triggered.connect(self.draw)
        self.pic_w.valueChanged.connect(self.draw)
        self.frame_id.valueChanged.connect(self.draw)

        self._update_canvas = True

    def showEvent(self, event):
        self._init_camera()
        super(self.__class__, self).showEvent(event)

    def resizeEvent(self, event):
        self._init_camera()
        super(self.__class__, self).resizeEvent(event)

    def closeEvent(self, event):
        self.camera_widget.close()
        super(self.__class__, self).closeEvent(event)

    def draw(self):
        # print(self._update_canvas)
        if self._update_canvas:
            w = self.rn.r.shape[0]
            h = self.rn.r.shape[1]
            c = self.rn.r.shape[2]

            img = np.array(self.rn.r)

            if self.view_back.isChecked():
                fil_pic = self.back_name.toPlainText()
                orig_back = cv2.imread("../spin/pic/%s.jpg" % fil_pic, cv2.IMREAD_UNCHANGED)
                back = letterbox_image(orig_back, (h, w))
                back.resize(w, h, c)
                back = back / 255
                pic_rate = self.pic_w.value()/10
                back_rate = 1.0 - pic_rate
                img = np.array(cv2.addWeighted(self.rn.r, pic_rate, back, back_rate, 5))

            # else:
            #     img = np.array(self.rn.r)

            if self.view_joints.isChecked() or self.view_joint_ids.isChecked() or self.view_bones.isChecked():
                img = self._draw_annotations(img)

            self.canvas.setScaledContents(False)
            self.canvas.setPixmap(self._to_pixmap(img))

    def _draw_annotations(self, img):
        self.joints2d.set(t=self.camera.t, rt=self.camera.rt, f=self.camera.f, c=self.camera.c, k=self.camera.k)

        height = self.canvas.height()
        if self.view_bones.isChecked():
            kintree = self.model.kintree_table[:, 1:]
            for k in range(kintree.shape[1]):
                cv2.line(img, (int(self.joints2d.r[kintree[0, k], 0]), int(height - self.joints2d.r[kintree[0, k], 1])),
                         (int(self.joints2d.r[kintree[1, k], 0]), int(height - self.joints2d.r[kintree[1, k], 1])),
                         (0.98, 0.98, 0.98), 3)

        if self.view_joints.isChecked():
            for j in self.joints2d.r:
                jj = height-j[1] # for opengl: flipx
                cv2.circle(img, (int(j[0]), int(jj)), 5, (0.38, 0.68, 0.15), -1)

        if self.view_joint_ids.isChecked():
            for k, j in enumerate(self.joints2d.r):
                jj = height-j[1]
                cv2.putText(img, str(k), (int(j[0]), int(jj)), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0.3, 0.23, 0.9), 2)

        return img

    def _init_model(self, gender=None):
        pose = None
        betas = None
        trans = None

        if self.model is not None:
            pose = self.model.pose.r
            betas = self.model.betas.r
            trans = self.model.trans.r

        if gender == None: 
            gender = self.model_gender
        else:
            self.model_gender = gender
        
        self.model = load_model(model_type=self.model_type, gender=self.model_gender)
        # print(self.model.r)

        if pose is not None:
            self.model.pose[:] = pose
            self.model.betas[:] = betas
            self.model.trans[:] = trans

        self.light.set(v=self.model, f=self.model.f, num_verts=len(self.model))
        self.rn.set(v=self.model, f=self.model.f)

        self.camera.set(v=self.model)
        self.joints2d.set(v=self.model.J_transformed)

        self.draw()

    def _init_camera(self, update_camera=False):
        w = self.canvas.width()
        h = self.canvas.height()

        if update_camera or w != self.frustum['width'] and h != self.frustum['height']:
            self.camera.set(rt=np.array([self.camera_widget.rot_0.value(), self.camera_widget.rot_1.value(),
                                         self.camera_widget.rot_2.value()]),
                            t=np.array([self.camera_widget.pos_0.value(), self.camera_widget.pos_1.value(),
                                        self.camera_widget.pos_2.value()]),
                            f=np.array([w, w]) * self.camera_widget.focal_len.value(),
                            c=np.array([w, h]) / 2.,
                            k=np.array([self.camera_widget.dist_0.value(), self.camera_widget.dist_1.value(),
                                        self.camera_widget.dist_2.value(), self.camera_widget.dist_3.value(),
                                        self.camera_widget.dist_4.value()]))

            self.frustum['width'] = w
            self.frustum['height'] = h

            self.light.set(light_pos=Rodrigues(self.camera.rt).T.dot(self.camera.t) * -10.)
            self.rn.set(frustum=self.frustum, camera=self.camera)
            flipXRotation = np.array([[1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0., 0.0],
                [0.0, 0., -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]])
            self.rn.camera.openglMat = flipXRotation #this is from setupcamera in utils
            self.rn.glMode = 'glfw'
            self.rn.sharedWin = None
            self.rn.overdraw = True
            self.rn.nsamples = 8
            self.rn.msaa = True  #Without anti-aliasing optimization often does not work.
            self.rn.initGL()           
            self.rn.debug = False

            # print(self.rn.r)

            self.draw()

    def _save_config_dialog(self):
        # filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save config', "../spin/ini_file/%s.ini" % self.back_name.toPlainText(), 'Config File (*.ini)')
        filename = "../spin/ini_file/%s.ini" % self.back_name.toPlainText()
        if filename:
            with open(str(filename), 'w') as fp:
                config = configparser.ConfigParser()
                config.add_section('Model')
                config.set('Model', 'gender', self.model_gender)
                config.set('Model', 'shape', ','.join(str(s) for s in self.model.betas.r))
                config.set('Model', 'pose', ','.join(str(p) for p in self.model.pose.r))
                config.set('Model', 'translation', ','.join(str(p) for p in self.model.trans.r))

                config.add_section('Camera')
                config.set('Camera', 'translation', ','.join(str(t) for t in self.camera.t.r))
                config.set('Camera', 'rotation', ','.join(str(r) for r in self.camera.rt.r))
                config.set('Camera', 'focal_length', str(self.camera_widget.focal_len.value()))
                config.set('Camera', 'center', '{},{}'.format(self.camera_widget.center_0.value(),
                                                              self.camera_widget.center_1.value()))
                config.set('Camera', 'distortion', ','.join(str(r) for r in self.camera.k.r))

                config.write(fp)

    def _open_config_dialog(self):
        # filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load config', "../spin/ini_file/", 'Config File (*.ini)')
        ini_folder_name = '../spin/ini_file'
        ini_files = os.listdir(ini_folder_name)  # 返回指定路径下的文件和文件夹列表
        ini_id = self.frame_id.value()
        filename_id = ini_files[ini_id]
        if filename_id:
            back_pic = filename_id.split(".")[0]
            self.back_name.setPlainText(back_pic)
            filename = '../spin/ini_file/%s' % filename_id
            config = configparser.ConfigParser()
            config.read(str(filename))

            self._update_canvas = False
            self._init_model(config.get('Model', 'gender'))

            shapes = np.fromstring(config.get('Model', 'shape'), dtype=np.float64, sep=',')
            poses = np.fromstring(config.get('Model', 'pose'), dtype=np.float64, sep=',')
            position = np.fromstring(config.get('Model', 'translation'), dtype=np.float64, sep=',')

            for key, shape in self._shapes():
                # val = shapes[key] / 5.0 * 50.0 + 50.0
                val = shapes[key] / 5.0 * 50000.0 + 50000.0
                shape.setValue(val)
            for key, pose in self._poses():
                if key == 0:
                    # val = (poses[key] - np.pi) / np.pi * 50.0 + 50.0
                    val = (poses[key]) / np.pi * 50000.0 + 50000.0  # 姿态朝向

                else:
                    # val = poses[key] / np.pi * 50.0 + 50.0
                    val = poses[key] / np.pi * 50000.0 + 50000.0
                pose.setValue(val)

            self.pos_0.setValue(position[0])
            self.pos_1.setValue(position[1])
            self.pos_2.setValue(position[2])

            cam_pos = np.fromstring(config.get('Camera', 'translation'), dtype=np.float64, sep=',')
            cam_rot = np.fromstring(config.get('Camera', 'rotation'), dtype=np.float64, sep=',')
            cam_dist = np.fromstring(config.get('Camera', 'distortion'), dtype=np.float64, sep=',')
            cam_c = np.fromstring(config.get('Camera', 'center'), dtype=np.float64, sep=',')
            cam_f = config.getfloat('Camera', 'focal_length')
            self.camera_widget.set_values(cam_pos, cam_rot, cam_f, cam_c, cam_dist)

            self._update_canvas = True
            self.draw()

    def _save_screenshot_dialog(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save screenshot', "../spin/out_pic/%s" % self.back_name.toPlainText(), 'Images (*.png *.jpg *.ppm)')
        if filename:
            img = np.array(self.rn.r)
            if self.view_joints.isChecked() or self.view_joint_ids.isChecked() or self.view_bones.isChecked():
                img = self._draw_annotations(img)
            cv2.imwrite(str(filename), np.uint8(img * 255))

    def _save_mesh_dialog(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save mesh', "../spin/out_mesh/%s" % self.back_name.toPlainText(), 'Mesh (*.obj)')
        if filename:
            with open(filename, 'w') as fp:
                for v in self.model.r:
                    fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

                for f in self.model.f + 1:
                    fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def _zoom(self, event):
        delta = -event.angleDelta().y() / 1200.0
        self.camera_widget.pos_2.setValue(self.camera_widget.pos_2.value() + delta)

    def _mouse_begin(self, event):
        if event.button() == 4: # middle
            self._moving = True
        elif event.button() == 1: # left
            self._rotating = True
        self._mouse_begin_pos = event.pos()

    def _mouse_end(self, event):
        self._moving = False
        self._rotating = False

    def _move(self, event):
        if self._moving:
            delta = event.pos() - self._mouse_begin_pos
            self.camera_widget.pos_0.setValue(self.camera_widget.pos_0.value() + delta.x() / 500.)
            self.camera_widget.pos_1.setValue(self.camera_widget.pos_1.value() + delta.y() / 500.)
            self._mouse_begin_pos = event.pos()
        elif self._rotating:
            delta = event.pos() - self._mouse_begin_pos
            self.camera_widget.rot_0.setValue(self.camera_widget.rot_0.value() + delta.y() / 300.)
            self.camera_widget.rot_1.setValue(self.camera_widget.rot_1.value() - delta.x() / 300.)
            self._mouse_begin_pos = event.pos()

    def _show_camera_widget(self):
        self.camera_widget.show()
        self.camera_widget.raise_()

    def _update_model(self, id):
        print(id)
        self.model = None
        self.model_type = model_type_list[id]

        self._update_canvas = False

        self.light = LambertianPointLight(vc=np.array([0.98, 0.98, 0.98]), light_color=np.array([1., 1., 1.]))

        self.rn.set(glMode='glfw',bgcolor=np.ones(3), frustum=self.frustum, camera=self.camera, vc=self.light,
                                  overdraw=False)

        self._init_model()
        self.model.pose[0] = np.pi
        self._init_camera(update_camera=True)

        self._reset_shape()
        self._reset_expression()
        self._reset_pose()
        self._reset_position()

        self._update_canvas = True
        self.draw()

    def _update_shape(self, id, val):
        # val = (val - 50) / 50.0 * 5.0
        val = (val - 50000) / 50000.0 * 5.0
        self.model.betas[id] = val
        print("当前shape_%d值为" % id, val)
        self.draw()

    def _update_exp(self, id, val):
        # val = (val - 50) / 50.0 * 5.0
        val = (val - 50000) / 50000.0 * 5.0
        if self.model_type=='smplx':
            self.model.betas[10 + id] = val
            self.draw()
        elif self.model_type=='flame':
            self.model.betas[10 + id] = val
            self.draw()

    def _reset_shape(self):
        self._update_canvas = False
        for key, shape in self._shapes():
            # shape.setValue(50)
            shape.setValue(50000)
        self._update_canvas = True
        self.draw()

    def _reset_expression(self):
        if self.model_type != 'smplx' and self.model_type != 'flame':
            return
        self._update_canvas = False
        for key, exp in self._expressions():
            # exp.setValue(50)
            exp.setValue(50000)
        self._update_canvas = True
        self.draw()

    def _update_pose(self, id, val):
        # val = (val - 50) / 50.0 * np.pi
        val = (val - 50000) / 50000.0 * np.pi

        # if id == 0:              # 姿态朝向
        #     val += np.pi

        if self.model_type == 'flame' and id>=5*3:
            return

        self.model.pose[id] = val
        print("当前pose_%d值为" % id, val)
        self.draw()

    def _reset_pose(self):
        self._update_canvas = False
        for key, pose in self._poses():
            # pose.setValue(50)
            pose.setValue(50000)
        self._update_canvas = True
        self.draw()

    def _update_position(self, id, val):
        self.model.trans[id] = val
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

        qimg = QtGui.QImage(im, im.shape[1], im.shape[0], im.strides[0], QtGui.QImage.Format_RGB888)

        return QtGui.QPixmap(qimg)

    def first_name(self):
        pkl_folder_name = '../spin/pkl_file'
        pkl_files = os.listdir(pkl_folder_name)  # 返回指定路径下的文件和文件夹列表
        first_pkl = pkl_files[0]
        first_pkl_name = first_pkl.split(".")[0]
        return first_pkl_name

    def _save_config_dialog_2(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save config', "../spin/ini_file/%s.ini" % self.back_name.toPlainText(), 'Config File (*.ini)')
        if filename:
            with open(str(filename), 'w') as fp:
                config = configparser.ConfigParser()
                config.add_section('Model')
                config.set('Model', 'gender', self.model_gender)
                config.set('Model', 'shape', ','.join(str(s) for s in self.model.betas.r))
                config.set('Model', 'pose', ','.join(str(p) for p in self.model.pose.r))
                config.set('Model', 'translation', ','.join(str(p) for p in self.model.trans.r))

                config.add_section('Camera')
                config.set('Camera', 'translation', ','.join(str(t) for t in self.camera.t.r))
                config.set('Camera', 'rotation', ','.join(str(r) for r in self.camera.rt.r))
                config.set('Camera', 'focal_length', str(self.camera_widget.focal_len.value()))
                config.set('Camera', 'center', '{},{}'.format(self.camera_widget.center_0.value(),
                                                              self.camera_widget.center_1.value()))
                config.set('Camera', 'distortion', ','.join(str(r) for r in self.camera.k.r))

                config.write(fp)

    def _open_config_dialog_2(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load config', "../spin/ini_file/", 'Config File (*.ini)')
        if filename:
            fil = filename.split("/")[-1].split(".")[0]
            self.back_name.setPlainText(fil)
            # pkl_folder_name = '../spin/pkl_file'
            # pkl_files = os.listdir(pkl_folder_name)  # 返回指定路径下的文件和文件夹列表
            # self.frame_id.setvalve(2)
            config = configparser.ConfigParser()
            config.read(str(filename))

            self._update_canvas = False
            self._init_model(config.get('Model', 'gender'))

            shapes = np.fromstring(config.get('Model', 'shape'), dtype=np.float64, sep=',')
            poses = np.fromstring(config.get('Model', 'pose'), dtype=np.float64, sep=',')
            position = np.fromstring(config.get('Model', 'translation'), dtype=np.float64, sep=',')

            for key, shape in self._shapes():
                # val = shapes[key] / 5.0 * 50.0 + 50.0
                val = shapes[key] / 5.0 * 50000.0 + 50000.0
                shape.setValue(val)
            for key, pose in self._poses():
                if key == 0:
                    # val = (poses[key] - np.pi) / np.pi * 50.0 + 50.0
                    val = (poses[key]) / np.pi * 50000.0 + 50000.0  # 姿态朝向

                else:
                    # val = poses[key] / np.pi * 50.0 + 50.0
                    val = poses[key] / np.pi * 50000.0 + 50000.0
                pose.setValue(val)

            self.pos_0.setValue(position[0])
            self.pos_1.setValue(position[1])
            self.pos_2.setValue(position[2])

            cam_pos = np.fromstring(config.get('Camera', 'translation'), dtype=np.float64, sep=',')
            cam_rot = np.fromstring(config.get('Camera', 'rotation'), dtype=np.float64, sep=',')
            cam_dist = np.fromstring(config.get('Camera', 'distortion'), dtype=np.float64, sep=',')
            cam_c = np.fromstring(config.get('Camera', 'center'), dtype=np.float64, sep=',')
            cam_f = config.getfloat('Camera', 'focal_length')
            self.camera_widget.set_values(cam_pos, cam_rot, cam_f, cam_c, cam_dist)

            self._update_canvas = True
            self.draw()


# 缩放
def letterbox_image(img, input_dim):
    """
    resize image with unchanged aspect ratio using padding
    让原图的高宽乘以同一个数，这样就不会改变比例，而这个数就是min( 高缩放的比例，宽缩放的比例)，然后padding周围区域使得缩放到指定大小。
    缩小： (2000, 4000) -> (200,200), min(200/4000, 200/2000) = 1/20, 2000 * 1/20 = 100， 4000 * 1/20 = 200
    新的尺度(100, 200)，再padding
    放大： (50, 100) -> (200, 200), min(4, 2) = 2, 50 * 2 = 100, 100 * 2 = 200
    新的尺度(100, 200)，再padding
    :param img: 原始图片
    :param input_dim: (w, h). 缩放后的尺度
    :return:
    """
    orig_w, orig_h = img.shape[1], img.shape[0]
    input_w, input_h = input_dim  # 缩放(orig_w, orig_h) -> (input_w, input_h)

    # 1，根据缩放前后的尺度，获取有效的新的尺度(new_w, new_h)
    min_ratio = min(input_w / orig_w, input_h / orig_h)
    new_w = int(orig_w * min_ratio)  # （new_w, new_h）是高宽比不变的新的尺度
    new_h = int(orig_h * min_ratio)
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 定义画布canvas，然后把有效区域resized_image填充进去
    canvas = np.full((input_dim[1], input_dim[0], 3), 128, np.uint8)  # 定义画布，大小是（200,200）

    # 画布_h - 高宽不变_new_h
    start_h = (input_h - new_h) // 2  # 开始插入的高度位置, 也是上下需要padding的大小
    start_w = (input_w - new_w) // 2  # 开始插入的宽度位置
    canvas[start_h:start_h + new_h, start_w:start_w + new_w, :] = resized_image

    return canvas




