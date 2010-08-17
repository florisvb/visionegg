import numpy as np
import socket, sys, threading, Queue, time, os, math, warnings
import flydra.kalman.data_packets as data_packets
import flydra.common_variables
import flydra.geom as geom
import Pyro.core

import VisionEgg
VisionEgg.start_default_logging(); VisionEgg.watch_exceptions()

from VisionEgg.Core import Screen, Viewport, FrameTimer, ModelView, \
     swap_buffers, SimplePerspectiveProjection
from VisionEgg.MoreStimuli import Rectangle3D
from VisionEgg.Gratings import SinGrating3D
from VisionEgg.Textures import Texture, TextureStimulus3D, SpinningDrum
import pygame
from pygame.locals import QUIT,KEYDOWN,MOUSEBUTTONDOWN
import OpenGL.GL as gl

import VR_math
import FlatRect

pi = np.pi


class Arena:
    # want a function that takes 'trace edges' and generates an arena instance
    def __init__(self):
        
        self.geometry = 'cube'
        self.texture_filename = 'checkboard.jpg'
        self.texture = Texture(self.texture_filename)

        self.verts =    [ (0,0,0),#0
                          (1,0,0),#1
                          (1,1,0),#2
                          (0,1,0),#3
                          (0,0,1),#4
                          (1,0,1),#5
                          (1,1,1),#6
                          (0,1,1),#7
                          ]
        self.face_verts = {}
        self.face_verts['+x'] = [2,6,5,1]
        self.face_verts['+y'] = [3,7,6,2]
        self.face_verts['-x'] = [0,4,7,3]
        self.face_verts['-y'] = [1,5,4,0]
        self.face_verts['+z'] = [4,5,6,7]
        self.face_verts['-z'] = [0,1,2,3]
        
        self.faces_3d = {}
        for k,v in self.face_verts.items():
            self.faces_3d[k] = [self.verts[v[0]], 
                                self.verts[v[1]],
                                self.verts[v[2]],
                                self.verts[v[3]],
                               ]
                               
        if 0:
            for face_name, face_verts in self.faces.iteritems():
                face = TextureStimulus3D(texture = self.texture,
                                         shrink_texture_ok=True,
                                         internal_format=gl.GL_RGBA,
                                         lowerleft=verts[faces[face_name][0]],
                                         upperleft=verts[faces[face_name][1]],
                                         upperright=verts[faces[face_name][2]],
                                         lowerright=verts[faces[face_name][3]],
                                         mipmaps_enabled=False,
                                         depth_test=True,
                                         )
                #vr_objects.append(face)
            
            
    def write_xml(self):
        # write xml file for use in liveviewer, plots, etc.
        return 0


#SHOW='cam:floor'
#SHOW='cam:test'
#SHOW='cam:+y'
SHOW='overview'
#SHOW='projector (calibrated)'

FOREST=False
DRUM = True
DOTS = False
UNIT_CUBE = False

BGCOLOR = (0.5,0.5,0.5)

TRIAL_DURATION = 30.0 # seconds
DO_EXPERIMENT = True
EXPERIMENT_STATES = [
    'waiting',
    'doing virtual reality drum radius 0.1',
    ## 'waiting',
    ## 'doing static drum radius 0.1',
    ## 'waiting',
    ## 'doing virtual reality drum radius 0.3',
    ## 'waiting',
    ## 'doing static drum radius 0.3',
    ## 'waiting',
    ## 'doing gray only stimulus'
    ]

# EXPERIMENT_STATE is an index into above list
EXPERIMENT_STATE = 0
EXPERIMENT_STATE_START_TIME = None
STATIC_FLY_XYZ = None

def in_trigger_volume(fly_xyz):
    return True

def should_i_advance_to_next_state(fly_xyz):
    state_string = EXPERIMENT_STATES[EXPERIMENT_STATE]
    state_string_split = state_string.split()
    if state_string_split[0] == 'waiting':
        if in_trigger_volume(fly_xyz):
            return True
        else:
            return False
    elif state_string_split[0] == 'doing':
        start_doing_time = EXPERIMENT_STATE_START_TIME
        now = time.time()
        dur = now-start_doing_time
        if dur > TRIAL_DURATION:
            return True
        ## if fly_xyz == None:
        ##     return True
        return False

global sendsock_host
sendsock_host = ('brain1',30041)

def advance_state(fly_xyz,obj_id,framenumber,sendsock):
    global EXPERIMENT_STATE, EXPERIMENT_STATE_START_TIME, mainbrain
    global STATIC_FLY_XYZ
    global sendsock_host

    EXPERIMENT_STATE = (EXPERIMENT_STATE+1) % len(EXPERIMENT_STATES)
    EXPERIMENT_STATE_START_TIME = time.time()
    prefix = '(%s %s) '%(obj_id,framenumber)
    mainbrain.log_message('<wtstim>',
                          EXPERIMENT_STATE_START_TIME,
                          prefix+EXPERIMENT_STATES[EXPERIMENT_STATE])
    print EXPERIMENT_STATES[EXPERIMENT_STATE]
    if EXPERIMENT_STATES[EXPERIMENT_STATE] != 'waiting':
        sendsock.sendto('x',sendsock_host)
    if EXPERIMENT_STATES[EXPERIMENT_STATE].startswith('doing static'):
        STATIC_FLY_XYZ = fly_xyz

class Listener(object):
    def __init__(self,sockobj):
        self.s = sockobj
        self.q = Queue.Queue()
    def run(self):
        while 1:
            #print 'listening for packet on',self.s
            buf, addr = self.s.recvfrom(4096)
            #print 'got packet:',buf
            self.q.put( buf )
    def get_list_of_bufs(self):
        result = []
        while 1:
            try:
                result.append( self.q.get_nowait() )
            except Queue.Empty:
                break
        return result
    def get_most_recent_single_fly_data(self):
        superpacket_bufs = self.get_list_of_bufs()
        if len(superpacket_bufs)==0:
            # no new data
            return
        buf = superpacket_bufs[-1] # most recent superpacket
        packets = data_packets.decode_super_packet( buf )
        packet = packets[-1] # most recent packet
        tmp = data_packets.decode_data_packet(packet)
        #(corrected_framenumber, timestamp, obj_ids, state_vecs, meanPs) = tmp
        return tmp

class ReplayListener(object):
    def __init__(self,kalman_filename):
        import flydra.a2.core_analysis as core_analysis
        import flydra.analysis.result_utils as result_utils

        self.ca = core_analysis.get_global_CachingAnalyzer()
        (obj_ids, use_obj_ids, is_mat_file, data_file,
         extra) = self.ca.initial_file_load(kalman_filename)
        self.data_file = data_file
        self.up_dir = None

        if 1:
            dynamic_model = extra['dynamic_model_name']
            print 'detected file loaded with dynamic model "%s"'%dynamic_model
            if dynamic_model.startswith('EKF '):
                dynamic_model = dynamic_model[4:]
            print '  for smoothing, will use dynamic model "%s"'%dynamic_model
        self.dynamic_model = dynamic_model
        self.fps = result_utils.get_fps( data_file )
        self.use_kalman_smoothing = True
        self.up_dir = (0,0,1)

    def set_obj_frame(self,obj_id,framenumber):
        my_rows = self.ca.load_data(
            obj_id, self.data_file,
            use_kalman_smoothing=self.use_kalman_smoothing,
            dynamic_model_name = self.dynamic_model,
            frames_per_second=self.fps,
            up_dir=self.up_dir)
        cond = (my_rows['frame'] == framenumber) & (my_rows['obj_id']==obj_id)
        idx = np.nonzero(cond)[0]
        if len(idx)!=1:
            raise ValueError('no unique frame for obj_id and framenumber')
        row = my_rows[idx[0]]

        next_xyz = row['x'], row['y'], row['z']
        self.next_results = obj_id, next_xyz, framenumber

    def get_fly_xyz(self,prefer_obj_id=None):
        return self.next_results

class InterpolatingListener(object):
    def __init__(self,sockobj,dummy=False):
        if sockobj is not None:
            self.listener = Listener(sockobj)
        self.last_data_time = -np.inf
        self.current_state_vec = None
        self.dummy=dummy
    def run(self):
        self.listener.run()
    def get_fly_xyz(self,prefer_obj_id=None):
        """
        Returns
        =======
        obj_id
        fly_xyz
        framenumber
        """

        if self.dummy:
            t = time.time()
            timestamp = t
            theta = (2*pi*t) / 5.0
            x = eye_loc_default[0] + 0.2*np.cos( theta )
            y = eye_loc_default[1] + 0.2*np.sin( theta )
            z = eye_loc_default[2] + 0.2*np.sin(theta*.2)
            state_vecs = [ (x,y,z) ]
            corrected_framenumber = 0
            meanP = 0
            return 0, (x,y,z), 0

        new_data_all = self.listener.get_most_recent_single_fly_data()
        now = time.time()
        if new_data_all is not None:
            (corrected_framenumber, acquire_timestamp, reconstruct_timestamp,
             obj_ids, state_vecs, meanPs) = new_data_all
            #corrected_framenumber, timestamp, state_vecs, meanP = new_data_all
            self.last_data_time = now
            if prefer_obj_id is not None:
                try:
                    idx = obj_ids.index(prefer_obj_id)
                except ValueError:
                    idx = 0
            else:
                idx = 0
            self.current_framenumber = corrected_framenumber
            self.current_obj_id = obj_ids[idx]
            self.current_state_vec = np.array(state_vecs[idx]) #convert to numpy
        dt = now-self.last_data_time
        if dt > 0.1:
            # return None if no recent target (recent defined as 100 msec)
            return None

        if dt <= 0.0:
            # return pure X,Y,Z
            return (self.current_obj_id,
                    self.current_state_vec[:3],
                    self.current_framenumber)

        state_vec = self.current_state_vec
        dx = dt*state_vec[3:] # velocity*time
        newx = state_vec[:3]+dx
        return self.current_obj_id, newx, self.current_framenumber

eye_loc_default = (0.5, 0.5, 0.5)

class DummyMainbrain:
    def log_message(self,*args,**kwds):
        return

class StimulusLocationUpdater(object):
    def __init__(self,stim,
                 offset=None,
                 inc=0.01,
                 inc1_dir=None,
                 inc2_dir=None,
                 ):
        if offset is None:
            offset=(0,0,0)
        if inc1_dir is None:
            inc1_dir = (1,0,0)
        if inc2_dir is None:
            inc2_dir = (0,1,0)
        self.stim=stim
        self.offset=geom.ThreeTuple(offset)
        self.inc=inc
        self.inc1_dir = geom.ThreeTuple(inc1_dir)
        self.inc2_dir = geom.ThreeTuple(inc2_dir)
    def update(self,loc):
        newloc = geom.ThreeTuple(loc)+self.offset
        inc = self.inc
        if 0:
            x_inc = geom.ThreeTuple((inc,0,0))
            y_inc = geom.ThreeTuple((0,inc,0))
            xy_inc = geom.ThreeTuple((inc,inc,0))
        else:
            x_inc = self.inc1_dir * inc
            y_inc = self.inc2_dir * inc
            xy_inc = (self.inc1_dir + self.inc2_dir) * inc
        p = self.stim.parameters
        p.vertex1=newloc.vals
        p.vertex2=(newloc+x_inc).vals
        p.vertex3=(newloc+xy_inc).vals
        p.vertex4=(newloc+y_inc).vals

def get_wall_dict( show, screen, corners_3d, corners_2d,
                   vr_objects, approx_view_dir, name):
    proj_math = VR_math.VRScreenData(corners_3d,#approx_view_dir=approx_view_dir,
                                     name=name)
    cam_viewport = Viewport(screen=screen,
                            stimuli=vr_objects,
                            auto_pixel_projection=False,
                            )
    result = dict(screen_data=proj_math,
                  cam_viewport=cam_viewport)
    if show in ['overview','projector (calibrated)']:
        framebuffer_copy_texture = Texture(
            texels=screen.get_framebuffer_as_image(format=gl.GL_RGBA) )
        screen_rect = TextureStimulus3D(
            texture = framebuffer_copy_texture,
            shrink_texture_ok=True,
            internal_format=gl.GL_RGBA,
            mipmaps_enabled=False, # don't pretend to do mipmaps on every frame
            )

        result['framebuffer_texture_object']=(
            screen_rect.parameters.texture.get_texture_object())

        if show=='overview':
            screen_rect.set(
                lowerleft=corners_3d[0],
                upperleft=corners_3d[1],
                upperright=corners_3d[2],
                lowerright=corners_3d[3],
                )
        elif show=='projector (calibrated)':
            def make3d(tup):
                return (tup[0],tup[1],-1.0)

            # set these to the corners of the rectangle being projected:
            screen_rect.set(
                lowerleft=make3d(corners_2d[0]),
                upperleft=make3d(corners_2d[1]),
                upperright=make3d(corners_2d[2]),
                lowerright=make3d(corners_2d[3]),
                )
        result['display_stimuli']=[screen_rect]

    return result

def save_wall_models( vr_walls, save_osg_info ):
    import fsee.scenegen.primlib as primlib
    import fsee.scenegen.osgwriter as osgwriter

    geode = osgwriter.Geode(states=['GL_LIGHTING OFF'])
    for wallname,wall in vr_walls.iteritems():
        data = wall['screen_data']
        osgprim = primlib.Prim()
        osgprim.texture_fname = wallname+'.png'
        count = 0
        quads = []
        normal = (0,0,1)
        osgprim.verts.append( data.LL_3d[:3,0] )
        osgprim.verts.append( data.UL_3d[:3,0] )
        osgprim.verts.append( data.UR_3d[:3,0] )
        osgprim.verts.append( data.LR_3d[:3,0] )
        osgprim.normals.append( normal )
        osgprim.normals.append( normal )
        osgprim.normals.append( normal )
        osgprim.normals.append( normal )
        osgprim.tex_coords.append( [0,0] )
        osgprim.tex_coords.append( [0,1] )
        osgprim.tex_coords.append( [1,1] )
        osgprim.tex_coords.append( [1,0] )
        quads.append( [count, count+1, count+2, count+3] )
        count += 4
        osgprim.prim_sets = [primlib.Quads( quads )]
        geode.append(osgprim.get_as_osg_geometry())
    m = osgwriter.MatrixTransform(np.eye(4))
    m.append(geode)

    g = osgwriter.Group()
    g.append(m)

    full_path = os.path.join(save_osg_info['dirname'],'model.osg')
    fd = open(full_path,'wb')
    g.save(fd)
    fd.close()

def main(connect_to_mainbrain=True,
         save_osg_info=None, # dict with info
         live_demo = False, # if True, never leave VR mode
         ):
    global EXPERIMENT_STATE
    global EXPERIMENT_STATE_START_TIME
    global mainbrain

    if save_osg_info is not None:
        save_osg = True
    else:
        save_osg = False

    sendsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if connect_to_mainbrain:
        assert not save_osg, 'cannot connect to mainbrain and save to .osg file'

        # make connection to flydra mainbrain
        my_host = '' # get fully qualified hostname
        my_port = 8322 # arbitrary number

        # create UDP socket object, grab the port
        sockobj = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print 'binding',( my_host, my_port)
        sockobj.bind(( my_host, my_port))

        # connect to mainbrain
        mainbrain_hostname = 'brain1'
        mainbrain_port = flydra.common_variables.mainbrain_port
        mainbrain_name = 'main_brain'
        remote_URI = "PYROLOC://%s:%d/%s" % (mainbrain_hostname,
                                             mainbrain_port,
                                             mainbrain_name)
        Pyro.core.initClient(banner=0)
        mainbrain = Pyro.core.getProxyForURI(remote_URI)
        mainbrain._setOneway(['log_message'])
        my_host_fqdn = socket.getfqdn(my_host)
        mainbrain.register_downstream_kalman_host(my_host_fqdn,my_port)

        listener = InterpolatingListener(sockobj)
        #listener = Listener(sockobj)
        listen_thread = threading.Thread(target=listener.run)
        listen_thread.setDaemon(True)
        listen_thread.start()
    else:
        if save_osg:
            listener = ReplayListener(save_osg_info['kalman_h5'])
        else:
            listener = InterpolatingListener(None,dummy=True)
        mainbrain = DummyMainbrain()

    screen=Screen(size=(1024,768),
                  bgcolor=(0,0,0),
                  alpha_bits=8,
                  #fullscreen=True,
                  )

    if 1:
        # this loads faster, for debugging only
        mipmaps_enabled=False
    else:
        mipmaps_enabled=True

    DEBUG='none' # no debugging
    #DEBUG='flicker' # alternate views on successive frames

    vr_objects = []

    on_eye_loc_update_funcs = []

    if 0:
        # floor stimulus

        # Get a texture
        filename = 'stones_and_cement.jpg'
        texture = Texture(filename)

        # floor is iso-z rectangle
        x0 = -1
        x1 = 1
        y0 = -.25
        y1 =  .25
        z0 = -.1
        z1 = -.1
        vr_object = FlatRect.FlatRect(texture=texture,
                                      shrink_texture_ok=1,
                                      lowerleft=(x0,y0,z0),
                                      upperleft=(x0,y1,z1),
                                      upperright=(x1,y1,z1),
                                      lowerright=(x1,y0,z0),
                                      tex_phase = 0.2,
                                      mipmaps_enabled=mipmaps_enabled,
                                      depth_test=True,
                                      )
        vr_objects.append( vr_object )

    if FOREST:
        # +y wall stimulus

        if 0:
            # wall iso-y rectangle
            x0 = -1
            x1 = 1
            y0 =  .25
            y1 =  .25
            z0 = -.1
            z1 = .5
            # Get a texture
            filename = 'stones_and_cement.jpg'
            texture = Texture(filename)

            vr_object = FlatRect.FlatRect(texture=texture,
                                          shrink_texture_ok=1,
                                          lowerleft=(x0,y0,z0),
                                          upperleft=(x0,y1,z1),
                                          upperright=(x1,y1,z1),
                                          lowerright=(x1,y0,z0),
                                          tex_phase = 0.2,
                                          mipmaps_enabled=mipmaps_enabled,
                                          depth_test=True,
                                          )
            vr_objects.append(vr_object)
        else:
            # forest of trees in +y direction
            filename = 'tree.png'
            tree_texture = Texture(filename)
            for x0 in np.arange(-1,1,.3):
                for y0 in np.arange(.1, 1.0, .2):
                #for y0 in [0.0]:
                    x1 = x0+0.1

                    y1 = y0
                    z0 = -0.5
                    z1 = 1.0
                    tree = TextureStimulus3D(texture = tree_texture,
                                             shrink_texture_ok=True,
                                             internal_format=gl.GL_RGBA,
                                             lowerleft=(x0,y0,z0),
                                             upperleft=(x0,y1,z1),
                                             upperright=(x1,y1,z1),
                                             lowerright=(x1,y0,z0),
                                             mipmaps_enabled=False,
                                             depth_test=True,
                                             )
                    vr_objects.append(tree)
    if FOREST:
        if 1:
            # forest of trees in -y direction
            filename = 'tree.png'
            tree_texture = Texture(filename)
            for x0 in np.arange(-1,1,.3):
                for y0 in np.arange(-.1, -1.0, -.2):
                #for y0 in [0.0]:
                    x1 = x0+0.1

                    y1 = y0
                    z0 = -0.5
                    z1 = 1.0
                    tree = TextureStimulus3D(texture = tree_texture,
                                             shrink_texture_ok=True,
                                             internal_format=gl.GL_RGBA,
                                             lowerleft=(x0,y0,z0),
                                             upperleft=(x0,y1,z1),
                                             upperright=(x1,y1,z1),
                                             lowerright=(x1,y0,z0),
                                             mipmaps_enabled=False,
                                             depth_test=True,
                                             )
                    vr_objects.append(tree)

    if UNIT_CUBE:
        arena = Arena()
        for face_name, face_verts in arena.face_verts.iteritems():
            face = TextureStimulus3D(texture = arena.texture,
                                     shrink_texture_ok=True,
                                     internal_format=gl.GL_RGBA,
                                     lowerleft=arena.verts[arena.face_verts[face_name][0]],
                                     upperleft=arena.verts[arena.face_verts[face_name][1]],
                                     upperright=arena.verts[arena.face_verts[face_name][2]],
                                     lowerright=arena.verts[arena.face_verts[face_name][3]],
                                     mipmaps_enabled=False,
                                     depth_test=True,
                                     )
            vr_objects.append(face)

    if DRUM:
        filename = os.path.join( os.path.split( __file__ )[0], 'panorama-checkerboard.png')
        texture = Texture(filename)
        # cylinder
        drum = SpinningDrum( position=(0.5,0.5,0.5), # center of WT
                             orientation=0.0,
                             texture=texture,
                             drum_center_elevation=90.0,
                             radius = (0.4),
                             height=0.3,
                             internal_format=gl.GL_RGBA,
                             )
        vr_objects.append(drum)

    if DOTS:
        # due to a bug in Vision Egg, these only show up if drawn last.
        if 1:
            # fly tracker negZ stimulus
            negZ_stim = Rectangle3D(color=(1,1,1,1),
                                     )
            updater = StimulusLocationUpdater(negZ_stim,offset=(0,0,-1),inc=0.1)
            on_eye_loc_update_funcs.append( updater.update )
            vr_objects.append( negZ_stim )

        if 1:
            # fly tracker plusY stimulus
            plusY_stim = Rectangle3D(color=(1,1,1,1),
                                     )
            updater = StimulusLocationUpdater(plusY_stim,
                                              offset=(0,1,0),
                                              inc1_dir=(1,0,0),
                                              inc2_dir=(0,0,1),
                                              inc=0.1)
            on_eye_loc_update_funcs.append( updater.update )
            vr_objects.append( plusY_stim )

        if 1:
            # fly tracker negY stimulus
            negY_stim = Rectangle3D(color=(1,1,1,1),
                                     )
            updater = StimulusLocationUpdater(negY_stim,
                                              offset=(0,-1,0),
                                              inc1_dir=(1,0,0),
                                              inc2_dir=(0,0,1),
                                              inc=0.1)
            on_eye_loc_update_funcs.append( updater.update )
            vr_objects.append( negY_stim )


    vr_walls = {}
    arena = Arena()
    if 0:
        # test
        ## corners_3d = [
        ##     ( 0.5,  0.15, 0.01),
        ##     (-0.5,  0.15, 0.01),
        ##     (-0.5, -0.15, 0.01),
        ##     ( 0.5, -0.15, 0.01),
        ##               ]
        testz = 0.
        ## corners_3d = [(-0.5, -0.15, testz),
        ##               ( 0.5, -0.15, testz),
        ##               ( 0.5,  0.15, testz),
        ##               (-0.5,  0.15, testz)]
        corners_3d = [(-0.5, 0.15, testz),
                      ( 0.5, 0.15, testz),
                      ( 0.5, -0.15, testz),
                      (-0.5, -0.15, testz)]

        corners_2d = [ (0,210),
                       (799,210),
                       (799,401),
                       (0,399)]
        name = 'test'
        approx_view_dir = (0,0,-1) # down
        vr_wall_dict = get_wall_dict( SHOW, screen,
                                      corners_3d, corners_2d, vr_objects,
                                      approx_view_dir, name)
        vr_walls[name] = vr_wall_dict
        del testz
    if 0:
        # test2
        testy = 0.15
        corners_3d = [(-0.5, testy, 0),
                      ( -0.5, testy, 0.3),
                      ( 0.5, testy, 0.3),
                      (0.5, testy, 0)]
        del testy

        corners_2d = [ (0,0),
                       (799,0),
                       (799,200),
                       (0,200)]
        name = 'test2'
        approx_view_dir = (0,0,-1) # down
        vr_wall_dict = get_wall_dict( SHOW, screen,
                                      corners_3d, corners_2d, vr_objects,
                                      approx_view_dir, name)
        vr_walls[name] = vr_wall_dict
    if 0:
        # floor
        # Measured Mar 24, 2009. Used calibration cal20081120.xml.
        corners_3d = arena.faces_3d['-z']
        corners_2d = [
            (1,401),
            (800,399),
            (800,209),
            (1,208),
            ]
        name = 'floor'
        approx_view_dir = None
        vr_wall_dict = get_wall_dict( SHOW, screen,
                                      corners_3d, corners_2d, vr_objects,
                                      approx_view_dir, name)
        vr_walls[name] = vr_wall_dict
    if 1:
        # order: LL, UL, UR, LR
        # +y wall
        corners_3d = arena.faces_3d['+y']

        corners_2d = [
            # LL
            (513, 1),
            # UL
            (513,768),
            #UR
            (1024,550),
            # LR
            (1024,50),
            ]

        name = '+y'
        approx_view_dir = None
        vr_wall_dict = get_wall_dict( SHOW, screen,
                                      corners_3d, corners_2d, vr_objects,
                                      approx_view_dir,name)
        vr_walls[name] = vr_wall_dict
    if 1:
        # order: LL, UL, UR, LR
        # -y wall
        corners_3d = arena.faces_3d['+x']

        corners_2d = [
            # LL
            (1,50),
            # UL
            (1,620),
            # UR
            (512,768),
            # LR
            (512,1),
            ]

        name = '-y'
        approx_view_dir = None
        vr_wall_dict = get_wall_dict( SHOW, screen,
                                      corners_3d, corners_2d, vr_objects,
                                      approx_view_dir,name)
        vr_walls[name] = vr_wall_dict

    if SHOW in ['overview','projector (calibrated)']:
        screen_stimuli = []
        for wall in vr_walls.itervalues():
            screen_stimuli.extend( wall['display_stimuli'] )

        if SHOW=='overview':
            # draw dot where VR camera is for overview
            VR_eye_stim = Rectangle3D(color=(.2,.2,.2,1), # gray
                                      depth_test=True,#requires VE 1.1.1.1
                                      )
            fly_stim_updater=StimulusLocationUpdater(VR_eye_stim)
            on_eye_loc_update_funcs.append( fly_stim_updater.update )

            VR_stimuli = []
            for wall in vr_walls.itervalues():
                VR_stimuli.extend( wall['cam_viewport'].parameters.stimuli )
            display_viewport = Viewport(
                screen=screen,
                projection=SimplePerspectiveProjection(fov_x=90.0),
                stimuli=VR_stimuli+screen_stimuli+[VR_eye_stim,
                                                   ],
                )

        elif SHOW=='projector (calibrated)':
            display_viewport = Viewport(
                screen=screen,
                stimuli=screen_stimuli,
                )
    else:
        # parse e.g. SHOW='cam:floor'
        camname = SHOW[4:]
        display_viewport = vr_walls[camname]['cam_viewport']

    last_log_message_time = -np.inf

    # OpenGL textures must be power of 2
    def next_power_of_2(f):
        return math.pow(2.0,math.ceil(math.log(f)/math.log(2.0)))
    fb_width_pow2  = int(next_power_of_2(screen.size[0]))
    fb_height_pow2  = int(next_power_of_2(screen.size[1]))

    if save_osg:
        listener.set_obj_frame(save_osg_info['obj'],save_osg_info['frame'])

    if 1:
        # initialize
        if live_demo:
            EXPERIMENT_STATE = 0
        else:
            EXPERIMENT_STATE = -1
        tmp = listener.get_fly_xyz()
        if tmp is not None:
            obj_id,fly_xyz,framenumber = tmp
        else:
            fly_xyz = None
            obj_id = None
            framenumber = None
        del tmp
        if not save_osg:
            advance_state(fly_xyz,obj_id,framenumber,sendsock)
        else:
            warnings.warn('save_osg mode -- forcing experiment state')
            EXPERIMENT_STATE = 1

    frame_timer = FrameTimer()
    quit_now = False
    while not quit_now:
        # infinite loop to draw stimuli
        if save_osg:
            quit_now = True # quit on next round

        # test for keypress or mouseclick to quit
        for event in pygame.event.get():
            if event.type in (QUIT,MOUSEBUTTONDOWN):
            #if event.type in (QUIT,KEYDOWN,MOUSEBUTTONDOWN):
                quit_now = True

        ## now = time.time()
        ## if now-last_log_message_time > 5.0: # log a message every 5 seconds
        ##     mainbrain.log_message('<wtstim>',
        ##                           time.time(),
        ##                           'This is my message.' )
        ##     last_log_message_time = now

        near = 0.001
        far = 10.0

        tmp = listener.get_fly_xyz(prefer_obj_id=obj_id)
        if tmp is not None:
            obj_id,fly_xyz,framenumber = tmp
        else:
            fly_xyz = None
            obj_id = None
            framenumber = None
        del tmp

        if not save_osg:
            if should_i_advance_to_next_state(fly_xyz):
                if not live_demo:
                    advance_state(fly_xyz,obj_id,framenumber,sendsock)

        state_string = EXPERIMENT_STATES[EXPERIMENT_STATE]
        state_string_split = state_string.split()

        draw_stimuli = True
        if state_string_split[0]=='waiting':
            draw_stimuli=False
        elif state_string=='doing gray only stimulus':
            draw_stimuli=False

        if fly_xyz is not None:
            if state_string.startswith('doing static'):
                fly_xyz = STATIC_FLY_XYZ

            for wall in vr_walls.itervalues():
                wall['screen_data'].update_VE_viewport( wall['cam_viewport'],
                                                        fly_xyz, near, far,
                                                        avoid_clipping=True)

            for func in on_eye_loc_update_funcs: # only used to draw dots and in overview mode
                func(fly_xyz)
        else:
            # no recent data
            draw_stimuli = False

        # render fly-eye views and copy to texture objects if necessary
        #screen.set(bgcolor=(.4,.4,0)) # ??
        for wallname,wall in vr_walls.iteritems():
            if wallname=='test':
                screen.set(bgcolor=(1,0,0)) # red
            else:
                screen.set(bgcolor=BGCOLOR)
            screen.clear() # clear screen

            if draw_stimuli:
                if DRUM and state_string.endswith('drum radius 0.1'):
                    drum.set(radius = 0.1)
                elif DRUM and state_string.endswith('drum radius 0.3'):
                    drum.set(radius = 0.3)
                # render fly-eye view
                wall['cam_viewport'].draw()

            if SHOW in ['overview','projector (calibrated)']:
                framebuffer_texture_object = wall['framebuffer_texture_object']

                # copy screen back-buffer to texture
                framebuffer_texture_object.put_new_framebuffer(
                    size=(fb_width_pow2,fb_height_pow2),
                    internal_format=gl.GL_RGB,
                    buffer='back',
                    )
            if save_osg:
                # save screen back buffer to image file
                pil_image = screen.get_framebuffer_as_image(
                    format=gl.GL_RGBA)
                if not os.path.exists(save_osg_info['dirname']):
                    os.mkdir(save_osg_info['dirname'])
                wall_fname = wallname+'.png'
                wall_full_path = os.path.join( save_osg_info['dirname'],
                                               wall_fname )
                print 'saving %s'%wall_fname
                pil_image.save(wall_full_path)

            if DEBUG=='flicker':
                swap_buffers() # swap buffers
                #time.sleep(3.0)

        if save_osg:
            save_wall_models( vr_walls, save_osg_info)

        if SHOW=='overview':
            now = time.time()
            overview_movement_tf = 0.1
            theta = (2*pi*now * overview_movement_tf)
            overview_eye_loc = (-.5 + 0.1*np.cos( theta ), # x
                                -1.5 + 0.1*np.sin( theta ), # y
                                2.0) # z

            camera_matrix = ModelView()
            camera_matrix.look_at( overview_eye_loc, # eye
                                   eye_loc_default, # look at fly center
                                   #screen_data.t_3d[:3,0], # look at upper left corner
                                   (0,0,1), # up
                                   )

            display_viewport.set(camera_matrix=camera_matrix)

        # clear screen again
        if SHOW=='overview':
            screen.set(bgcolor=(0.0,0.0,0.8)) # blue
        else:
            screen.set(bgcolor=(0.0,0.0,0.0)) #black
        screen.clear() # clear screen
        display_viewport.draw() # draw the viewport and hence the stimuli

        swap_buffers() # swap buffers
        frame_timer.tick() # notify the frame time logger that we just drew a frame
    frame_timer.log_histogram() # print frame interval histogram

def main_wrap():
    from optparse import OptionParser
    usage = '%prog [options]'

    parser = OptionParser(usage)

    parser.add_option("--standalone", action='store_true',
                      help="do not attempt to connect to mainbrain",
                      default=False)

    parser.add_option("--live-demo", action='store_true',
                      help="normal operation but never leave VR mode",
                      default=False)

    parser.add_option("--save-osg-dirname",type='string')
    parser.add_option("--save-osg-kalman-h5",type='string')
    parser.add_option("--save-osg-obj",type='int')
    parser.add_option("--save-osg-frame",type='int')

    (options, args) = parser.parse_args()

    if (options.save_osg_dirname is not None or
        options.save_osg_obj is not None or
        options.save_osg_frame is not None):

        save_osg_info=dict(dirname=options.save_osg_dirname,
                           kalman_h5=options.save_osg_kalman_h5,
                           obj=options.save_osg_obj,
                           frame=options.save_osg_frame,
                           )
        main(connect_to_mainbrain=False,
             save_osg_info=save_osg_info)
    elif options.standalone:
        global sendsock_host
        sendsock_host = ('',30041)
        main(connect_to_mainbrain=False)
    else:
        try:
            main(connect_to_mainbrain=True,
                 live_demo=options.live_demo)
        except Pyro.errors.URIError:
            sys.stderr.write('could not connect to mainbrain. (Hint: re-try '
                             'with --standalone option.) Quitting.\n')
            sys.exit(1)

if __name__=='__main__':
    main_wrap()

