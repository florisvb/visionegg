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
import FlatRect
import VR_math

import os
import time


pi = np.pi

#SHOW='cam:floor'
#SHOW='cam:test'
#SHOW='cam:+y'
#SHOW='overview'
SHOW='projector (calibrated)'

FOREST=False
DRUM = False
DOTS = False
UNIT_CUBE = False

BGCOLOR = (1,1,1)

TRIAL_DURATION = 30.0 # seconds
DO_EXPERIMENT = True







#eye_loc_default = (0.35, -.05, 0.15)



def get_wall_dict( show, screen, corners_3d, corners_2d,
                   approx_view_dir, name):
    proj_math = VR_math.VRScreenData(corners_3d,#approx_view_dir=approx_view_dir,
                                     name=name)
    cam_viewport = Viewport(screen=screen,
                            auto_pixel_projection=False,
                            )
    result = dict(screen_data=proj_math,
                  cam_viewport=cam_viewport)
    
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

def main(live_demo = False, # if True, never leave VR mode
         ):


    

    os.environ['SDL_VIDEO_WINDOW_POS']="0,0"
    screen=Screen(size=(1024,768),
                  bgcolor=(0,0,0),
                  alpha_bits=8,
                  frameless=True
                  #fullscreen=True,
                  )

    if 1:
        # this loads faster, for debugging only
        mipmaps_enabled=False
    else:
        mipmaps_enabled=True

    DEBUG='none' # no debugging
    #DEBUG='flicker' # alternate views on successive frames



    vr_walls = {}
    
    
        
    ########## WALL 2 #################
    if 1:
        # order: LL, UL, UR, LR
        # -y wall
        corners_3d = [
            # LL
            (0.43654656410217285, -0.1443936675786972, 0.025561617687344551),
            # UL
            (0.4312690794467926, -0.14406071603298187, 0.27475747466087341),
            # UR
            (0.2, -0.14475925266742706, 0.23684273660182953), #-0.49650067090988159
            # LR
            (0.2, -0.14341457188129425, -0.018632272258400917), #-0.52826571464538574
            ]

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

        name = '-y'
        approx_view_dir = None
        vr_wall_dict = get_wall_dict( SHOW, screen,
                                      corners_3d, corners_2d,
                                      approx_view_dir,name)
        vr_walls[name] = vr_wall_dict
        
        
    ########## WALL 3 #################
    if 1:
        # order: LL, UL, UR, LR
        # -y wall
        corners_3d = [
            # LL = 1.LR
            (0.42773362994194031, 0.14023984968662262, 0.022966273128986359),
            # UL = 1.UR
            (0.42402097582817078, 0.14513731002807617, 0.25622266530990601),
            # UR = 2.UL
            (0.4312690794467926, -0.14406071603298187, 0.27475747466087341),
            # LR = 2.LL
            (0.43654656410217285, -0.1443936675786972, 0.025561617687344551),
            ]

        corners_2d = [
            # LL
            #(81, 573), # good 3D number for this point, but not directly under UL
            (1,50),
            # UL
            (1,620),
            # UR
            (512,768),
            # LR
            (512,1),
            ]

        name = '+x'
        approx_view_dir = None
        vr_wall_dict = get_wall_dict( SHOW, screen,
                                      corners_3d, corners_2d,
                                      approx_view_dir,name)
        vr_walls[name] = vr_wall_dict
        
    


    screen_stimuli = []
    for wall in vr_walls.itervalues():
        screen_stimuli.extend( wall['display_stimuli'] )

    
    display_viewport = Viewport(
        screen=screen,
        stimuli=screen_stimuli,
        )


    # OpenGL textures must be power of 2
    def next_power_of_2(f):
        return math.pow(2.0,math.ceil(math.log(f)/math.log(2.0)))
    fb_width_pow2  = int(next_power_of_2(screen.size[0]))
    fb_height_pow2  = int(next_power_of_2(screen.size[1]))

    frame_timer = FrameTimer()

	
	
	# draw static white walls
    draw_stimuli = True
    for wallname,wall in vr_walls.iteritems():
        
        screen.set(bgcolor=BGCOLOR)
        screen.clear() # clear screen
        wall['cam_viewport'].draw()


        framebuffer_texture_object = wall['framebuffer_texture_object']

        # copy screen back-buffer to texture
        framebuffer_texture_object.put_new_framebuffer(
            size=(fb_width_pow2,fb_height_pow2),
            internal_format=gl.GL_RGB,
            buffer='back',
            )
        
    screen.set(bgcolor=(0.0,0.0,0.0)) #black
    screen.clear() # clear screen
    display_viewport.draw() # draw the viewport and hence the stimuli
    swap_buffers() # swap buffers
        
    
    
    quit_now = False
    while not quit_now:
        # infinite loop to draw stimuli
        

        # test for keypress or mouseclick to quit
        for event in pygame.event.get():
            if event.type in (QUIT,MOUSEBUTTONDOWN):
            #if event.type in (QUIT,KEYDOWN,MOUSEBUTTONDOWN):
                quit_now = True

        pass

        
        

	
       



if __name__=='__main__':
    main()

