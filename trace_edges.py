#!/usr/bin/env python
import VisionEgg
VisionEgg.start_default_logging(); VisionEgg.watch_exceptions()

from VisionEgg.Core import *
import pygame
from pygame.locals import *
import Numeric
import numpy as np
import time
import copy


def eqn_of_line (p1,p2,t):
    my = ((p2[1]-p1[1]) / t)
    by = p1[1]
    mx = ((p2[0]-p1[0]) / t)
    bx = p1[0]
    return my,by,mx,bx


class Edge:

    def __init__(self,p1,p2):
        self.p1 = [float(i) for i in p1]
        self.p2 = [float(i) for i in p2]
        
        # find the dimension we can parameterize along that is not zero: self.t
        h = np.abs(self.p2[0]-self.p1[0])
        v = np.abs(self.p2[1]-self.p1[1])
        
        if h > v:
            self.t = np.abs(h)
            self.t0 = min(self.p1[0],self.p2[0])
        elif v > h:
            self.t = np.abs(v)
            self.t0 = min(self.p1[1],self.p2[1])
        else:
            self.t = 1
            self.t0 = 0
            print 'Warning: this edge is not a line!'
            
        self.my,self.by,self.mx,self.bx = eqn_of_line(self.p1,self.p2,self.t)
        
    def eqn (self,t):       
        y = self.my*t+self.by
        x = self.mx*t+self.bx
        return x,y
        
    def trace (self):
        param = np.arange(0,self.t+1)
        x,y = self.eqn(param)
        return x,y

class Wall:

    def __init__(self,LL,UL,UR,LR):
        
        self.LL = LL
        self.UL = UL
        self.UR = UR
        self.LR = LR
        
        # find bounding box:         
        self.left_bound = min(LL[0],UL[0])
        self.top_bound = max(UL[1],UL[1])
        self.right_bound = max(UR[0],LR[0])
        self.bottom_bound = min(LR[1],LL[1])
        
        self.screen_h = self.right_bound-self.left_bound
        self.screen_v = self.top_bound-self.bottom_bound
        
        # define the edges
        left_edge = Edge(self.UL,self.LL)
        top_edge = Edge(self.UR,self.UL)
        right_edge = Edge(self.LR,self.UR)
        bottom_edge = Edge(self.LL,self.LR)
        
        # put all edges in a list:
        self.edges = [left_edge,top_edge,right_edge,bottom_edge]
  
    
            
        
def main():



    screen_horiz_pixels = 1024
    screen_vert_pixels = 768
    pix_rad = 2

    os.environ['SDL_VIDEO_WINDOW_POS']="0,0"
    screen=Screen(size=(screen_horiz_pixels ,screen_vert_pixels),
                      bgcolor=(0,0,0),
                      alpha_bits=8,
                      frameless=True
                      #fullscreen=True,
                      )
    screen.set( bgcolor = (0.0,0.0,0.0) ) # black (RGB)
    #screen.set( bgcolor = (.5,.5,.5) ) # gray (RGB)

    white_data = (Numeric.zeros((screen_horiz_pixels,screen_vert_pixels,3))).astype(Numeric.UnsignedInt8)
    #red_data = white_data.copy()
    #red_data[:,:,1:] = 0 # zero non-red channels

    #blue_data = white_data.copy()
    #blue_data[:,:,:-1] = 0 # zero non-blue channels

    frame_timer = FrameTimer() # start frame counter/timer
    count = 0
    quit_now = 0

    # This style of main loop is an alternative to using the
    # VisionEgg.FlowControl module.
    #while not quit_now:
        #for event in pygame.event.get():
            #if event.type in (QUIT,KEYDOWN,MOUSEBUTTONDOWN):
                #quit_now = 1
                
    if 0:
        
        
        
        # first show blank screen (for background image), on mouse click begin tracing
        pixels = copy.copy(white_data)
        
        blank = True
        while blank:
            screen.put_pixels(pixels=pixels,
                              position=(0,0),
                              anchor="lowerleft"
                              )
            swap_buffers() # display what we've drawn
            frame_timer.tick() # register frame draw with timer
        
            for event in pygame.event.get():
                if event.type in (QUIT,KEYDOWN,MOUSEBUTTONDOWN):
                    blank = False
        
        
        
        # (vertical, horizontal)
        
        LL = (50,50)
        UL = (670,1+pix_rad)
        UR = (768-pix_rad,512)
        LR = (1+pix_rad,512)
        wall_1 = Wall(LL,UL,UR,LR)

        
        walls = [wall_1]
        
        for wall in walls:
        
            pixels = copy.copy(white_data)
        
            # initialize - numbers dont really matter
            old_px = 1+pix_rad
            old_py = 1+pix_rad
            
            
            # trace the edges of this wall
            for edge in wall.edges:
                x,y = edge.trace()

                for i in range(len(x)):
                
                    screen.clear()
                
                    # figure out if we start at beginning or end, to get a nice continuous trace - check first and last new pixel to see which is closer to old pixel.
                    if i == 0:
                        px_0 = int(np.round(x[0]))
                        py_0 = int(np.round(y[0]))
                        
                        px_0_err = np.abs(px_0 - old_px)
                        py_0_err = np.abs(py_0 - old_py)
                        p_0_err = px_0_err + py_0_err
                        
                        px_f = int(np.round(x[-1]))
                        py_f = int(np.round(y[-1]))
                        
                        px_f_err = np.abs(px_f - old_px)
                        py_f_err = np.abs(py_f - old_py)
                        p_f_err = px_f_err + py_f_err
                        
                        
                        if p_0_err < p_f_err:
                            direction = 'forward'
                        elif p_f_err < p_0_err:
                            direction = 'backward'
                    
                    if direction is 'forward':
                        counter = i
                    elif direction is 'backward':
                        counter = len(x)-i-1
                
                
                    px = int(np.round(x[counter]))
                    py = int(np.round(y[counter]))
                    
                    print px, py
                 
                    # get a random pixel (entire screen)
                    #px = np.random.random_integers(1+pix_rad,screen_horiz_pixels-pix_rad)
                    #py = np.random.random_integers(2+pix_rad,screen_vert_pixels-pix_rad)
                    
                    # make old pixels black again:
                    pixels[old_px-pix_rad:old_px+pix_rad,old_py-pix_rad:old_py+pix_rad,:] = (Numeric.ones((pix_rad*2,pix_rad*2,3))*0).astype(Numeric.UnsignedInt8)
                    
                    # make new pixels white
                    pixels[px-pix_rad:px+pix_rad,py-pix_rad:py+pix_rad,:] = (Numeric.ones((pix_rad*2,pix_rad*2,3))*255).astype(Numeric.UnsignedInt8)

                
                    screen.put_pixels(pixels=pixels,
                                      position=(0,0),
                                      anchor="lowerleft"
                                      )
                    swap_buffers() # display what we've drawn
                    frame_timer.tick() # register frame draw with timer
                    
                    
                    old_px = px
                    old_py = py
                    
                    
                    
        # keep showing blank screen to allow for quitting flydra etc. until mousclick
        pixels = copy.copy(white_data)
        
        blank = True
        while blank:
            screen.put_pixels(pixels=pixels,
                              position=(0,0),
                              anchor="lowerleft"
                              )
            swap_buffers() # display what we've drawn
            frame_timer.tick() # register frame draw with timer
        
            for event in pygame.event.get():
                if event.type in (QUIT,KEYDOWN,MOUSEBUTTONDOWN):
                    blank = False
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                
        
        
    frame_timer.log_histogram()


if __name__=='__main__':
    main()

