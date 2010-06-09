from VisionEgg.Textures import TextureStimulusBaseClass, SpinningDrum
import VisionEgg.ParameterTypes as ve_types
import VisionEgg.GL as gl # get all OpenGL stuff in one namespace
import cgtypes # cgkit 1.x

class FlatRect(SpinningDrum):
    parameters_and_defaults = {

        'tex_phase':(0.0, # varies between 0-1
                     ve_types.Real),

        'lowerleft':((0.0,0.0,-1.0),
                     ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                    ve_types.Sequence4(ve_types.Real)),
                     "vertex position (units: eye coordinates)"),
        'lowerright':((1.0,0.0,-1.0),
                      ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                     ve_types.Sequence4(ve_types.Real)),
                      "vertex position (units: eye coordinates)"),
        'upperleft':((0.0,1.0,-1.0),
                     ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                    ve_types.Sequence4(ve_types.Real)),
                     "vertex position (units: eye coordinates)"),
        'upperright':((1.0,1.0,-1.0),
                      ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                     ve_types.Sequence4(ve_types.Real)),
                      "vertex position (units: eye coordinates)"),
        'depth_test':(False,
                      ve_types.Boolean),
        }
    def __init__(self,**kw):
        SpinningDrum.__init__(self,**kw)

    def draw(self):
    	"""Redraw the stimulus on every frame.
        """
        p = self.parameters
        if p.texture != self._using_texture: # self._using_texture is from TextureStimulusBaseClass
            self._reload_texture()
            self.rebuild_display_list()
        if p.on:
            # Set OpenGL state variables
            if p.depth_test:
                gl.glEnable(  gl.GL_DEPTH_TEST )
            else:
                gl.glDisable( gl.GL_DEPTH_TEST )
            gl.glEnable( gl.GL_TEXTURE_2D )  # Make sure textures are drawn
            gl.glEnable( gl.GL_BLEND ) # Contrast control implemented through blending

            # All of the contrast control stuff is somewhat arcane and
            # not very clear from reading the code, so here is how it
            # works in English. (Not that it makes it any more clear!)
            #
            # In the final "textured fragment" (before being blended
            # to the framebuffer), the color values are equal to those
            # of the texture (with the exception of texels around the
            # edges which have their amplitudes reduced due to
            # anti-aliasing and are intermediate between the color of
            # the texture and mid-gray), and the alpha value is set to
            # the contrast.  Blending occurs, and by choosing the
            # appropriate values for glBlendFunc, adds the product of
            # fragment alpha (contrast) and fragment color to the
            # product of one minus fragment alpha (contrast) and what
            # was already in the framebuffer.

            gl.glBlendFunc( gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA )

            gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_DECAL)

            # clear modelview matrix
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glPushMatrix()
            try:
                gl.glColor4f(0.5,0.5,0.5,p.contrast) # Set the polygons' fragment color (implements contrast)

                if not self.constant_parameters.mipmaps_enabled:
                    if p.texture_min_filter in TextureStimulusBaseClass._mipmap_modes:
                        raise RuntimeError("Specified a mipmap mode in texture_min_filter, but mipmaps not enabled.")
                self.texture_object.set_min_filter( p.texture_min_filter )
                self.texture_object.set_mag_filter( p.texture_mag_filter )
                self.texture_object.set_wrap_mode_s( p.texture_wrap_s )
                self.texture_object.set_wrap_mode_t( p.texture_wrap_t )

                if 1:

                    tex_phase = p.tex_phase % 1.0 # make 0 <= tex_phase < 1.0

                    TINY = 1.0e-10
                    tex = p.texture
                    tex.update()

                    if tex_phase < TINY: # it's effectively zero

                        gl.glBegin(gl.GL_QUADS)
                        gl.glTexCoord2f(tex.buf_lf,tex.buf_bf)
                        gl.glVertex(*p.lowerleft)

                        gl.glTexCoord2f(tex.buf_rf,tex.buf_bf)
                        gl.glVertex(*p.lowerright)

                        gl.glTexCoord2f(tex.buf_rf,tex.buf_tf)
                        gl.glVertex(*p.upperright)

                        gl.glTexCoord2f(tex.buf_lf,tex.buf_tf)
                        gl.glVertex(*p.upperleft)
                        gl.glEnd() # GL_QUADS

                    else:
                        # Convert tex_phase into texture buffer fraction
                        buf_break_f = ( (tex.buf_rf - tex.buf_lf) * (1.0-tex_phase) ) + tex.buf_lf

                        r = cgtypes.vec3(p.lowerright)
                        l = cgtypes.vec3(p.lowerleft)
                        quad_x_lower = (r-l)*tex_phase + l

                        r = cgtypes.vec3(p.upperright)
                        l = cgtypes.vec3(p.upperleft)
                        quad_x_upper = (r-l)*tex_phase + l

                        gl.glBegin(gl.GL_QUADS)

                        # First quad

                        gl.glTexCoord2f(buf_break_f,tex.buf_bf)
                        gl.glVertex(*p.lowerleft)

                        gl.glTexCoord2f(tex.buf_rf,tex.buf_bf)
                        gl.glVertex(*quad_x_lower)

                        gl.glTexCoord2f(tex.buf_rf,tex.buf_tf)
                        gl.glVertex(*quad_x_upper)

                        gl.glTexCoord2f(buf_break_f,tex.buf_tf)
                        gl.glVertex(*p.upperleft)

                        # Second quad

                        gl.glTexCoord2f(tex.buf_lf,tex.buf_bf)
                        gl.glVertex(*quad_x_lower)

                        gl.glTexCoord2f(buf_break_f,tex.buf_bf)
                        gl.glVertex(*p.lowerright)

                        gl.glTexCoord2f(buf_break_f,tex.buf_tf)
                        gl.glVertex(*p.upperright)

                        gl.glTexCoord2f(tex.buf_lf,tex.buf_tf)
                        gl.glVertex(*quad_x_upper)
                        gl.glEnd() # GL_QUADS

            finally:
                gl.glMatrixMode(gl.GL_MODELVIEW)
                gl.glPopMatrix()
