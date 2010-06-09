from __future__ import division
import numpy as np
import warnings
import flydra.geom as geom
import cgtypes # cgkit 1.x

# See http://www.cs.utah.edu/classes/cs6360/Lectures/frustum.pdf for a
# description of some of this stuff.

# See also "The Visual Display Transformation" by Warren Robinett and
# Richard Holloway and "Transparently supporting a wide range of VR
# and stereoscopic display devices" by Dave Pape, Dan Sandin, Tom
# DeFanti

# 3D wind tunnel coordinates (These are made homogenous, an extra
# coordinate of value unity is appended, so that a translation can be
# implemented by a linear transformation, making this a full affine
# transform.)

vec_names_3d = ['LL_3d', 'UL_3d', 'UR_3d', 'LR_3d']

D2R = np.pi/180.0

def make_frustum(l,r,b,t,n,f):
    # from OpenGL specification for glFrustum
    return np.array( [[ 2*n/(r-l), 0, (r+l)/(r-l), 0],
                      [0, 2*n/(t-b), (t+b)/(t-b), 0],
                      [0,0,-(f+n)/(f-n),-2*f*n/(f-n)],
                      [0,0,-1,0]], dtype=np.float)

def make_perspective( fovy, aspectRatio, zNear, zFar ):
    # this was made from osg/Matrix_implementation.cpp
    tan_fovy = np.tan(D2R*fovy*0.5)
    right = tan_fovy * aspectRatio * zNear
    left = -right
    top = tan_fovy * zNear
    bottom = -top
    return make_frustum(left,right,bottom,top,zNear,zFar)

def make_look_at(eye, center, up):
    # simlar to gluLookAt.
    # this was made from osg/Matrix_implementation.cpp
    eye = cgtypes.vec3(*eye)
    center = cgtypes.vec3(*center)
    up = cgtypes.vec3(*up)

    f = center-eye
    f=f.normalize()

    s=f.cross(up)
    s=s.normalize()

    u=s.cross(f)
    u=u.normalize()

    arr = np.array( [[s[0], u[0], -f[0], 0],
                     [s[1], u[1], -f[1], 0],
                     [s[2], u[2], -f[2], 0],
                     [0,0,0,1]], dtype=np.float).T
    trans = np.eye(4)
    trans[:3,3] = [-eye[0], -eye[1], -eye[2]]
    result = np.dot( arr, trans)
    return result

def project_points_onto_plane(verts,plane):
    a,b,c,d=plane
    v = np.array((a,b,c)) # normal to plane
    absv = np.sqrt( np.sum( v**2) )
    C = []
    for pt in verts:
        x0,y0,z0=pt
        # See http://mathworld.wolfram.com/Point-PlaneDistance.html
        dist = (a*x0+b*y0+c*z0+d)/absv
        closest = pt-dist*v
        C.append(closest)
    C = np.array(C)
    return C

def get_plane_of_best_fit(verts):
    vertsH = np.concatenate( (verts, np.ones((4,1))), axis=1) # make homogenous
    #vertsH = verts
    A = np.array( vertsH )
    u,d,vt=np.linalg.svd(A,full_matrices=True)
    Pt = vt[3,:] # plane parameters
    plane = Pt[0:4]
    return plane

def ensure_coplanar(verts,also_ensure_parallel=False):
    """ensure that four vertices are exactly co-planar

    If also_ensure_parallel is True, then assumeing the points are
    laid out in this order:

     1-----------------2
     |                 |
     |                 |
     0-----------------3

    Point 2 will be adjusted so that:
      1->2 is parallel to 0->3
      3->2 is parallel to 0->1
    """

    # step 1: find plane of best fit
    plane = get_plane_of_best_fit( verts )

    # step 2: find points on plane closest to verts
    C = project_points_onto_plane(verts,plane)

    if also_ensure_parallel:
        assert verts.shape == (4,3)
        # step 3: make parallelogram
        #      [ make pt2 = pt0 + (pt3-pt0) + (pt1-pt0) ]
        C[2,:] = C[0,:] + (C[3,:]-C[0,:]) + (C[1,:]-C[0,:])
    return C

def test_ensure_planar_and_parallel():
    all_tests = []

    # ......................
    verts = [ [0,0,0],
              [0,1,0],
              [1,1,0],
              [1,0,0]]
    expected = verts
    all_tests.append( (verts,expected,True) )

    # ......................
    verts = [ [0,0,1],
              [0,1,1],
              [1,1,1],
              [1,0,1]]
    expected = verts
    all_tests.append( (verts,expected,True) )

    ## # ......................
    ## verts = [ [0,0,1.1],
    ##           [0,1,0.9],
    ##           [1,1,1],
    ##           [1,0,1]]
    ## expected = [ [0,0,1],
    ##           [0,1,1],
    ##           [1,1,1],
    ##           [1,0,1]]
    ## all_tests.append( (verts,expected,True) )

    for verts,expected,parallel in all_tests:
        verts = np.array(verts,dtype=np.float)
        expected = np.array(expected,dtype=np.float)
        actual = ensure_coplanar(verts,also_ensure_parallel=parallel)
        assert actual.shape==expected.shape
        assert np.allclose(actual,expected)

class VRScreenData:

    def __init__(self,corners_3d,
                 #approx_view_dir=None,
                 name=None):

        self.name = name
        corners_3d = np.asarray( corners_3d )
        assert corners_3d.shape == (4,3)

        corners_3d = ensure_coplanar(corners_3d,also_ensure_parallel=True)
        self.plane = get_plane_of_best_fit(corners_3d)

        # convert 3D locations to homogeneous vectors ( shape ==(4,1) )
        for i,name in enumerate(vec_names_3d):
            setattr(self,name,list(corners_3d[i])+[1])
            orig = getattr(self,name)
            vec = np.array( [orig], dtype=np.float).T
            assert vec.shape ==(4,1)

            # ensure w == 1
            vec = vec/vec[3] # this won't work for points at infinity
            setattr(self,name,vec)

            tt=geom.ThreeTuple(vec[:3,0])
            setattr(self,name[:2],tt)

        self.X = self.LR-self.LL
        self.Y = self.UL-self.LL

        self.width = abs(self.X)
        self.X_S = self.X*(1./self.width)

        self.height = abs(self.Y)
        self.Y_S = self.Y*(1./self.height)
        self.Z_S = self.X_S.cross(self.Y_S)
        self.unit_Z = self.Z_S * (1.0/abs(self.Z_S))

        # screen to world coords
        self.s2wld = np.array( [[self.X_S[0], self.Y_S[0], self.Z_S[0], 0],
                                [self.X_S[1], self.Y_S[1], self.Z_S[1], 0],
                                [self.X_S[2], self.Y_S[2], self.Z_S[2], 0],
                                [ 0,       0 ,     0,    1]],
                               dtype=np.float)
        # world to screen coords
        self.view1 = np.dual.inv(self.s2wld)

    def get_view_and_projection_for_eye(self,eye,near,far,avoid_clipping=False):
        """Get view and projection matrices

        Arguments
        ---------
        eye : vector
            Eye position
        near : float
            Near clipping plane in world coordinate units
        far : float
            Far clipping plane in world coordinate units

        Returns
        -------
        view_matrix : array
        frustum_parameters : tuple

        See 'Transparently supporting a wide range of VR and
        stereoscopic display devices' by Dave Pape, Dan Sandin, Tom
        DeFanti.
        """
        while 1:
            E_S = eye-self.LL

            L = E_S.dot(self.X_S)
            R = self.width-L
            B = E_S.dot(self.Y_S)
            T = self.height-B

            distance = E_S.dot(self.Z_S)
            if (not avoid_clipping) or (distance >= near):
                # normal case
                break
            else:
                warnings.warn('moving eye back to avoid clipping')
                eye = eye + 1e-3*self.unit_Z # move 1 mm if units are meters
        if 1:
            frac = near/distance
            l = -L*frac
            r =  R*frac
            b = -B*frac
            t =  T*frac

            n = near
            f = far

            trans = np.eye(4)
            trans[:3,3] = [-eye[0], -eye[1], -eye[2]]
            view = np.dot(self.view1,trans)
        return view, (l,r,b,t,n,f)

    def update_VE_viewport(self,viewport,eye_loc,near,far,avoid_clipping=False):
        view_matrix, frustum=self.get_view_and_projection_for_eye(
            geom.ThreeTuple(eye_loc),near,far,avoid_clipping=avoid_clipping)

        viewport.parameters.camera_matrix.parameters.matrix = view_matrix.T

        projection_matrix = make_frustum(*frustum)
        viewport.parameters.projection.parameters.matrix = projection_matrix.T
