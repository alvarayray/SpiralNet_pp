from __future__ import print_function
import os, re, json
import numpy as np
import vtk
from collections import Counter
from utils.progress_bar import ProgressBarWrapper
from collections import defaultdict

import functools

class LazyProperty(object):

    def __init__(self, func):
        self.func = func
        functools.update_wrapper(self, func)

    def __get__(self, obj, objtype):
        if obj is None:
            return self

        value = self.func(obj)
        setattr(obj, self.func.__name__, value)
        return value
 
class OBJ():
    
    """ generic function of loading and exporting OBJ file

        ***** attributes *****

        vtxs
        --------------
        shape (num_vtxs, 3): numpy array of the 3D vertices

        faces
        --------------
        shape (num_faces, n): numpy array of the faces, `n` represents max of (the number of vertices on a face). the values are vertex IDs (0-indexed)

        triangular_faces
        --------------
        shape (num_triangular_faces, 3): numpy array of the faces after triangularization, also 0-indexed vertex IDs

        normals
        --------------
        list of normals as in the obj file <to be supplemented>

        textures
        --------------
        list of textures as in the obj file <to be clarified>

        path
        --------------
        str: the path to the .obj file loaded


        ***** property *****

        has_triangular_faces_only
        --------------
        boolean: True / False indicating if it is a mesh with only triangular faces



        ***** methods *****

        obj = OBJ.load_file('/home/alvin/test.obj')
        s = obj.export_str()   # export the obj as string
        obj.export_file('/home/alvin/text.obj', overwrite=False)

    """
    symmetry_tolerance = 0.000001

    def __init__(self, vtxs=None, faces=None, triangular_faces=None, normals=None, textures=None, path=None):

        self.vtxs = vtxs
        self.faces = faces
        self.triangular_faces = triangular_faces
        self.normals = normals
        self.textures = textures
        self.path = path

        if triangular_faces is None and faces is not None:
            # below is not in an utilities function yet, because it is still a mess for the MeshGenerator.triangulate_faces function due to -1 issue
            max_vertices_in_face = max([len(vtxs_id) for vtxs_id in faces])

            if max_vertices_in_face == 3:
                triangular_faces = faces.copy()

            else:  # contain quad-face, 5-vertex faces, or more
                # triangulate_faces function from MeshGenerator gets 1-index as input and return 0-index as result!!!
                # so need to +1 to get 1-index as input to return 0-index as output!!!
                triangular_faces = MeshGenerator.triangulate_faces(faces + 1).copy()

            self.triangular_faces = triangular_faces

    def __eq__(self, obj):
        """ only checking vertices and faces, not checking normals/textures/paths
        """
        if not np.array_equal(self.vtxs, obj.vtxs):
            return False
        if not np.array_equal(self.faces, obj.faces):
            return False
        if not np.array_equal(self.triangular_faces, obj.triangular_faces):
            return False

        return True

    def __ne__(self, obj):
        return not self.__eq__(obj)


    def __copy__(self):
        new_copy = type(self)(**self.__dict__)
        return new_copy

    @property
    def has_triangular_faces_only(self):
        return np.array_equal(self.faces, self.triangular_faces)

    @LazyProperty
    def neighbour_faces(self):
        """ get all neighbouring face ids for each face
            only work for triangular faces for now
            return {
                0: set([2, 3, 4, 5, 6, 7]),
                2: set([0, 3, 4, 5, 8, 9]),
                ...
            }
        """
        vtx_in_face = defaultdict(list)
        mapping = defaultdict(list)

        for face_id in range(len(self.triangular_faces)):
            face = self.triangular_faces[face_id]
            for vtx_id in face:
                vtx_in_face[vtx_id].append(face_id)

        for face_id in range(len(self.triangular_faces)):
            face = self.triangular_faces[face_id]
            for vtx_id in face:
                possible_face_ids = vtx_in_face[vtx_id]
                mapping[face_id].extend(possible_face_ids)
            mapping[face_id] = set(mapping[face_id])
            # remove itself
            mapping[face_id].remove(face_id)

        return mapping

    def find_neighbour_face_ids(self, face_id):
        return self.neighbour_faces[face_id]

    @LazyProperty
    def edges(self):
        """ get all neighbouring vertices for each vertex
            {
                vtx_id1 : set([1, 2, 3]),
                vtx_id2 : set([2, 4, 5]),
                ...
            }
        """
        edge_mapping = defaultdict(list)
        for face_id in range(len(self.triangular_faces)):
            face = self.triangular_faces[face_id]
            for vtx_id in face:
                edge_mapping[vtx_id].extend(face.tolist())

        for vtx_id in edge_mapping:
            edge_mapping[vtx_id] = set(edge_mapping[vtx_id])
            edge_mapping[vtx_id].remove(vtx_id)

        return edge_mapping

    def find_neighbour_vtx_ids(self, vtx_id):
        return self.edges[vtx_id]

    @LazyProperty
    def edge_face_mapping(self):
        """ get all faces that an edge (formed by 2 vertices) belongs to
            return {
                (vtx_id1, vtx_id2): [face_id1, face_id2, ...],
            }
        """
        edge_face_mapping = defaultdict(list)
        for face_id in range(len(self.triangular_faces)):
            face = self.triangular_faces[face_id]
            edge_face_mapping[(face[0], face[1])].append(face_id)
            edge_face_mapping[(face[1], face[0])].append(face_id)
            edge_face_mapping[(face[0], face[2])].append(face_id)
            edge_face_mapping[(face[2], face[0])].append(face_id)
            edge_face_mapping[(face[1], face[2])].append(face_id)
            edge_face_mapping[(face[2], face[1])].append(face_id)
        return edge_face_mapping

    def find_face_ids_from_edge(self, vtx_id1, vtx_id2):
        return self.edge_face_mapping[(vtx_id1, vtx_id2)]

    def is_symmetric(self, debug=False):
        return self._check_symmetry(debug=debug)['is_symmetric']

    def get_symmetry_mapping(self, debug=False):
        return self._check_symmetry(debug=debug)['symmetry_mapping']

    @staticmethod
    def get_global_symmetry_mapping(vbid_version):
        # not using import MODEL_DIR to avoid circular imports
        file_path = os.path.join('/srv/machine_learning/models/assets', 'ground_truth', 'obj_symmetry_mapping', 'symmetry_mapping_v%d.json' % vbid_version)
        if not os.path.exists(file_path):
            raise FileNotFoundError()

        with open(file_path) as f:
            mapping = json.load(f)

        # convert string key to int
        m = {}
        for k,v in mapping.items():
            m[int(k)] = mapping[k]

        return m

    def _check_symmetry(self, debug=False):
        # divide the vertices into left, middle and right parts with the x-coordinate
        # by convention, the left hand of a body should be pointing to the +ve x-axis
        left_mask = self.vtxs[:,0] > 0
        middle_mask = self.vtxs[:,0] == 0
        right_mask = self.vtxs[:,0] < 0

        if debug:
            print('{} vertices on the left. (x > 0)'.format(left_mask.sum()))
            print('{} vertices on the middle. (x = 0)'.format(middle_mask.sum()))
            print('{} vertices on the right. (x < 0)'.format(right_mask.sum()))

        if left_mask.sum() != right_mask.sum():
            print('Error: no. of vertices are different on the left ({}) and right ({})'.format(left_mask.sum(), right_mask.sum()))
            return {'is_symmetric': False, 'symmetry_mapping': None}

        def find_symmetry_point_idx(vtxs, vtx):
            # suppose there is an imaginary perfect symmetric vertex of the input vertex
            symmetry_vtx = [-vtx[0], vtx[1], vtx[2]]
            # look for the existing vertex which has the shortest Manhattan distance with the imaginary symmetric vertex
            abs_diff = np.abs(vtxs - symmetry_vtx).sum(axis=1)
            min_value = np.min(abs_diff)

            symmetry_idx = np.argmin(abs_diff)
            return symmetry_idx, min_value

        symmetry_mapping = {}
        symmetric = True

        for i, vtx in enumerate(self.vtxs):
            # find the symmetric vertices of the ones on the left side
            # leave the right vertices untouched
            if right_mask[i]:
                continue

            if middle_mask[i]:
                symmetry_mapping[i] = i
                continue

            symmetry_idx, diff = find_symmetry_point_idx(self.vtxs, vtx)
            if diff > self.symmetry_tolerance:
                if debug:
                    print('Error: The point can\'t be paired idx:{} ({}), closest idx {} ({})'.format(i, vtx, symmetry_idx, self.vtxs[symmetry_idx]))
                symmetric = False
                continue

            # the vertex with index i and the vertex with symmetry_idx are on the same side
            if left_mask[i] ^ right_mask[symmetry_idx]:
                if debug:
                    print('Error: The symmetry point pair {} ({}) and {} ({}) should not be on the same side.'.format(i, vtx, symmetry_idx, self.vtxs[symmetry_idx]))
                symmetric = False
                continue
            if symmetry_idx in symmetry_mapping:
                if debug:
                    print('Error: DUPLICATE MATCHING! {} ({}) already paired with {} ({}), cannot pair with {} ({}) again'.format(
                        symmetry_idx, self.vtxs[symmetry_idx], symmetry_mapping[symmetry_idx], self.vtxs[symmetry_mapping[symmetry_idx]], i, vtx)
                    )
                symmetric = False
                continue

            # create the mapping pair in the symmetry_mapping dict
            symmetry_mapping[i] = symmetry_idx
            symmetry_mapping[symmetry_idx] = i

        if symmetric:
            return {'is_symmetric': True, 'symmetry_mapping': symmetry_mapping}
        else:
            return {'is_symmetric': False, 'symmetry_mapping': None}

    def create_symmetric_obj(self, symmetric_mapping, alg='average', debug=True):
        '''
        Create a symmetric object with adjusted vertex coordinates

        Params:

        symmetric_mapping (dict): a dict indicating the symmetric pair of each vertex using their indices
        alg (str): the basis of coordinate adjustment
            'average': follow the average of the symmetric pairs
            'left: follow the left side of the symmetric pairs
            'right': follow the right side of the symmetric pairs
        debug (bool):

        Returns:

        OBJ (object): the OBJ object with adjusted vertex coordinates

        '''
        if alg not in ['left', 'right', 'average']:
            raise Exception('set alg to be either "left", "right" or "average"')

        if debug:
            print('Checking symmetry...')
        sym_result = self._check_symmetry()
        if sym_result['is_symmetric']:
            print('this OBJ is already symmetric')
            return None

        vtxs = self.vtxs.copy()

        # for each symmetric vertex pairs
        for idx1, idx2 in symmetric_mapping.items():
            # self-mapping
            if idx1 == idx2:
                # if the x coordinate of the vertice is not 0
                if vtxs[idx1][0] != 0:
                    if debug:
                        print('vtx #{}: {} to {}'.format(idx1, vtxs[idx1], [0, vtxs[idx1][1], vtxs[idx1][2]]))
                    vtxs[idx1][0] = 0

            else:
                # if the vtxs[idx1] is not a mirror of vtxs[idx2] along the x-direction
                if not np.array_equal(vtxs[idx1] * [-1, 1, 1], vtxs[idx2]):

                    if alg == 'average':
                        average_vtx = ( vtxs[idx1] * [-1, 1, 1] + vtxs[idx2] ) / 2
                        if debug:
                            print('vtx #{}: {} to {}'.format(idx1, vtxs[idx1], average_vtx * [-1, 1, 1]))
                            print('vtx #{}: {} to {}'.format(idx2, vtxs[idx2], average_vtx))
                        vtxs[idx1] = average_vtx * [-1, 1, 1]
                        vtxs[idx2] = average_vtx
                        continue
                    # assume to be left side or right side of the body for these alg types
                    # alg = left means it follows left side of the body
                    # left side : x-coordinate is positive
                    if vtxs[idx1][0] < 0:
                        left_idx = idx2
                        right_idx = idx1
                    else:
                        left_idx = idx1
                        right_idx = idx2

                    if alg == 'left':
                        if debug:
                            print('vtx #{}: {} to {}'.format(right_idx, vtxs[right_idx], vtxs[left_idx] * [-1, 1, 1]))
                        # change the coordinates of the right vertex according to the left index
                        vtxs[right_idx] = vtxs[left_idx] * [-1, 1, 1]
                    elif alg == 'right':
                        if debug:
                            print('vtx #{}: {} to {}'.format(left_idx, vtxs[left_idx], vtxs[right_idx] * [-1, 1, 1]))
                        # change the coordinates of the left vertex according to the right index
                        vtxs[left_idx] = vtxs[right_idx] * [-1, 1, 1]

        return OBJ(vtxs=vtxs, faces=self.faces)

    @staticmethod
    def read_lines(lines, vtxs_only):

        vertices = []
        faces = []
        normals = []
        textures = []

        if vtxs_only:
            for line in lines:
                if line.startswith('v '):
                    vertices.append(line.split()[1:4])
            vertices = np.array(vertices, dtype=float)
            return {
                'vertices': vertices,
                'faces': [],
                'triangular_faces': [],
                'textures': [],
                'normals': [],
            }

        for line in lines:

            if not line.strip():
                continue

            if line.startswith('v '):
                vertices.append([float(v) for v in line.split()[1:4]])

            elif line.startswith('f '):
                # only handle this specific type: f 6/4/1 3/5/3 7/6/5 (v/vt/vn)
                # get vertex ID only
                faces.append(
                    [
                        int(v_vt_vn.split('/')[0])
                        for v_vt_vn
                        in line.split()[1:]
                    ]
                )

            elif line.startswith('vn '):
                normals.append([float(v) for v in line.split()[1:]])

            elif line.startswith('vt '):
                textures.append([float(v) for v in line.split()[1:]])

        vertices = np.array(vertices)
        faces = np.array(faces, dtype=np.int32)

        max_vertices_in_face = max([len(vtxs_id) for vtxs_id in faces])

        if max_vertices_in_face == 3:
            # make it 0-index for python
            faces = faces - 1
            triangular_faces = faces.copy()

        else:  # contain quad-face, 5-vertex faces, or more
            # triangulate_faces function from MeshGenerator gets 1-index as input and return 0-index as result!!!
            # so only minus 1 for faces below
            triangular_faces = MeshGenerator.triangulate_faces(faces).copy()
            faces = faces - 1

        return {
            'vertices': vertices,
            'faces': faces,
            'triangular_faces': triangular_faces,
            'textures': textures,
            'normals': normals,
        }

    @classmethod
    def from_str(cls, s):
        lines = s.split('\n')
        result = cls.read_lines(lines)

        vertices = result['vertices']
        faces = result['faces']
        triangular_faces = result['triangular_faces']
        normals = result['normals']
        textures = result['textures']

        return cls(vtxs=vertices, faces=faces, triangular_faces=triangular_faces, normals=normals, textures=textures)

    @classmethod
    def load_file(cls, path, vtxs_only=False):

        if not os.path.exists(path):
            raise Exception('File not found: %s' % path)


        with open(path) as f:
            lines = f.readlines()
            result = cls.read_lines(lines, vtxs_only)

        vertices = result['vertices']
        faces = result['faces']
        triangular_faces = result['triangular_faces']
        normals = result['normals']
        textures = result['textures']

        return cls(vtxs=vertices, faces=faces, triangular_faces=triangular_faces, normals=normals, textures=textures, path=path)

    def export_str(self, triangular_faces=False, normals=True, textures=True):

        vtxs_str = '\n'.join(['v {:19.15f} {:19.15f} {:19.15f}'.format(*vtx) for vtx in self.vtxs])

        if triangular_faces:
            faces_str = '\n'.join(['f {} {} {}'.format(*(face+1)) for face in self.triangular_faces])
        else:
            faces_str = '\n'.join(['f ' + ' '.join([str(face_id+1) for face_id in face]) for face in self.faces])

        if normals and self.normals:
            normals_str = '\n'.join(['vn {:19.15f} {:19.15f} {:19.15f}'.format(*vn) for vn in self.normals])
        else:
            normals_str = ''

        if textures and self.textures:
            textures_str = '\n'.join(['vt {:19.15f} {:19.15f}'.format(*texture) for texture in self.textures])
        else:
            textures_str = ''

        obj_str = 'g default\n{vtxs_str}\n{textures_str}\n{normals_str}\n{faces_str}'.format(
            vtxs_str=vtxs_str,
            textures_str=textures_str,
            normals_str=normals_str,
            faces_str=faces_str,
        )

        return obj_str


    def export_file(self, path, overwrite=False, **kwargs):

        path_exists = os.path.exists(path)

        if path_exists and not overwrite:
            raise Exception('file existed, supply another path or overwrite=True')

        obj_file_str = self.export_str(**kwargs)

        with open(path, 'w') as f:
            f.write(obj_file_str)

        if path_exists and overwrite:
            print('Overwritten: %s' % path)
        else:
            print('Exported to: %s' % path)

    def __str__(self):
        return '<OBJ> of vertices (%d, %d) and %d faces' % (
            self.vtxs.shape[0],
            self.vtxs.shape[1],
            len(self.faces),
        )

    def __repr__(self):
        return '<OBJ> of vertices (%d, %d) and %d faces' % (
            self.vtxs.shape[0],
            self.vtxs.shape[1],
            len(self.faces),
        )

def obj_to_vtk(obj, triangular_faces=True):

    if not triangular_faces:
        raise NotImplementedError

    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    for vtx in obj.vtxs:
        pts.InsertNextPoint(*vtx.tolist())
        poly.SetPoints(pts)

    faces = vtk.vtkCellArray()
    for face in obj.triangular_faces:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, face[0])
        triangle.GetPointIds().SetId(1, face[1])
        triangle.GetPointIds().SetId(2, face[2])
        faces.InsertNextCell(triangle)
    poly.SetPolys(faces)

    return poly

def vtk_to_obj(poly):
    """ only handling triangular mesh for now
    """

    vtxs = []
    for i in range(poly.GetNumberOfPoints()):
        x,y,z = poly.GetPoint(i)
        vtxs.append([x,y,z])
    vtxs = np.asarray(vtxs)
    vtxs.shape

    faces = []
    for i in range(poly.GetNumberOfPolys()):
        triangle = poly.GetCell(i)
        vtx_ids = [triangle.GetPointIds().GetId(j) for j in range(3)]
        faces.append(vtx_ids)
    faces = np.asarray(faces)

    obj = OBJ(vtxs=vtxs, faces=faces, triangular_faces=faces)

    return obj


def load_all_children(dir_path, filename_pattern, criterion=lambda _:True):
    '''Yield every matched files in the directory recursively.
    Inputs:
        - dir_path: the directory path
        - filename_pattern: the regelar expression pattern used on the filename itself
        - criterion: the filtering criterion used on the absolute filename:
            - example_pattern = r'[A-Z]{3}-[A-Z|0-9]{7}-[0-9]{4}.*\.obj'
    Outputs:
        - absolute filename that matches the pattern.
    '''
    for file_name in os.listdir(dir_path):
        if re.match(filename_pattern, file_name): # is a matched file
            if criterion(os.path.join(dir_path, file_name)):
                yield os.path.join(dir_path, file_name)
        elif os.path.isdir(os.path.join(dir_path, file_name)): # is a folder
            for result in load_all_children(os.path.join(dir_path, file_name), filename_pattern, criterion):
                yield result

def safe_to_float(aStr):
    try:
        return float(aStr)
    except:
        pass


class MeshGenerator:
    '''
    Example:
    >>> MESH_DATA = '/nfs/zfs/home/ying/alvaform_db'
    >>> mg = MeshGenerator(dir_path=MESH_DATA)FF
    >>> vtx_buffer, f_buffer, vbid_buffer = mg.load_all_meshes(['Men', 'MenXL'])
    '''

    VBID_to_bodytype = None
    bodytype_mapping = None
    cls_list = ['Men', 'MenXL', 'Women', 'WomenXL', 'Boy', 'Girl', 'Infant', 'unknown']

    @staticmethod
    def get_vbid(absolute_path):
        '''get vbid from a absolute file path'''
        # extract the direct parent directory name as the vbid
        return os.path.split(os.path.split(absolute_path)[0])[1]

    @classmethod
    def load_mesh(cls, file_name, want_faces=False, want_original_faces=False, want_normal=False):
        vertices = []
        faces = []
        normals = []
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 0:
                    if line.startswith('v '):
                        vertices.append([float(v) for v in line.split()[1:4]])
                    elif (want_faces or want_original_faces) and line.startswith('f '):
                        faces.append([int(v_vt_vn.split('/')[0]) for v_vt_vn in line.split()[1:]])
                    elif want_normal and line.startswith('vn '):
                        normals.append([float(v) for v in line.split()[1:]])
        vertices = np.array(vertices)
        faces = np.array(faces)
        deliverables = [vertices]
        if want_faces:
            triangle_faces = cls.triangulate_faces(np.asarray(faces))
            deliverables.append(triangle_faces)
        if want_original_faces:
            deliverables.append(faces)
        if want_normal:
            deliverables.append(normals)
        return deliverables if len(deliverables) > 1 else deliverables[0]

    @staticmethod
    def triangulate_faces(original_faces):
        """ this is greedy -- a file may contain faces with both 3 vertices / 4 vertices
            TODO: fix it
        """
        if len(original_faces.shape) == 2:
            # Divide a rectangle into two triangles
            triangle_faces_1 = original_faces[:,:-1]-1
            triangle_faces_2 = original_faces[:,[2,3,0]]-1
            return np.concatenate((triangle_faces_1, triangle_faces_2), axis=0)
        elif len(original_faces.shape) < 2:
            # Divide a polygon into multiple triangles
            # UPDATE: WHAT IS IT? in what cases len(original_faces.shape) can be 1?
            triangle_faces = []
            for face in original_faces:
                for i in range(len(face)-2):
                    triangle_face = [face[0]-1, face[i+1]-1, face[i+2]-1]
                    triangle_faces.append(triangle_face)
            return np.asarray(triangle_faces)

    @staticmethod
    def load_vertices(file_name):
        from pandas import isnull
        with open(file_name, 'r') as f:
            line_long = f.read()
            start_idx = line_long.find('v ')
            end_idx = line_long.find('vt ')
            lines = line_long[start_idx:end_idx].replace('v', '').split()
            vertices_flat = np.asarray([safe_to_float(x) for x in lines])
            vertices = vertices_flat[~isnull(vertices_flat)].reshape(-1,3)
        return vertices

    @staticmethod
    def save_mesh(file_name, output_filename, vtxs):
        vtxs_stepper = 0
        with open(file_name, 'rb') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                tokens = line.split()
                if len(tokens) > 0:
                    if tokens[0] == 'v':
                        vtx = vtxs[vtxs_stepper]
                        vtxs_stepper += 1
                        vtx_str = 'v {:6f} {:6f} {:6f}\n'.format(vtx[0], vtx[1], vtx[2])
                        lines[i] = vtx_str
        f.close()

        print('Start saving {}...'.format(output_filename))
        with open(output_filename, 'wb') as of:
            output_str = ''.join(lines)
            of.write(output_str)
        of.close

    def find_body_type(self, VBID):
        #first time calling
        if self.VBID_to_bodytype is None or self.bodytype_mapping is None:
            # Body type dict
            BODYTYPE_FILE = os.path.join('/nfs/zfs/home/ying/3D_avatar_automation/data_extraction/v3_12032018', 'alvaform_bodytype.json')
            BODYTYPE_MAPPING_FILE = os.path.join('/nfs/zfs/home/ying/3D_avatar_automation/data_extraction/', 'bodytype_mapping.json')

            # Vertex idx
            self.VBID_to_bodytype = json.load(open(BODYTYPE_FILE, 'rb')) # A mapping from VBID to raw body type
            self.bodytype_mapping = json.load(open(BODYTYPE_MAPPING_FILE, 'rb')) # A mapping from standard body type to raw body type

            self.bodytype_mapping['Men'] += ['Men_Tall', 'men', 'Men_Short', 'Men_Large']
            self.bodytype_mapping['MenXL'] += ['Men_XL-XXL']
            self.bodytype_mapping['Boy'] += ['Boy_Slim']
            self.bodytype_mapping['Girl'] += ['Girl_Slim']
            # bodytype_mapping['Women'] += ['Petite_Plus']

        body_type = 'unknown'
        if VBID not in self.VBID_to_bodytype:
            return body_type
        else:
            raw_body_type = self.VBID_to_bodytype[VBID]
            for standard_name, names in self.bodytype_mapping.items():
                if raw_body_type in names:
                    body_type = standard_name
                    break
        return body_type

    def __init__(
        self, dir_path='/home/ben/data/alvaform_db',
        filename_pattern = r'^[A-Z]{3}-[A-Z|0-9]{7}-[0-9]{4}((?!Tpose).)*\.obj$',
        verbose = True
    ):
        '''
        @Params
            - dir_path         (optional) : The root directory where contains all the .obj files
            - filename_pattern (optional) : The regular expression pattern that distinguish a valid .obj file
        '''
        self.dir_path = dir_path
        self.filename_pattern = filename_pattern
        self.verbose = verbose
        if self.verbose:
            print('Mesh Generator Initiated.')
            print(' * Data home is set to {}'.format(dir_path))

    def load_all_meshes(self, clss, parallel=False, require_path=False, expected_vtxs_shape=None):
        '''
        Load all the meshes from the target directory
        Inputs:
            - clss     (required) : list of required classes.
                - Candidate classes include Men, MenXL, Women, WomenXL, Boy, Girl, Infant
            - parallel (optional) : if set to True, will load the meshes in parrallel, greatly saves the loading time
                - requrie joblib library for parrallel processing. pip install joblib if you don't
            - require_path(optional) : if set to True, will only return the filenames of eligeble mesh.
            - expected_vtxs_shape : if given, will check the shape of input mesh and raise error if not satisfied
        Outputs:
            - vtx_buffer: list of vertices of the mesh, each in shape (7w, 3).
            - f_buffer: list of faces
            - vbid_buffer: list of VBIDs
        '''
        cls_list = self.cls_list

        dir_path = self.dir_path
        pattern = self.filename_pattern

        if not all(cls in cls_list for cls in clss):
            raise ValueError('The class name has to be one of %s.' % ', '.join(cls_list))

        criterion = lambda x:self.find_body_type(self.get_vbid(x)) in clss
        file_names = list(load_all_children(dir_path, pattern, criterion))

        vbid_buffer = []
        vtx_buffer = []
        valid_file_names = []
        _, f_buffer = self.load_mesh(file_names[0], want_faces=True)

        if self.verbose:
            print(' * Start loading files of {} from {}'.format(', '.join(clss), os.path.split(dir_path)[1]))
            print()
        if not parallel:
            # for file_name in tqdm(file_names):

            filename_iter = ProgressBarWrapper(file_names, 'BLUE') if self.verbose else filenames
            for file_name in filename_iter:
                vertices = self.load_vertices(file_name)
                if expected_vtxs_shape is not None and vertices.shape != expected_vtxs_shape:
                    continue
                vtx_buffer.append(vertices)
                vbid = self.get_vbid(file_name)
                vbid_buffer.append(vbid)
                valid_file_names.append(file_name)

            vtx_buffer = np.asarray(vtx_buffer, dtype=np.float64)
            f_buffer = np.asarray(f_buffer, dtype=np.int64)
            vbid_buffer = np.asarray(vbid_buffer)
            file_names = np.asarray(valid_file_names)

            if self.verbose:
                print() # add a line break so that the progress bar looks better

            if require_path:
                return vtx_buffer, f_buffer, vbid_buffer, file_names
            return vtx_buffer, f_buffer, vbid_buffer
        else:
            raise NotImplementedError("Parallel processing is not available now.")
            for file_name in file_names:
                vbid = self.get_vbid(file_name)
                vbid_buffer.append(vbid)
            return f_buffer, vbid_buffer, file_names
        #------------------------------------------COPY-TO-YOUR-NOTEBOOK-----------------------------------------------#
        # Copy the following code to the notebook where you want to load meshes in parallel
        # import numpy, multiprocessing; from joblib import Parallel, delayed
        # def load_mesh_in_main(file_name):
        #     vertices = []
        #     with open(file_name, 'rb') as f:
        #         for line in f.readlines():
        #             if len(line) > 0 and line.startswith('v'):
        #                 vertices.append([float(v) for v in line.split()[1:4]])
        #     f.close()
        #     return numpy.array(vertices)
        # def load_all_meshes_in_main(f_buffer, vbid_buffer, file_names):
        #     cpu_cores = multiprocessing.cpu_count()
        #     vtx_buffer = Parallel(n_jobs=cpu_cores)(delayed(load_mesh_in_main)(i) for i in file_names)
        #     return vtx_buffer, f_buffer, vbid_buffer
        #------------------------------------------COPY-TO-YOUR-NOTEBOOK-----------------------------------------------#

    def dataset_status(self, dir_path=None, require_filenames=False):
        '''
        Get the number of meshes of each type in the target directory.
        Input:
            - dir_path: the designated directory. if not specified, will use the class level dir_path.
        '''
        if dir_path == None:
            dir_path = self.dir_path
        pattern = self.filename_pattern

        body_types = []

        file_names = list(load_all_children(dir_path, pattern))
        for file_name in file_names:
            vbid = self.get_vbid(file_name)
            body_type = self.find_body_type(vbid)
            body_types.append(body_type)

        print('\nDataset Statistics:')
        for k, v in Counter(body_types).items():
            print('  - {}: {}'.format(k, v))

        if require_filenames:
            return file_names

    def save_vtxs_as_obj(self, vtxs, vbid, descripor='', output_dir='output_obj'):
        '''
        Save the mesh vertices as a .obj file
        Inputs:
            - vtxs: the mesh vertices to be saved
            - vbid: the faces and normals will be from the .obj with the designated vbid
            - descrpitor: additional str to be added to the .obj file.
                - E.g. decriptor = 'tall_man_'
            - output_dir: where the generated .obj will be saved.
        Outputs:

        '''
        dir_path = self.dir_path
        pattern = self.filename_pattern
        file_names = load_all_children(dir_path, pattern)
        for file_name in file_names:
            aVbid = self.get_vbid(file_name)
            if aVbid == vbid:
                output_filename = '{}_{}with_hands.obj'.format(get_vbid(file_name), descripor)
                output_path = os.path.join(os.getcwd(), output_dir, output_filename)
                self.save_mesh(file_name, output_path, vtxs)
                return output_path
        raise ValueError('VBID: {} not found under directory {}'.format(vbid, dir_path))
