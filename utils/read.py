import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import openmesh as om
from utils.mesh_loading import OBJ

def read_mesh(path):
    mesh = om.read_trimesh(path)
    face = torch.from_numpy(mesh.face_vertex_indices()).T.type(torch.long)
    x = torch.tensor(mesh.points().astype('float32'))
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)
    return Data(x=x, edge_index=edge_index, face=face)

def read_mesh_obj(path):
    mesh = OBJ.load_file(path)
    face = torch.from_numpy(mesh.triangular_faces).T.type(torch.long)
    x = torch.tensor(mesh.vtxs.astype('float32'))
    edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
    edge_index = to_undirected(edge_index)
    return Data(x=x, edge_index=edge_index, face=face)
    
