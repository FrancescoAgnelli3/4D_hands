from .vector_fields import *
from .GCDE_class import *

def make_model(input_dim, hid_dim, hid_hid_dim, num_layers, num_nodes, cheb_k, embed_dim, g_type, output_dim, solver): #, horizon):
    vector_field_f = FinalTanh_f(input_channels= input_dim, hidden_channels= hid_dim,
                                    hidden_hidden_channels= hid_hid_dim,
                                    num_hidden_layers= num_layers)
    vector_field_g = VectorField_g(input_channels= input_dim, hidden_channels= hid_dim,
                                    hidden_hidden_channels= hid_hid_dim,
                                    num_hidden_layers= num_layers, num_nodes= num_nodes, cheb_k= cheb_k, embed_dim= embed_dim,
                                    g_type= g_type)
    model = NeuralGCDE(func_f=vector_field_f, func_g=vector_field_g, num_layers=num_layers, input_channels= input_dim, hidden_channels= hid_dim,
                                    output_channels= output_dim, num_nodes = num_nodes, embed_dim = embed_dim,
                                    atol=1e-9, rtol=1e-7, solver= solver) #, horizon= horizon)
    return model