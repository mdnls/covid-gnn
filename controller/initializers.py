import torch
from models.nets import MLPEncoder, SpikeEncoder, MLPDecoder, GraphConvLayer, SoftmaxReluClassifier

def init_optimizer(params, cnf):
    if(cnf.training.algo == "adam"):
        opt = torch.optim.Adam(params,
                               lr=cnf.training.lr,
                               betas=(cnf.training.param.beta1, cnf.training.param.beta2))
    else:
        raise ValueError(f"Unrecognized training algo: {cnf.training.algo}")
    return opt

def init_nets(cnf):
    n = cnf.setting.n_vertices
    Tk = cnf.setting.time_window

    #encoder
    if(cnf.model.encode.arch == "spike"):
        Enc = SpikeEncoder(n_vertices=cnf.setting.n_vertices,
                         t_timesteps=cnf.setting.time_window,
                         state_dim=cnf.model.encode.state_dim)
    elif(cnf.model.encode.arch.endswith("-fcn")):
        n_layers = int(cnf.model.encode.arch.split("-fcn")[0])
        dh = cnf.model.encode.state_dim
        Enc = MLPDecoder(n_vertices=n, t_timesteps=Tk, layers=(n_layers+1)*[dh], state_dim=dh)
    else:
        raise ValueError(f"Unrecognized encoding model: {cnf.model.encode.arch}")

    #propagate
    if(cnf.model.propagate.arch == "graph-conv"):
        Prop = GraphConvLayer(state_dim=cnf.model.propagate.state_dim)
    else:
        raise ValueError(f"Unrecognized propagation model: {cnf.model.propagate.arch}")

    #decoder
    if(cnf.model.decode.arch.endswith("-fcn")):
        n_layers = int(cnf.model.decode.arch.split("-fcn")[0])
        dh = cnf.model.decode.state_dim
        Dec = MLPDecoder(n_vertices=n, t_timesteps=Tk, layers=[Tk*dh] + n_layers*[dh], state_dim=dh)
    else:
        raise ValueError(f"Unrecognized decodng model: {cnf.model.decode.arch}")

    #predict
    if(cnf.model.predict.arch.endswith("-softmax-fcn")):
        n_layers = int(cnf.model.predict.arch.split("-softmax-fcn")[0])
        dh = cnf.model.predict.state_dim
        Pred = SoftmaxReluClassifier(layers=n_layers*[dh] + [3])
    else:
        raise ValueError(f"Unrecognized predict model: {cnf.model.predict.arch}")

    return (Enc, Prop, Dec, Pred)