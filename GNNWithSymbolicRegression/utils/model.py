from typing import Callable, Tuple, Dict, Any, Optional
import numpy as onp
import jax
from jax import vmap, jit
import jax.numpy as np
from jax_md import space, dataclasses, quantity, partition, smap, util
import haiku as hk
from collections import namedtuple
from functools import partial, reduce
from utils import *
from utils import _canonicalize_node_state, _apply_node_fn, _apply_edge_fn, _apply_global_fn

# Typing

Array = util.Array
PyTree = Any
Box = space.Box
f32 = util.f32
f64 = util.f64

InitFn = Callable[..., Array]
CallFn = Callable[..., Array]

DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList



class GraphNetwork:
  def __init__(self,
               edge_fn: Callable[[Array], Array],
               node_fn: Callable[[Array], Array],
               global_fn: Callable[[Array], Array]):
    self._node_fn = (None if node_fn is None else
                     partial(_apply_node_fn, node_fn=vmap(node_fn)))          #node function = global function #node_fn=vmap(node_fn)

    self._edge_fn = (None if edge_fn is None else
                     partial(_apply_edge_fn, edge_fn=vmap(vmap(edge_fn))))

    self._global_fn = (None if global_fn is None else
                       partial(_apply_global_fn, global_fn=global_fn))

  def __call__(self, graph: my_GraphTuple) -> my_GraphTuple:
    if self._edge_fn is not None:
      graph = dataclasses.replace(graph, edges=self._edge_fn(graph))
    if self._node_fn is not None:
      graph = dataclasses.replace(graph, nodes=self._node_fn(graph))
    if self._global_fn is not None:
      graph = dataclasses.replace(graph, globals=self._global_fn(graph))

    return graph


class my_GraphNetEncoder(hk.Module):
  def __init__(self,
               n_recurrences: int,
               edge_mlp_sizes: Tuple[int, ...],
               node_mlp_sizes: Tuple[int, ...],
               global_mlp_sizes: Tuple[int, ...],
               mlp_kwargs: Optional[Dict[str, Any]]=None,
               name: str='my_GraphNetEncoder'):
    super(my_GraphNetEncoder, self).__init__(name=name)

    if mlp_kwargs is None:
      mlp_kwargs = {}

    self._n_recurrences = n_recurrences

    node_embedding_fn = lambda name: lambda x: x

    edge_embedding_fn = lambda name: hk.nets.MLP(
        output_sizes=edge_mlp_sizes,
        w_init=hk.initializers.VarianceScaling(1.0),
        b_init=hk.initializers.VarianceScaling(1.0),
        activate_final=True,
        name=name,
        **mlp_kwargs)

    global_embedding_fn = lambda name: lambda x: x

    node_model_fn = lambda name: lambda x: x

    edge_model_fn = lambda name: lambda *args: hk.nets.MLP(
        output_sizes=edge_mlp_sizes + (1,),
        w_init=hk.initializers.VarianceScaling(1.0),
        b_init=hk.initializers.VarianceScaling(1.0),
        activate_final=False,
        name=name,
        **mlp_kwargs)(np.concatenate(args, axis=-1),dropout_rate = 0,rng=20)

    global_model_fn = lambda name: lambda x: x

    self._encoder = GraphIndependent(
        edge_embedding_fn('EdgeEncoder'),
        node_embedding_fn('NodeEncoder'),
        global_embedding_fn('GlobalEncoder'))
    self._propagation_network = lambda: GraphNetwork(
        edge_model_fn('EdgeFunction'),
        node_model_fn('NodeFunction'),
        global_model_fn('GlobalFunction')
        )

  def __call__(self, graph: my_GraphTuple) -> my_GraphTuple:
    encoded = self._encoder(graph)
    outputs = encoded

    for _ in range(self._n_recurrences):
      inputs = concatenate_graph_features((outputs, encoded))
      outputs = self._propagation_network()(inputs)

    return outputs, outputs.edges



class EnergyGraphNet(hk.Module):
  def __init__(self,
               n_recurrences: int,
               edge_mlp_sizes: Tuple[int, ...],
               node_mlp_sizes: Tuple[int, ...],
               global_mlp_sizes: Tuple[int, ...],
               mlp_kwargs: Optional[Dict[str, Any]]=None,
               name: str='Energy'):
    super(EnergyGraphNet, self).__init__(name=name)

    if mlp_kwargs is None:
      mlp_kwargs = {
        #'w_init': hk.initializers.VarianceScaling(1.0),
        #'b_init': hk.initializers.VarianceScaling(1.0),
        #activation': jax.nn.softplus
        }

    self._graph_net = my_GraphNetEncoder(n_recurrences,edge_mlp_sizes, node_mlp_sizes, global_mlp_sizes, mlp_kwargs)

  def __call__(self, graph: my_GraphTuple) -> np.ndarray:
    output, edges_feature = self._graph_net(graph)
    return 1/2 * np.sum(output.edges), edges_feature


def graph_network_myself(
                  nodes: Optional[Array]=None,
                  n_recurrences: int=1,                               
                  edge_mlp_sizes: Tuple[int, ...]= None,                   
                  node_mlp_sizes: Tuple[int, ...]= None,
                  global_mlp_sizes: Tuple[int, ...]= None,
                  fractional_coordinates=False,
                  mlp_kwargs: Optional[Dict[str, Any]]=None
                  ) -> Tuple[InitFn,
                             Callable[[PyTree, Array], Array]]:
  nodes = _canonicalize_node_state(nodes)

  @hk.without_apply_rng
  @hk.transform
  def model(R: Array,box_size: Box, r_cutoff: float, **kwargs) -> Array:
    N = R.shape[0]

    #Calculate distance based on PBC
    #Using the PBC function from JAX-MD source code
    latvec = np.array([[box_size, 0., 0.],
                   [0., box_size, 0.],
                   [0., 0., box_size]])
    displacement_fn, shift_fn = space.periodic_general(latvec,fractional_coordinates=fractional_coordinates)
    d = partial(displacement_fn, **kwargs)
    d = space.map_product(d)
    dR = d(R, R) #pairwirse distance, to generate edge feature
    dr_2 = space.square_distance(dR) #pairwise square distance, to check if in cutoff range
    dr = np.sqrt(dr_2)
    dr = np.reshape(dr, dr.shape + (1,))

    if 'nodes' in kwargs:
      _nodes = _canonicalize_node_state(kwargs['nodes'])
    else:
      _nodes = np.zeros((N, 1), R.dtype) if nodes is None else nodes

    edge_idx = np.broadcast_to(np.arange(N)[np.newaxis, :], (N, N))
    edge_idx = np.where(dr_2 < r_cutoff ** 2, edge_idx, N)

    _globals = np.zeros((1,), R.dtype)

    net = EnergyGraphNet(n_recurrences, edge_mlp_sizes, node_mlp_sizes, global_mlp_sizes, mlp_kwargs)
    return net(my_GraphTuple(_nodes, dr, _globals, edge_idx)), dr, edge_idx   # pytype: disable=wrong-arg-count

  return model.init, model.apply














