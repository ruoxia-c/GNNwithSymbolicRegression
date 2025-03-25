from typing import Callable, Tuple, Dict, Any, Optional
import numpy as onp
import jax
from jax import vmap, jit
import jax.numpy as np
from jax_md import space, dataclasses, quantity, partition, smap, util
import haiku as hk
from collections import namedtuple
from functools import partial, reduce

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

@dataclasses.dataclass
class my_GraphTuple(object):
    nodes: np.ndarray
    edges: np.ndarray
    globals: np.ndarray
    edge_idx: np.ndarray


def concatenate_graph_features(graphs: Tuple[my_GraphTuple, ...]) -> my_GraphTuple:
  return my_GraphTuple(
      nodes=np.concatenate([g.nodes for g in graphs], axis=-1),
      edges=np.concatenate([g.edges for g in graphs], axis=-1),
      globals=np.concatenate([g.globals for g in graphs], axis=-1),
      edge_idx=graphs[0].edge_idx,  # pytype: disable=wrong-keyword-args
  )


def GraphIndependent(edge_fn: Callable[[Array], Array],
                     node_fn: Callable[[Array], Array],
                     global_fn: Callable[[Array], Array],
                     ) -> Callable[[my_GraphTuple], my_GraphTuple]:
  identity = lambda x: x
  _node_fn = vmap(node_fn) if node_fn is not None else identity
  _edge_fn = vmap(vmap(edge_fn)) if edge_fn is not None else identity
  _global_fn = global_fn if global_fn is not None else identity
  def embed_fn(graph):
    graph = dataclasses.replace(graph, nodes=_node_fn(graph.nodes))
    graph = dataclasses.replace(graph, edges=_edge_fn(graph.edges))
    graph = dataclasses.replace(graph, globals=_global_fn(graph.globals))
    return graph
  return embed_fn



def _apply_node_fn(graph: my_GraphTuple,
                   node_fn: Callable[[Array,Array, Array, Array], Array]
                   ) -> Array:
  mask = graph.edge_idx < graph.nodes.shape[0]
  mask = mask[:, :, np.newaxis]

  if graph.edges is not None:
    # TODO: Should we also have outgoing edges?
    flat_edges = np.reshape(graph.edges, (-1, graph.edges.shape[-1]))
    edge_idx = np.reshape(graph.edge_idx, (-1,))
    incoming_edges = jax.ops.segment_sum(
        flat_edges, edge_idx, graph.nodes.shape[0] + 1)[:-1]
    outgoing_edges = np.sum(graph.edges * mask, axis=1)
  else:
    incoming_edges = None
    outgoing_edges = None

  if graph.globals is not None:
    _globals = np.broadcast_to(graph.globals[np.newaxis, :],
                               graph.nodes.shape[:1] + graph.globals.shape)
  else:
    _globals = None

  return node_fn(graph.edges)                


def _apply_edge_fn(graph: my_GraphTuple,
                   edge_fn: Callable[[Array, Array, Array, Array], Array]
                   ) -> Array:
  if graph.nodes is not None:
    incoming_nodes = graph.nodes[graph.edge_idx]
    outgoing_nodes = np.broadcast_to(
        graph.nodes[:, np.newaxis, :],
        graph.edge_idx.shape + graph.nodes.shape[-1:])
  else:
    incoming_nodes = None
    outgoing_nodes = None

  if graph.globals is not None:
    _globals = np.broadcast_to(graph.globals[np.newaxis, np.newaxis, :],
                               graph.edge_idx.shape + graph.globals.shape)
  else:
    _globals = None

  mask = graph.edge_idx < graph.nodes.shape[0]
  mask = mask[:, :, np.newaxis]
  return edge_fn(graph.edges, incoming_nodes, outgoing_nodes) * mask       


def _apply_global_fn(graph: my_GraphTuple,
                     global_fn: Callable[[Array, Array, Array], Array]
                     ) -> Array:
  nodes = None if graph.nodes is None else np.sum(graph.nodes, axis=0)

  if graph.edges is not None:
    mask = graph.edge_idx < graph.nodes.shape[0]
    mask = mask[:, :, np.newaxis]
    edges = np.sum(graph.edges * mask, axis=(0, 1))
  else:
    edges = None

  return global_fn(graph.globals)                       


def _canonicalize_node_state(nodes: Optional[Array]) -> Optional[Array]:
    if nodes is None:
        return nodes

    if nodes.ndim == 1:
        nodes = nodes[:, np.newaxis]

    if nodes.ndim != 2:
        raise ValueError(
        'Nodes must be a [N, node_dim] array. Found {}.'.format(nodes.shape))

    return nodes









