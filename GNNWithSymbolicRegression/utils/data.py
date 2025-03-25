

import argparse
from jax import lax
from jax import grad
import warnings
warnings.simplefilter('ignore')
from jax import random
import optax
from jax_md import energy, space, simulate, quantity

import matplotlib
import matplotlib.pyplot as plt

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
from model import *


def build_dataset(POS, Energy,totalsize,train_size):
  noTotal = POS.shape[0]

  II = onp.random.permutation(range(noTotal))
  all_position = POS[II]
  all_energies = Energy[II]
  #all_forces = Force[II]
  noTotal = totalsize
  noTr = int(noTotal * train_size)
  print(noTr)
  train_data = all_position[:noTr]
  test_data = all_position[noTr:noTotal]

  train_energies = all_energies[:noTr]
  test_energies = all_energies[noTr:noTotal]

  #train_forces = all_forces[:noTr]
  #test_forces = all_forces[noTr:]

  return ((train_data, train_energies),
          (test_data, test_energies))