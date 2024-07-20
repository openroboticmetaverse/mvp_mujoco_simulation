# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for all models."""

import pathlib
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

# Internal import.


_ROOT_DIR = pathlib.Path(__file__).parent.parent
_MODEL_DIRS = [f for f in _ROOT_DIR.iterdir() if f.is_dir()]
_MODEL_XMLS: List[pathlib.Path] = []
_MJX_MODEL_XMLS: List[pathlib.Path] = []


def _get_xmls(pattern: str) -> List[pathlib.Path]:
  for d in _MODEL_DIRS:
    yield from d.glob(pattern)

_MODEL_XMLS = list(_get_xmls('scene*.xml'))
_MJX_MODEL_XMLS = list(_get_xmls('scene*mjx.xml'))

# Total simulation duration, in seconds.
_MAX_SIM_TIME = 1.0
# Scale for the pseudorandom control noise.
_NOISE_SCALE = 1.0


def _pseudorandom_ctrlnoise(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    i: int,
    noise: float,
) -> None:
  for j in range(model.nu):
    ctrlrange = model.actuator_ctrlrange[j]
    if model.actuator_ctrllimited[j]:
      center = 0.5 * (ctrlrange[1] + ctrlrange[0])
      radius = 0.5 * (ctrlrange[1] - ctrlrange[0])
    else:
      center = 0.0
      radius = 1.0
    data.ctrl[j] = center + radius * noise * (2*mujoco.mju_Halton(i, j+2) - 1)


class ModelsTest(parameterized.TestCase):
  """Tests that MuJoCo models load and do not emit warnings."""

  @parameterized.parameters(_MODEL_XMLS)
  def test_compiles_and_steps(self, xml_path: pathlib.Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    i = 0
    while data.time < _MAX_SIM_TIME:
      _pseudorandom_ctrlnoise(model, data, i, _NOISE_SCALE)
      mujoco.mj_step(model, data)
      i += 1
    # Check no warnings were triggered during the simulation.
    if not all(data.warning.number == 0):
      warning_info = '\n'.join([
          f'{mujoco.mjtWarning(enum_value).name}: count={count}'
          for enum_value, count in enumerate(data.warning.number) if count
      ])
      self.fail(f'MuJoCo warning(s) encountered:\n{warning_info}')


class MjxModelsTest(parameterized.TestCase):
  """Tests that MJX models load and do not return NaNs."""

  @parameterized.parameters(_MJX_MODEL_XMLS)
  def test_compiles_and_steps(self, xml_path: pathlib.Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model = mjx.put_model(model)
    data = mjx.make_data(model)
    ctrlrange = jp.where(
        model.actuator_ctrllimited[:, None],
        model.actuator_ctrlrange,
        jp.array([-10.0, 10.0]),
    )

    def step(x, _):
      data, rng = x
      rng, key = jax.random.split(rng)
      ctrl = jax.random.uniform(
          key,
          shape=(model.nu,),
          minval=ctrlrange[:, 0],
          maxval=ctrlrange[:, 1],
      )
      data = mjx.step(model, data.replace(ctrl=ctrl))
      return (data, rng), ()

    (data, _), _ = jax.lax.scan(
        step,
        (data, jax.random.PRNGKey(0)),
        (),
        length=min(_MAX_SIM_TIME // model.opt.timestep, 100),
    )

    self.assertFalse(jp.isnan(data.qpos).any())


if __name__ == '__main__':
  absltest.main()
