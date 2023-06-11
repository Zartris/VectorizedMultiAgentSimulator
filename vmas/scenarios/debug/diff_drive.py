#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import random
import typing
from typing import List

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Sphere
from vmas.simulator.dynamics.diff_drive import DiffDriveDynamics
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """
        Differential drive example scenario
        Run this file to try it out

        The first agent has differential drive dynamics.
        You can control its forward input with the LEFT and RIGHT arrows.
        You can control its rotation with N and M.

        The second agent has standard vmas holonomic dynamics.
        You can control it with WASD
        You can control its rotation withQ and E.

        """
        # T
        # self.plot_grid = True
        self.n_agents = kwargs.get("n_agents", 1)

        # Make world
        self.dt = 0.1
        world = World(batch_dim, device, dt=self.dt, substeps=10, drag=0.25)
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent {i}",
                collide=True,
                shape=Sphere(1.),
                render_action=True,
                u_range=1,
                u_rot_range=1,
                u_rot_multiplier=1,
            )
            if i == 0:
                # TODO:: needs better solution
                # The drag needs to be computed in DiffDriveDynamics instead of later
                # So the agents drag is set to 0. and we give the drag to the dynamics
                agent.dynamics = DiffDriveDynamics(agent, world, integration="rk4")
                # After construction of the dynamics the agent drag can be set to 0.
                agent._drag = 1.

            world.add_agent(agent)

        self._init_text()

        return world

    def _init_text(self):
        from vmas.simulator import rendering

        state = rendering.RenderStateSingleton()
        # here the index an be customized to change the position of the text
        offset = len(state.text_lines)

        # I used a list here but you could also add variables:
        self.custom_obs_text_index = 0 + offset
        state.text_lines.append(rendering.TextLine(self.custom_obs_text_index))
        self.custom_rew_text_index = 1 + offset
        state.text_lines.append(rendering.TextLine(self.custom_rew_text_index))

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=0.1,
            x_bounds=(-1, 1),
            y_bounds=(-1, 1),
        )

    def process_action(self, agent: Agent):
        if hasattr(agent, 'dynamics'):
            # agent.dynamics.process_force()

            # assuming input is in forces
            torque = agent.action.u_rot
            angular_vel = torque / agent.moment_of_inertia * self.dt
            agent.action.u_rot = angular_vel
            agent.dynamics.process_force()
            agent.action.u_rot = torque

        # try:
        #     agent.dynamics.process_force()
        # except AttributeError:
        #     pass

    def reward(self, agent: Agent):
        return torch.zeros(self.world.batch_dim)

    def observation(self, agent: Agent):
        observations = [
            agent.state.pos, agent.state.rot,
            agent.state.vel, agent.state.ang_vel,
        ]
        return torch.cat(
            observations,
            dim=-1,
        )

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering
        state = rendering.RenderStateSingleton()

        # Agent rotation
        for agent in self.world.agents:
            color = Color.BLACK.value
            line = rendering.Line(
                (0, 0),
                (agent.shape.circumscribed_radius(), 0),
                width=1,
            )
            xform = rendering.Transform()
            xform.set_rotation(agent.state.rot[env_index])
            xform.set_translation(*agent.state.pos[env_index])
            line.add_attr(xform)
            line.set_color(*color)
            state.onetime_geoms.append(line)

        # DO NOT COMMIT THIS INTO MASTER, IT IS ONLY TO SHOW TEXT RENDER EXAMPLE
        state.text_lines[self.custom_obs_text_index].set_text(f"custom obs text {random.randint(0, 100)}")
        state.text_lines[self.custom_rew_text_index].set_text(f"custom rew text {random.randint(0, 100)}")
        return []


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=False)
