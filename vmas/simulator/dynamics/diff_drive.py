#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import math
from typing import Union

import torch
import typing
from torch import Tensor
import vmas.simulator.core
import vmas.simulator.utils
from vmas.simulator.utils import TorchUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.core import Agent, Entity


class DiffDriveDynamics:
    def __init__(
            self,
            agent: vmas.simulator.core.Agent,
            world: vmas.simulator.core.World,
            integration: str = "rk4",  # one of "euler", "rk4"
    ):
        assert integration == "rk4" or integration == "euler"
        assert (
                agent.action.u_rot_range != 0
        ), "Agent with diff drive dynamics needs non zero u_rot_range"

        self.agent = agent
        self.world = world
        # apply drag to the agent
        self.drag = agent.drag if agent.drag is not None else world.drag
        self.dt = world.dt
        self.integration = integration

    def reset(self, index: Union[Tensor, int] = None):
        pass

    def euler(self, f, rot):
        return f(rot)

    def runge_kutta_force(self, f, rot):
        k1 = f(rot)
        k2 = f(rot + self.dt * k1[2] / 2)
        k3 = f(rot + self.dt * k2[2] / 2)
        k4 = f(rot + self.dt * k3[2])

        return (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def runge_kutta_position(self, local_vel, angular_vel, sub_dt):
        local_vel_x = local_vel[:, X].unsqueeze(-1)
        local_vel_y = local_vel[:, Y].unsqueeze(-1)
        current_ang_vel = angular_vel.unsqueeze(-1)

        def f(rot):
            return torch.cat(
                [sub_dt * (local_vel_x * torch.cos(rot) - local_vel_y * torch.sin(rot)),
                 sub_dt * (local_vel_x * torch.sin(rot) + local_vel_y * torch.cos(rot)),
                 sub_dt * current_ang_vel], dim=-1
            )

        current_heading = self.agent.state.rot
        k1 = f(current_heading)
        k2 = f(current_heading + k1[:, 2].unsqueeze(-1) / 2)
        k3 = f(current_heading + k2[:, 2].unsqueeze(-1) / 2)
        k4 = f(current_heading + k3[:, 2].unsqueeze(-1))

        return self.agent.state.pos + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)[:, :2]

    def integrate_state(self, force: Tensor, torque: Tensor, agent_index: int, substep_index: int, sub_dt: float,
                        world_drag: float,
                        x_semidim: float, y_semidim: float):
        if self.agent.movable:
            # Compute translation
            if substep_index == 0:
                if self.agent.drag is not None:
                    self.agent.state.vel = self.agent.state.vel * (1 - self.agent.drag)
                else:
                    self.agent.state.vel = self.agent.state.vel * (1 - world_drag)
            # TODO:: Might need a more precise method for velocity transfer
            # correct velocity to be in the direction of the agent
            local_vel = self.global_to_local(self.agent.state.vel)
            vel_norm = self.agent.state.vel.norm(dim=1)
            # if x is negative, we are moving backwards
            local_vel[:, X] = torch.sign(local_vel[:, X]) * vel_norm
            local_vel[:, Y] = 0

            # Note the added local velocity y is coming from noise and friction
            local_force = self.global_to_local(force[:, agent_index])
            accel = local_force / self.agent.mass
            local_vel += accel * sub_dt
            self.local_to_global(local_vel, out=self.agent.state.vel)

            if self.agent.max_speed is not None:
                local_vel = TorchUtils.clamp_with_norm(
                    local_vel, self.agent.max_speed
                )

            if self.agent.v_range is not None:
                # before = local_vel.clone()
                local_vel = local_vel.clamp(
                    -self.agent.v_range, self.agent.v_range
                )
                # if not before.equal(local_vel):
                #     debug = 0
            self.local_to_global(local_vel, out=self.agent.state.vel)

            # Use runge kutta to compute the position:
            new_pos = self.runge_kutta_position(local_vel, self.agent.state.ang_vel.squeeze(-1), sub_dt)
            # new_pos_t = self.agent.state.pos + self.agent.state.vel * sub_dt
            if x_semidim is not None:
                new_pos[:, X] = torch.clamp(
                    new_pos[:, X], -x_semidim, x_semidim
                )
            if y_semidim is not None:
                new_pos[:, Y] = torch.clamp(
                    new_pos[:, Y], -y_semidim, y_semidim
                )
            self.agent.state.pos = new_pos

        if self.agent.rotatable:
            # Compute rotation
            if substep_index == 0:
                if self.agent.drag is not None:
                    self.agent.state.ang_vel = self.agent.state.ang_vel * (1 - self.agent.drag)
                else:
                    self.agent.state.ang_vel = self.agent.state.ang_vel * (1 - world_drag)
            self.agent.state.ang_vel += (
                                                torque[:, agent_index] / self.agent.moment_of_inertia
                                        ) * sub_dt
            self.agent.state.rot += self.agent.state.ang_vel * sub_dt

    def apply_action_force(self, force: Tensor, index: int, substep: int):
        if self.agent.movable:
            if substep == 0:
                # Update the action so visualizations can use it
                self.agent.action.u_local = self.agent.action.u.clone()

            noise = (
                torch.randn(
                    *self.agent.action.u.shape, device=self.world.device, dtype=torch.float32
                )
                * self.agent.u_noise
                if self.agent.u_noise
                else 0.0
            )
            self.agent.action.u_local += noise
            if self.agent.max_f is not None:
                self.agent.action.u_local = TorchUtils.clamp_with_norm(
                    self.agent.action.u_local, self.agent.max_f
                )
            if self.agent.f_range is not None:
                self.agent.action.u_local = torch.clamp(
                    self.agent.action.u_local, -self.agent.f_range, self.agent.f_range
                )
            # Convert from local to global:
            self.agent.action.u = self.local_to_global(self.agent.action.u_local)

            force[:, index] += self.agent.action.u
        assert not force.isnan().any()

    def local_to_global(self, u, out=None):
        if out is None:
            out = u.clone()
        sin_angle = torch.sin(self.agent.state.rot.squeeze(-1))
        cos_angle = torch.cos(self.agent.state.rot.squeeze(-1))

        u_x = u[:, vmas.simulator.utils.X]
        u_y = u[:, vmas.simulator.utils.Y]

        # x_global_force = u_x * cos_angle - u_y * sin_angle
        # y_global_force = u_x * sin_angle + u_y * cos_angle

        out[:, vmas.simulator.utils.X].copy_(u_x * cos_angle - u_y * sin_angle)
        out[:, vmas.simulator.utils.Y].copy_(u_x * sin_angle + u_y * cos_angle)
        return out

    def global_to_local(self, u, out=None):
        if out is None:
            out = u.clone()

        sin_angle = torch.sin(self.agent.state.rot.squeeze(-1))
        cos_angle = torch.cos(self.agent.state.rot.squeeze(-1))

        # x_local_force = u[:, vmas.simulator.utils.X] * cos_angle + u[:, vmas.simulator.utils.Y] * sin_angle
        # y_local_force = -u[:, vmas.simulator.utils.X] * sin_angle + u[:, vmas.simulator.utils.Y] * cos_angle

        u_x = u[:, vmas.simulator.utils.X]
        u_y = u[:, vmas.simulator.utils.Y]

        out[:, vmas.simulator.utils.X].copy_(u_x * cos_angle + u_y * sin_angle)
        out[:, vmas.simulator.utils.Y].copy_(-u_x * sin_angle + u_y * cos_angle)

        return out
