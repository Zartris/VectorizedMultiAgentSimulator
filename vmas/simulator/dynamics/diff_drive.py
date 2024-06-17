#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import copy
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
        dyn_limits: dict,
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
        self.dyn_limits = copy.deepcopy(dyn_limits)

    def reset(self, index: Union[Tensor, int] = None):
        pass

    def euler(self, f, rot):
        return f(rot)

    # def runge_kutta_force(self, f, rot):
    #     k1 = f(rot)
    #     k2 = f(rot + self.dt * k1[2] / 2)
    #     k3 = f(rot + self.dt * k2[2] / 2)
    #     k4 = f(rot + self.dt * k3[2])

    #     return (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    # def runge_kutta_position(self, local_vel, angular_vel, sub_dt):
    #     local_vel_x = local_vel[:, X].unsqueeze(-1)
    #     local_vel_y = local_vel[:, Y].unsqueeze(-1)
    #     current_ang_vel = angular_vel.unsqueeze(-1)

    #     def f(rot):
    #         return torch.cat(
    #             [
    #                 sub_dt
    #                 * (local_vel_x * torch.cos(rot) - local_vel_y * torch.sin(rot)),
    #                 sub_dt
    #                 * (local_vel_x * torch.sin(rot) + local_vel_y * torch.cos(rot)),
    #                 sub_dt * current_ang_vel,
    #             ],
    #             dim=-1,
    #         )

    #     current_heading = self.agent.state.rot
    #     k1 = f(current_heading)
    #     k2 = f(current_heading + k1[:, 2].unsqueeze(-1) / 2)
    #     k3 = f(current_heading + k2[:, 2].unsqueeze(-1) / 2)
    #     k4 = f(current_heading + k3[:, 2].unsqueeze(-1))

    #     return self.agent.state.pos + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)[:, :2]

    def clamp_linear_accel(self, linear_accel):
        return torch.clamp(
            linear_accel,
            min=-self.dyn_limits.max_acc_meter_per_sec2,
            max=self.dyn_limits.max_acc_meter_per_sec2,
        )

    def clamp_linear_vel(self, linear_vel):
        return torch.clamp(
            linear_vel,
            min=self.dyn_limits.min_vel_meter_per_sec,
            max=self.dyn_limits.max_vel_meter_per_sec,
        )

    def clamp_rot_accel(self, rot_accel):
        return torch.clamp(
            rot_accel,
            min=-self.dyn_limits.max_rot_acc_rad_per_sec2,
            max=self.dyn_limits.max_rot_acc_rad_per_sec2,
        )

    def update_linear_control(self, current_velocity, linear_accel, dt):
        # limit acceleration
        valid_acc = self.clamp_linear_accel(linear_accel)
        # then compute new velocity
        vel = current_velocity + valid_acc * dt
        valid_new_vel = self.clamp_linear_vel(vel)
        return valid_new_vel

    def update_angular_control(self, current_orientation, control_input, dt):
        valid_angular_vel = self.clamp_rot_accel(control_input)
        return current_orientation + valid_angular_vel * dt

    def runge_kutta_position(self, u, dt):
        """
        Perform a single integration step using the fourth-order Runge-Kutta (RK4) method
        to estimate the next position and velocity of a differential drive robot based on
        its current state and control inputs.

        Parameters:
        - x (torch.Tensor): The current state tensor of the robot, with shape [4, batch_size].
        The state vector for each element in the batch includes:
        [x position, y position, orientation (theta), linear velocity].
        - u (torch.Tensor): The control input tensor for the robot, with shape [2, batch_size].
        The control vector for each element in the batch includes:
        [linear acceleration, angular velocity].

        The function computes the new state by applying the control inputs over the duration
        of a timestep, which is determined by the attribute `self.dt`.

        Returns:
        - torch.Tensor: The next state tensor of the robot, with the same shape as `x`.
        Each state vector includes the updated [x position, y position, orientation (theta),
        linear velocity].

        Notes:
        - The `derivatives` function inside `runge_kutta_position` calculates the change rates
        for the state variables.
        - `update_linear_control` and `update_angular_control` methods (assumed to be defined
        elsewhere within the class) are used to update the linear velocity and orientation
        based on the control inputs and the current state of the robot.
        - This method assumes that the tensors are handled by PyTorch and that operations
        on tensors are performed in a vectorized form across a batch of data points.

        Example:
        ```
        # Assuming `robot` is an instance of the class containing this method,
        # `current_state` is a tensor describing the initial conditions of the robots,
        # and `control_inputs` is a tensor of control inputs for each robot.
        next_state = robot.runge_kutta_position(current_state, control_inputs)
        ```
        """

        def derivatives(state, control):
            # Unpack the state and control for clarity
            _, _, orientation, velocity = state
            linear_control, angular_control = control

            # Update velocities and orientations based on control types
            new_velocity = self.update_linear_control(
                velocity, linear_control, dt
            )  # since this velocity is a scaled version we cannot just do this
            new_orientation = self.update_angular_control(
                orientation, angular_control, dt
            )

            # Compute the derivatives
            dx = new_velocity * torch.cos(new_orientation)
            dy = new_velocity * torch.sin(new_orientation)
            d_orientation = new_orientation - orientation  # change in orientation
            d_velocity = new_velocity - velocity  # change in velocity

            return torch.stack([dx, dy, d_orientation, d_velocity])

        def rk4_step(state, control, dt):
            # This inner function performs a single RK4 step.
            k1 = derivatives(state, control)
            k2 = derivatives(state + 0.5 * dt * k1, control)
            k3 = derivatives(state + 0.5 * dt * k2, control)
            k4 = derivatives(state + dt * k3, control)
            new_state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            return new_state

        pos = self.agent.state.pos
        orientation = self.agent.state.rot
        velocity = torch.norm(self.agent.state.vel, dim=-1).unsqueeze(-1)
        x = torch.cat([pos, orientation, velocity], dim=-1).permute(1, 0)
        u = u.permute(1, 0)
        # Starting point for RK4 integration
        next_state = rk4_step(x, u, dt)

        # After computing the next state, ensure the velocity does not exceed the limit
        next_velocity = next_state[3]
        next_velocity_clamped = torch.clamp(
            next_velocity,
            min=self.agent.min_speed,
            max=self.agent.max_speed,
        )
        # Replace the entire next_state tensor, instead of modifying it in-place
        next_state = torch.cat(
            (next_state[:3], next_velocity_clamped.unsqueeze(0)), 0
        ).permute(1, 0)
        # print("next_state", next_state)
        # print("action", u)
        return next_state

    def integrate_state(
        self,
        force: Tensor,
        torque: Tensor,
        agent_index: int,
        substep_index: int,
        sub_dt: float,
        world_drag: float,
        x_semidim: float,
        y_semidim: float,
    ):
        if self.agent.rotatable:
            # Compute rotation
            if substep_index == 0:
                if self.agent.drag is not None:
                    self.agent.set_ang_vel(
                        self.agent.state.ang_vel * (1 - self.agent.drag), None
                    )
                else:
                    self.agent.set_ang_vel(
                        self.agent.state.ang_vel * (1 - world_drag), None
                    )

            self.agent.set_ang_vel(
                self.agent.state.ang_vel
                + (torque[:, agent_index] / self.agent.moment_of_inertia) * sub_dt,
                None,
            )

        if self.agent.movable:
            # Compute translation
            if substep_index == 0:
                if self.agent.drag is not None:
                    self.agent.set_vel(
                        self.agent.state.vel * (1 - self.agent.drag), None
                    )
                else:
                    self.agent.set_vel(self.agent.state.vel * (1 - world_drag), None)
            # TODO:: Might need a more precise method for velocity transfer
            # correct velocity to be in the direction of the agent
            # local_vel = self.global_to_local(self.agent.state.vel)
            local_vel = torch.norm(self.agent.state.vel, dim=1).unsqueeze(-1)
            vel_y = torch.zeros_like(local_vel)
            local_vel = torch.cat([local_vel, vel_y], dim=1)
            # # vel_norm = self.agent.state.vel.norm(dim=1)
            # # if x is negative, we are moving backwards
            # local_vel[:, X] = local_vel
            # local_vel[:, Y] = 0

            # Note the added local velocity y is coming from noise and friction
            # local_force = self.global_to_local(force[:, agent_index])
            local_force = force[:, agent_index]  # we saved the local force

            accel = local_force / self.agent.mass
            local_vel += accel * sub_dt
            # if substep_index == 0:
            #     print("accel", torch.sign(accel[0, 0]) * torch.norm(accel[0]))

            if self.agent.max_speed is not None:
                local_vel = torch.clamp(
                    local_vel, self.agent.min_speed, self.agent.max_speed
                )
                # local_vel = TorchUtils.clamp_with_norm(local_vel, self.agent.max_speed)

            if self.agent.v_range is not None:
                # before = local_vel.clone()
                local_vel = local_vel.clamp(-self.agent.v_range, self.agent.v_range)
                # if not before.equal(local_vel):
                #     debug = 0

            # self.local_to_global(local_vel, out=self.agent.state.vel)

            # Use runge kutta to compute the position from lin accel and angular vel
            control_action = torch.cat(
                [accel[:, None, X], self.agent.state.ang_vel], dim=-1
            )
            new_state = self.runge_kutta_position(control_action, sub_dt)
            new_pos = new_state[:, :2]
            new_rot = new_state[:, None, 2]
            new_local_vel = torch.stack(
                [new_state[:, 3], torch.zeros_like(new_state[:, 3])], dim=-1
            )

            self.agent.set_rot(new_rot, None)
            self.local_to_global(new_local_vel, out=self.agent.state.vel)

            # new_pos_t = self.agent.state.pos + self.agent.state.vel * sub_dt
            if x_semidim is not None:
                new_pos[:, X] = torch.clamp(new_pos[:, X], -x_semidim, x_semidim)
            if y_semidim is not None:
                new_pos[:, Y] = torch.clamp(new_pos[:, Y], -y_semidim, y_semidim)
            self.agent.set_pos(new_pos, None)
            # if substep_index == 0:
            #     print("new_pos", new_pos[0])

    def apply_action_force(self, force: Tensor, index: int, substep: int):
        if self.agent.movable:
            if substep == 0:
                # Update the action so visualizations can use it
                self.agent.action.u_local = self.agent.action.u.clone()

            noise = (
                torch.randn(
                    *self.agent.action.u.shape,
                    device=self.world.device,
                    dtype=torch.float32,
                )
                * self.agent.u_noise
                if self.agent.u_noise is not None
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

            force[:, index] += self.agent.action.u_local  # self.agent.action.u
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
