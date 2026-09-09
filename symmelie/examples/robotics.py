"""Run: python -m examples.robotics. SE(3) twists and D4 planar features."""
import torch
from symmelie import se, dihedral, EquivariantLinear


def main():
    torch.manual_seed(0)
    algebra, points = se(3)
    twists = algebra.adjoint()  # [angular velocity, linear velocity]
    wrenches = twists.dual()    # [torque, force]
    velocity = torch.randn(8, 6, dtype=torch.float64)
    wrench = torch.randn_like(velocity)
    change_of_frame = torch.tensor([.2, -.1, .3, 1., 2., -.5], dtype=velocity.dtype)
    model = EquivariantLinear(twists, twists)
    error = (model(twists.act(change_of_frame, velocity)) - twists.act(change_of_frame, model(velocity))).abs().max()
    power = (velocity * wrench).sum(-1)
    transformed_power = (twists.act(change_of_frame, velocity) * wrenches.act(change_of_frame, wrench)).sum(-1)
    print(f"SE(3) twist equivariance error: {error.item():.3e}")
    print(f"Twist/wrench power invariance error: {(power-transformed_power).abs().max().item():.3e}")
    point = torch.tensor([1., 0., 0., 1.], dtype=velocity.dtype)
    print(f"Transformed homogeneous point: {points.act(change_of_frame, point).tolist()}")

    _, planar = dihedral(4)  # 90-degree rotations and mirror reflection
    mirror_model = EquivariantLinear(planar, planar)
    x = torch.randn(8, 2, dtype=velocity.dtype)
    error = (mirror_model(planar.act_discrete(1, x)) - planar.act_discrete(1, mirror_model(x))).abs().max()
    print(f"D4 reflection equivariance error: {error.item():.3e}")


if __name__ == "__main__":
    main()
