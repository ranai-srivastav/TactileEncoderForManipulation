from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import RobotStateOnlyLSTM


def run_case(device: str) -> None:
    batch_size = 2
    timesteps = 4
    frames_per_sec = 1
    ft_dim = 6
    gripper_dim = 2
    robot_dim = 36
    height = width = 224

    model = RobotStateOnlyLSTM(robot_dim=robot_dim, hidden_dim=32, dropout=0.1).to(device)
    model.train()

    tactile = torch.randn(batch_size, timesteps, frames_per_sec, 3, height, width, device=device)
    rgb = torch.randn(batch_size, timesteps, frames_per_sec, 3, height, width, device=device)
    ft = torch.randn(batch_size, timesteps, ft_dim, device=device)
    gripper = torch.randn(batch_size, timesteps, gripper_dim, device=device)
    robot = torch.randn(batch_size, timesteps, robot_dim, device=device, requires_grad=True)
    gripper_force = torch.randn(batch_size, 1, device=device)
    target = torch.randn(batch_size, 1, device=device)

    logits = model(tactile, rgb, ft, gripper, gripper_force, robot=robot)
    assert logits.shape == (batch_size, 1), logits.shape

    loss = torch.nn.functional.mse_loss(logits, target)
    loss.backward()

    assert robot.grad is not None
    grad_norm = robot.grad.norm().item()
    assert grad_norm > 0.0, grad_norm
    print(f"{device}: ok, logits_shape={tuple(logits.shape)}, robot_grad_norm={grad_norm:.6f}")


def main() -> None:
    run_case("cpu")
    if torch.cuda.is_available():
        run_case("cuda")
    else:
        print("cuda: skipped (not available)")


if __name__ == "__main__":
    main()
