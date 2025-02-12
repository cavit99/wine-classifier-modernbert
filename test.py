import matplotlib.pyplot as plt
from torch.optim import AdamW, Optimizer
import torch
from torch.optim.lr_scheduler import LambdaLR  # Added import here
#from wsd_scheduler import WSDLearningRateScheduler  # Import your scheduler
from typing import List

class WSDLearningRateScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_ratio, decay_ratio, min_lr_ratio, total_steps, last_epoch=-1):
        self.warmup_ratio = warmup_ratio
        self.decay_ratio = decay_ratio
        self.min_lr_ratio = min_lr_ratio
        self.total_steps = total_steps
        self.peak_lr = None
        self.min_lr = None

        self.warmup_steps = int(self.warmup_ratio * self.total_steps)
        self.decay_steps = int(self.decay_ratio * self.total_steps)
        self.stable_steps = self.total_steps - self.warmup_steps - self.decay_steps

        if self.stable_steps < 0:
            raise ValueError("Total steps are too small for the given warmup and decay ratios.")

        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, current_step):
        if self.peak_lr is None:  # Initialize peak_lr and min_lr here
            self.peak_lr = [group['lr'] for group in self.optimizer.param_groups]
            self.min_lr = [lr * self.min_lr_ratio for lr in self.peak_lr]
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        elif current_step < self.warmup_steps + self.stable_steps:
            return 1.0
        elif current_step < self.warmup_steps + self.stable_steps + self.decay_steps:
            decay_step = current_step - self.warmup_steps - self.stable_steps
            decay_step = min(decay_step, self.decay_steps - 1)  # Clamp, as in PPLX
            return [(min_lr * peak_lr) / (decay_step / self.decay_steps * (peak_lr - min_lr) + min_lr) / peak_lr for peak_lr, min_lr in zip(self.peak_lr, self.min_lr)]
        else:
            return [min_lr / peak_lr for peak_lr, min_lr in zip(self.peak_lr, self.min_lr)]


def simulate_scheduler(
    optimizer: Optimizer,
    warmup_ratio: float,
    decay_ratio: float,
    min_lr_ratio: float,
    total_steps: int,
):
    """Simulates and visualizes the WSD learning rate scheduler."""

    scheduler = WSDLearningRateScheduler(
        optimizer, warmup_ratio, decay_ratio, min_lr_ratio, total_steps
    )

    learning_rates = []
    for step in range(total_steps):
        optimizer.step()  # Simulate the optimizer step (important for get_lr())
        scheduler.step()
        learning_rates.append(scheduler.get_last_lr())

    # Print key values
    print(f"Warmup Steps: {scheduler.warmup_steps}")
    print(f"Stable Steps: {scheduler.stable_steps}")
    print(f"Decay Steps: {scheduler.decay_steps}")
    print(f"Peak LR: {scheduler.peak_lr}")
    print(f"Min LR: {scheduler.min_lr}")

    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(len(optimizer.param_groups)):
        plt.plot(
            range(total_steps),
            [lr[i] for lr in learning_rates],
            label=f"Param Group {i}",
        )
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title("WSD Learning Rate Schedule")
    plt.grid(True)
    plt.legend()
    plt.show()

    return learning_rates


def test_scheduler():
    """Runs various test cases for the scheduler."""

    # --- Test Case 1: Standard Case ---
    print("-" * 30, "\nTest Case 1: Standard Case")
    optimizer = AdamW([{"params": [torch.randn(10)], "lr": 1e-3}])
    simulate_scheduler(
        optimizer=optimizer,
        warmup_ratio=0.1,
        decay_ratio=0.2,
        min_lr_ratio=0.1,
        total_steps=1000,
    )

    # --- Test Case 2: Multiple Parameter Groups ---
    print("-" * 30, "\nTest Case 2: Multiple Parameter Groups")
    optimizer = AdamW(
        [
            {"params": [torch.randn(10)], "lr": 1e-3},
            {"params": [torch.randn(5)], "lr": 5e-4},
        ]
    )
    simulate_scheduler(
        optimizer=optimizer,
        warmup_ratio=0.05,
        decay_ratio=0.15,
        min_lr_ratio=0.05,
        total_steps=2000,
    )

    # --- Test Case 3: No Warmup ---
    print("-" * 30, "\nTest Case 3: No Warmup")
    optimizer = AdamW([{"params": [torch.randn(10)], "lr": 2e-4}])
    simulate_scheduler(
        optimizer=optimizer,
        warmup_ratio=0.0,
        decay_ratio=0.3,
        min_lr_ratio=0.2,
        total_steps=500,
    )

    # --- Test Case 4: No Decay ---
    print("-" * 30, "\nTest Case 4: No Decay")
    optimizer = AdamW([{"params": [torch.randn(10)], "lr": 5e-5}])
    simulate_scheduler(
        optimizer=optimizer,
        warmup_ratio=0.2,
        decay_ratio=0.0,
        min_lr_ratio=0.1,  # min_lr_ratio won't have an effect here
        total_steps=800,
    )

    # --- Test Case 5: Short Total Steps (Edge Case) ---
    print("-" * 30, "\nTest Case 5: Short Total Steps")
    optimizer = AdamW([{"params": [torch.randn(10)], "lr": 1e-4}])
    try:
        simulate_scheduler(
            optimizer=optimizer,
            warmup_ratio=0.5,
            decay_ratio=0.6,
            min_lr_ratio=0.1,
            total_steps=100,  # This will cause stable_steps to be negative
        )
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")

    # --- Test Case 6: Only Warmup and Decay (No Stable Phase) ---
    print("-" * 30, "\nTest Case 6: Only Warmup and Decay")
    optimizer = AdamW([{"params": [torch.randn(10)], "lr": 1e-4}])
    simulate_scheduler(
        optimizer=optimizer,
        warmup_ratio=0.4,
        decay_ratio=0.6,
        min_lr_ratio=0.1,
        total_steps=500,
    )
    # --- Test Case 7: Very Small Peak LR ---
    print("-" * 30, "\nTest Case 7: Very Small Peak LR")
    optimizer = AdamW([{"params": [torch.randn(10)], "lr": 1e-7}])  # Tiny LR
    simulate_scheduler(
        optimizer=optimizer,
        warmup_ratio=0.1,
        decay_ratio=0.2,
        min_lr_ratio=0.1,
        total_steps=1000,
    )

    # --- Test Case 8: Zero Decay Ratio ---
    print("-" * 30, "\nTest Case 8: Zero Decay Ratio")
    optimizer = AdamW([{"params": [torch.randn(10)], "lr": 1e-4}])
    simulate_scheduler(
        optimizer=optimizer,
        warmup_ratio=0.1,
        decay_ratio=0.0,  # Zero decay
        min_lr_ratio=0.1,
        total_steps=1000,
    )

if __name__ == "__main__":
    test_scheduler()