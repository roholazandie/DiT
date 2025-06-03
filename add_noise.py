import cv2
import numpy as np
import os

def get_beta_schedule(beta_schedule: str,
                      beta_start: float,
                      beta_end: float,
                      num_diffusion_timesteps: int,
                      cosine_s: float = 0.008) -> np.ndarray:
    """
    Create beta schedules:
      - "linear":   linear interpolation from beta_start to beta_end
      - "quad":     square of a linear ramp in sqrt space
      - "warmup10": linear warmup over first 10% then constant
      - "warmup50": linear warmup over first 50% then constant
      - "const":    constant = beta_end
      - "jsd":      1/T, 1/(T-1), ..., 1
      - "cosine":   Nichol & Dhariwal cosine schedule
    """
    T = num_diffusion_timesteps

    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)

    elif beta_schedule == "quad":
        betas = (np.linspace(beta_start**0.5,
                              beta_end**0.5,
                              T, dtype=np.float64) ** 2)

    elif beta_schedule == "warmup10":
        warmup_time = int(T * 0.1)
        betas = beta_end * np.ones(T, dtype=np.float64)
        betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)

    elif beta_schedule == "warmup50":
        warmup_time = int(T * 0.5)
        betas = beta_end * np.ones(T, dtype=np.float64)
        betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)

    elif beta_schedule == "const":
        betas = beta_end * np.ones(T, dtype=np.float64)

    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(T, 1, T, dtype=np.float64)

    elif beta_schedule == "cosine":
        steps = np.arange(T + 1, dtype=np.float64)
        t_div = (steps / T + cosine_s) / (1 + cosine_s)
        alphas_cum = np.cos(t_div * np.pi / 2) ** 2
        alphas_cum = alphas_cum / alphas_cum[0]
        betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])

    else:
        raise NotImplementedError(f"Unknown beta schedule: {beta_schedule}")

    assert betas.shape == (T,)
    return betas.astype(np.float32)


def progressive_noise_ddpm_to_video(input_path: str,
                                    output_video_path: str,
                                    num_steps: int = 10,
                                    beta_start: float = 1e-4,
                                    beta_end: float = 0.02,
                                    beta_schedule: str = 'linear',
                                    fps: int = 10):
    """
    Read an image x_0, normalize to [0,1], then for each t=1..T sample
      x_t = sqrt(bar_alpha_t)*x_0 + sqrt(1 - bar_alpha_t)*epsilon,
    where epsilon ~ N(0, I).  Instead of saving PNGs, collect each x_t as a frame
    and write out a single MP4 video.

    Parameters:
    - input_path: Path to the input image (x_0).
    - output_video_path: Path (including filename.mp4) for the output video.
    - num_steps: Total diffusion timesteps T.
    - beta_start: The first beta in the schedule.
    - beta_end: The final beta in the schedule.
    - beta_schedule: One of 'linear','quad','warmup10','warmup50','const','jsd','cosine'.
    - fps: Frames per second for the output video.
    """
    # 1) Load x_0 and normalize to float32 [0,1]
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image at {input_path}")
    img = img.astype(np.float32) / 255.0   # now in [0,1]

    H, W, C = img.shape

    # 2) Build betas, alphas, and bar_alphas
    betas = get_beta_schedule(beta_schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              num_diffusion_timesteps=num_steps)
    alphas = 1.0 - betas                       # alpha_t = 1 - beta_t
    bar_alphas = np.cumprod(alphas, axis=0)    # bar_alpha_t = ∏_{s=1}^t alpha_s

    # 3) Prepare VideoWriter (use 'mp4v' codec for .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (W, H)                      # note: (width, height)
    )
    if not video_writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_video_path}")

    # 4) Generate frames and write to video
    for t in range(1, num_steps + 1):
        # Sample epsilon_t ~ N(0, I), shape = (H, W, C)
        eps = np.random.normal(loc=0.0, scale=1.0, size=(H, W, C)).astype(np.float32)

        # Compute x_t = sqrt(bar_alpha_t)*x_0 + sqrt(1 - bar_alpha_t)*eps
        sqrt_bar_alpha = np.sqrt(bar_alphas[t - 1])
        sqrt_one_minus_bar = np.sqrt(1.0 - bar_alphas[t - 1])
        x_t = sqrt_bar_alpha * img + sqrt_one_minus_bar * eps

        # Clip to [0,1], then convert to uint8 [0,255] for saving as a video frame
        x_t_clipped = np.clip(x_t, 0.0, 1.0)
        frame = (x_t_clipped * 255).astype(np.uint8)

        # Write the frame
        video_writer.write(frame)

        beta_t = betas[t - 1]
        bar_alpha_t = bar_alphas[t - 1]
        print(f"[{beta_schedule:^7}] t={t:02d}/{num_steps} "
              f"β_t={beta_t:.5f}  barα_t={bar_alpha_t:.5f}")

    # 5) Release the VideoWriter
    video_writer.release()
    print(f"Video saved to: {output_video_path}")


if __name__ == "__main__":
    input_image = "cat.jpg"

    # Example: Linear schedule, T = 50, writes an MP4 at 10 fps
    progressive_noise_ddpm_to_video(
        input_path=input_image,
        output_video_path="ddpm_diffusion_linear.mp4",
        num_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule='linear',
        fps=10
    )

    # Example: Cosine schedule, T = 50
    progressive_noise_ddpm_to_video(
        input_path=input_image,
        output_video_path="ddpm_diffusion_cosine.mp4",
        num_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule='cosine',
        fps=10
    )
