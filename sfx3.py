import subprocess
from pathlib import Path
import sys

class HunyuanFoley3060:
    def __init__(self, model_dir: str, python_exec: str = sys.executable):
        self.model_dir = Path(model_dir)
        self.python = python_exec

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        print(f"[HunyuanFoley] Using model: {self.model_dir}")
        print(f"[HunyuanFoley] Python: {self.python}")

    def text_to_sfx(self, prompt: str, out_dir="sfx_out"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python,
            str(self.model_dir / "infer.py"),     # FIXED
            "--model_path", str(self.model_dir),  # FIXED
            "--model_size", "xl",
            "--enable_offload",
            "--single_prompt", prompt,
            "--output_dir", str(out_dir)
        ]

        print("[HunyuanFoley] Running text → SFX...")
        subprocess.run(cmd, check=True)
        return out_dir

    def video_to_foley(self, video_path: str, prompt: str, out_dir="foley_out"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python,
            str(self.model_dir / "infer.py"),     # FIXED
            "--model_path", str(self.model_dir),  # FIXED
            "--model_size", "xl",
            "--enable_offload",
            "--single_video", video_path,
            "--single_prompt", prompt,
            "--output_dir", str(out_dir)
        ]

        print("[HunyuanFoley] Running video → Foley...")
        subprocess.run(cmd, check=True)
        return out_dir



if __name__ == "__main__":
    # Example usage:
    foley = HunyuanFoley3060("HunyuanVideo-Foley")

    # Text → SFX
    foley.text_to_sfx("a metallic clang echoing in a cave")

    # Video → Foley
    foley.video_to_foley("input.mp4", "footsteps on gravel")
