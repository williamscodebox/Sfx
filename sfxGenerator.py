import torch
from diffusers import StableAudioPipeline
from pathlib import Path
import datetime

class SoundEffectGenerator:
    def __init__(self, model_id="stabilityai/stable-audio-open-1.0", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SFX] Loading Stable Audio Open on {self.device}...")

        self.pipe = StableAudioPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        print("[SFX] Model loaded successfully.")

    def generate_sfx(self, prompt: str, duration_seconds: int = 4, output_dir="generated_sfx"):
        """
        Generate a short sound effect from text.
        """

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/sfx_{timestamp}.wav"

        print(f"[SFX] Generating sound effect: '{prompt}' ({duration_seconds}s)")

        audio = self.pipe(
            prompt=prompt,
            num_inference_steps=30,
            audio_end_in_s=duration_seconds,
            guidance_scale=3.5
        ).audios[0]

        self.pipe.save_audio(audio, filename)
        print(f"[SFX] Saved: {filename}")

        return filename


if __name__ == "__main__":
    gen = SoundEffectGenerator()

    # Example prompts
    prompts = [
        "a sharp metallic clang in a cave",
        "footsteps on snow, crisp and isolated",
        "a sci-fi energy pulse charging up",
        "a cartoon boing spring sound",
    ]

    for p in prompts:
        gen.generate_sfx(p, duration_seconds=3)
