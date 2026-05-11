import torch
import soundfile as sf
from diffusers import StableAudioPipeline

class StableAudioSFX:
    def __init__(self, device="cuda"):
        self.pipe = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0",
            torch_dtype=torch.float16
        ).to(device)

    def generate(self, prompt, output="output.wav", seconds=4):
        print(f"[StableAudio] Generating: {prompt}")

        audio = self.pipe(
            prompt,
            num_inference_steps=30,
            audio_length_in_s=seconds,
            guidance_scale=3.5
        ).audios[0]

        sf.write(output, audio, 44100)
        print(f"[StableAudio] Saved: {output}")
        return output


if __name__ == "__main__":
    sfx = StableAudioSFX()
    sfx.generate("footsteps on gravel", "footsteps.wav")
