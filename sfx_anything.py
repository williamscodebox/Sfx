import torch
import soundfile as sf
from audiocraft.models import AudioLDM2

class AudioLDM2SFX:
    def __init__(self, device="cuda"):
        self.model = AudioLDM2.get_pretrained("audioldm2-large").to(device)

    def generate(self, prompt, output="output.wav", seconds=4):
        print(f"[AudioLDM2] Generating: {prompt}")

        audio = self.model.generate(
            text=prompt,
            audio_length_in_s=seconds,
            guidance_scale=3.5
        )[0]

        sf.write(output, audio.cpu().numpy(), 44100)
        print(f"[AudioLDM2] Saved: {output}")
        return output


if __name__ == "__main__":
    sfx = AudioLDM2SFX()
    sfx.generate("footsteps on gravel", "footsteps.wav")
