import os
from pathlib import Path
from typing import List, Tuple, Optional
import torch
import numpy as np
import soundfile as sf
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip


class SFXModelEngine:
    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 48000,
        sfx_dir: str = "generated_sfx",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32  # FP32 as requested
        self.sample_rate = sample_rate

        self.sfx_dir = Path(sfx_dir)
        self.sfx_dir.mkdir(parents=True, exist_ok=True)

        print(f"[SFX] Using device={self.device}, dtype={self.dtype}")
        self.model = self._load_model()

    # -------------------------------------------------
    # TODO: Wire your actual open-source model here
    # -------------------------------------------------
    def _load_model(self):
        """
        Load your open-source SFX model here.

        Example patterns (you fill in from the model's README):

        - HuggingFace pipeline
        - Custom PyTorch checkpoint
        - Diffusers pipeline
        """
        # PSEUDOCODE EXAMPLES (replace with real code):
        #
        # from diffusers import StableAudioPipeline
        # pipe = StableAudioPipeline.from_pretrained(
        #     "your/model-id",
        #     torch_dtype=self.dtype
        # ).to(self.device)
        # return pipe
        #
        # or:
        #
        # model = YourModelClass.from_pretrained("path_or_id")
        # model.to(self.device, dtype=self.dtype)
        # model.eval()
        # return model
        #
        raise NotImplementedError(
            "Implement _load_model() with your chosen open-source SFX model."
        )

    # -------------------------------------------------
    # Core: text → SFX
    # -------------------------------------------------
    def text_to_sfx(
        self,
        prompt: str,
        duration_seconds: float = 3.0,
        out_name: Optional[str] = None,
    ) -> Path:
        """
        Generate a short SFX clip from text using the loaded model.
        Returns a WAV file path.
        """

        if out_name is None:
            safe = "".join(c for c in prompt[:40] if c.isalnum() or c in "-_ ").strip()
            if not safe:
                safe = "sfx"
            out_name = safe.replace(" ", "_")

        out_path = self.sfx_dir / f"{out_name}.wav"

        print(f"[SFX] Generating: '{prompt}' ({duration_seconds:.2f}s) → {out_path}")

        # -------------------------------------------------
        # TODO: Replace this with your model's generate call
        # -------------------------------------------------
        #
        # PSEUDOCODE EXAMPLES:
        #
        # 1) Diffusers-style pipeline:
        # audio = self.model(
        #     prompt=prompt,
        #     audio_end_in_s=duration_seconds,
        #     num_inference_steps=50,
        #     guidance_scale=3.5,
        # ).audios[0]  # shape: (samples,)
        #
        # 2) Custom model returning torch.Tensor:
        # with torch.no_grad():
        #     audio = self.model.generate(
        #         prompt=prompt,
        #         duration=duration_seconds,
        #         sample_rate=self.sample_rate,
        #     )  # shape: (1, samples) or (samples,)
        #
        # Make sure `audio` ends up as a 1D NumPy array at self.sample_rate.
        #
        raise NotImplementedError(
            "Implement the model's text→audio generation inside text_to_sfx()."
        )

        # Example post-processing once you have `audio` as 1D float32 NumPy:
        #
        # audio = np.asarray(audio, dtype=np.float32)
        # max_val = np.max(np.abs(audio))
        # if max_val > 1.0:
        #     audio = audio / max_val
        #
        # sf.write(out_path, audio, self.sample_rate)
        # return out_path

    # -------------------------------------------------
    # Core: video → Foley (using text prompts per cue)
    # -------------------------------------------------
    def video_add_sfx(
        self,
        video_path: str | Path,
        cues: List[Tuple[float, str]],
        output_path: str | Path = "video_with_sfx.mp4",
        sfx_gain_db: float = 0.0,
    ) -> Path:
        """
        Add generated SFX to a video at given timestamps.

        cues: list of (time_in_seconds, text_prompt_for_sfx)
        """

        video_path = Path(video_path)
        output_path = Path(output_path)

        print(f"[FOLEY] Loading video: {video_path}")
        clip = VideoFileClip(str(video_path))

        audio_layers = [clip.audio] if clip.audio is not None else []

        for t, prompt in cues:
            print(f"[FOLEY] Cue at {t:.2f}s → '{prompt}'")
            sfx_file = self.text_to_sfx(prompt, duration_seconds=3.0)
            sfx_clip = AudioFileClip(str(sfx_file)).set_start(t)

            if sfx_gain_db != 0.0:
                factor = 10 ** (sfx_gain_db / 20.0)
                sfx_clip = sfx_clip.volumex(factor)

            audio_layers.append(sfx_clip)

        final_audio = CompositeAudioClip(audio_layers)
        final_clip = clip.set_audio(final_audio)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[FOLEY] Writing output: {output_path}")
        final_clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
        )

        clip.close()
        final_clip.close()
        return output_path


if __name__ == "__main__":
    # This will fail until you implement _load_model() and text_to_sfx()
    engine = SFXModelEngine()
    # Example intended usage:
    # sfx = engine.text_to_sfx("a sharp metallic clang in a cave", 2.0)
    # cues = [(0.5, "whoosh"), (1.2, "metallic clang"), (2.0, "cartoon boing")]
    # engine.video_add_sfx("input.mp4", cues, "output_with_sfx.mp4")
