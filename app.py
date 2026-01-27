"""
Modal web endpoint for Facebook MusicGen Large inference.

Deploy with: modal deploy app.py
Run locally with: modal serve app.py
"""

import modal
import io
import base64

# Define the Modal app
app = modal.App("musicgen-large-inference")

# Create a container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "ffmpeg",
        "pkg-config",
        "libavformat-dev",
        "libavcodec-dev",
        "libavdevice-dev",
        "libavutil-dev",
        "libswscale-dev",
        "libswresample-dev",
        "libavfilter-dev",
    )
    .pip_install(
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "audiocraft",
        "pydantic>=2.0.0",
    )
)


def download_model():
    """Download and cache the MusicGen model at image build time."""
    from audiocraft.models import MusicGen

    MusicGen.get_pretrained("facebook/musicgen-large")


# Build the image with model pre-downloaded
image = image.run_function(download_model, gpu="A10G")


@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=120,
)
class MusicGenModel:
    """MusicGen Large model class for Modal."""

    @modal.enter()
    def load_model(self):
        """Load model when container starts."""
        import torch
        from audiocraft.models import MusicGen

        self.model = MusicGen.get_pretrained("facebook/musicgen-large")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model loaded on {self.device}")

    @modal.method()
    def generate(
        self,
        prompt: str,
        duration: float = 8.0,
        temperature: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        cfg_coef: float = 3.0,
    ) -> bytes:
        """
        Generate music from a text prompt.

        Args:
            prompt: Text description of the music to generate
            duration: Length of generated audio in seconds (max 30)
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            cfg_coef: Classifier-free guidance coefficient

        Returns:
            WAV audio bytes
        """
        import torch
        import torchaudio

        # Clamp duration to reasonable limits
        duration = max(1.0, min(duration, 30.0))

        # Set generation parameters
        self.model.set_generation_params(
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_coef=cfg_coef,
        )

        # Generate audio
        with torch.inference_mode():
            wav = self.model.generate([prompt])

        # Convert to WAV bytes
        audio_tensor = wav[0].cpu()
        sample_rate = self.model.sample_rate

        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, sample_rate, format="wav")
        buffer.seek(0)

        return buffer.read()

    @modal.web_endpoint(method="POST")
    def generate_endpoint(self, request: dict) -> dict:
        """
        Web endpoint for music generation.

        Request body:
        {
            "prompt": "upbeat electronic dance music with heavy bass",
            "duration": 8.0,
            "temperature": 1.0,
            "top_k": 250,
            "top_p": 0.0,
            "cfg_coef": 3.0
        }

        Returns:
        {
            "audio_base64": "<base64 encoded WAV>",
            "sample_rate": 32000,
            "duration": 8.0
        }
        """
        prompt = request.get("prompt", "")
        if not prompt:
            return {"error": "prompt is required"}

        duration = float(request.get("duration", 8.0))
        temperature = float(request.get("temperature", 1.0))
        top_k = int(request.get("top_k", 250))
        top_p = float(request.get("top_p", 0.0))
        cfg_coef = float(request.get("cfg_coef", 3.0))

        audio_bytes = self.generate(
            prompt=prompt,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            cfg_coef=cfg_coef,
        )

        return {
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "sample_rate": self.model.sample_rate,
            "duration": duration,
        }

    @modal.web_endpoint(method="GET")
    def health(self) -> dict:
        """Health check endpoint."""
        return {"status": "ok", "model": "facebook/musicgen-large"}


@app.local_entrypoint()
def main(prompt: str = "upbeat electronic dance music", duration: float = 8.0):
    """Local entrypoint for testing."""
    model = MusicGenModel()
    audio_bytes = model.generate.remote(prompt=prompt, duration=duration)

    output_path = "output.wav"
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    print(f"Generated audio saved to {output_path}")
