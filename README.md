# MusicGen Inference on Modal

A serverless web endpoint for Facebook's MusicGen Large model running on Modal.

## Setup

1. Install Modal:
```bash
pip install modal
```

2. Authenticate with Modal:
```bash
modal token new
```

## Deployment

Deploy the endpoint to Modal:
```bash
modal deploy app.py
```

For local development/testing:
```bash
modal serve app.py
```

## API Usage

### Generate Music (POST)

**Endpoint:** `https://<your-modal-app>.modal.run/generate_endpoint`

**Request:**
```json
{
    "prompt": "upbeat electronic dance music with heavy bass",
    "duration": 8.0,
    "temperature": 1.0,
    "top_k": 250,
    "top_p": 0.0,
    "cfg_coef": 3.0
}
```

**Parameters:**
- `prompt` (required): Text description of the music to generate
- `duration` (optional): Length in seconds, 1-30 (default: 8.0)
- `temperature` (optional): Sampling temperature, higher = more random (default: 1.0)
- `top_k` (optional): Top-k sampling parameter (default: 250)
- `top_p` (optional): Top-p nucleus sampling parameter (default: 0.0)
- `cfg_coef` (optional): Classifier-free guidance coefficient (default: 3.0)

**Response:**
```json
{
    "audio_base64": "<base64 encoded WAV audio>",
    "sample_rate": 32000,
    "duration": 8.0
}
```

### Health Check (GET)

**Endpoint:** `https://<your-modal-app>.modal.run/health`

**Response:**
```json
{
    "status": "ok",
    "model": "facebook/musicgen-large"
}
```

## Example Usage

### Python
```python
import requests
import base64

response = requests.post(
    "https://<your-modal-app>.modal.run/generate_endpoint",
    json={
        "prompt": "calm acoustic guitar melody",
        "duration": 10.0
    }
)

data = response.json()
audio_bytes = base64.b64decode(data["audio_base64"])

with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

### cURL
```bash
curl -X POST "https://<your-modal-app>.modal.run/generate_endpoint" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "epic orchestral music", "duration": 8}' \
    | jq -r '.audio_base64' | base64 -d > output.wav
```

## Local Testing

Run generation locally via Modal:
```bash
modal run app.py --prompt "jazzy piano solo" --duration 10
```

## Model Information

- **Model:** facebook/musicgen-large (3.3B parameters)
- **GPU:** NVIDIA A10G (24GB VRAM)
- **Sample Rate:** 32kHz
- **Max Duration:** 30 seconds

## Cost Considerations

The endpoint uses an A100 GPU. Modal charges based on GPU-seconds used. The container has a 120-second idle timeout to minimize costs between requests.
