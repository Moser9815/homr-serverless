# HOMR OMR — RunPod Serverless Endpoint

A serverless GPU endpoint for optical music recognition using [HOMR](https://github.com/liebharc/homr). Takes a photograph of sheet music and returns structured note/rest data as JSON.

## License

This project is licensed under **AGPL-3.0** as required by HOMR's license. See [LICENSE](LICENSE) for details.

HOMR itself is AGPL-3.0 licensed. This wrapper service is published in compliance with that license.

## How It Works

1. Receives a base64-encoded image of sheet music
2. Runs HOMR optical music recognition (UNet segmentation + transformer sequence recognition)
3. Parses the MusicXML output into structured JSON
4. Returns notes, rests, and metadata

## API

### Input
```json
{
    "input": {
        "image": "<base64-encoded image>",
        "clef": "treble",
        "tempo": 120,
        "time_signature": "4/4"
    }
}
```

### Output
```json
{
    "success": true,
    "notes": [
        {
            "pitch": 60,
            "pitch_name": "C4",
            "start_time": 0.0,
            "end_time": 0.5,
            "duration": 0.5,
            "duration_type": "quarter",
            "beat": 1.0,
            "measure": 1,
            "confidence": 0.9
        }
    ],
    "rests": [...],
    "note_count": 42,
    "rest_count": 5,
    "metadata": {
        "tempo": 120,
        "time_signature": "4/4",
        "clef": "treble",
        "detected_key": "G",
        "total_measures": 16,
        "detection_method": "homr",
        "processing_time": 12.3
    },
    "musicxml": "<raw MusicXML output>"
}
```

## Deployment (RunPod Serverless)

### Build
```bash
docker build -t homr-serverless .
```

### Push to Docker Hub
```bash
docker tag homr-serverless moser9815/homr-serverless:latest
docker push moser9815/homr-serverless:latest
```

### RunPod Configuration
- **Container Image**: `moser9815/homr-serverless:latest`
- **GPU**: Any CUDA 12.x compatible GPU
- **Container Disk**: 20 GB
- **Idle Timeout**: 0 (scale to zero)

## Local Testing

```bash
pip install -r requirements.txt
python test_local.py path/to/sheet_music.png --clef treble --tempo 120
```

## Credits

- [HOMR](https://github.com/liebharc/homr) by Christian Liebhardt (AGPL-3.0)
- [oemer](https://github.com/BreezeWhite/oemer) — UNet segmentation models (MIT)
- [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR) — Transformer architecture (Apache-2.0)
