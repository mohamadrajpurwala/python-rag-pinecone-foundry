from faster_whisper import WhisperModel

def audio(filePath:str):
    model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
    )

    segments, info = model.transcribe(filePath)

    transcript_segments = []

    for segment in segments:
        transcript_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    for segment in transcript_segments:
        print(
            f"[{segment['start']:.2f} - {segment['end']:.2f}] "
            f"{segment['text']}"
        )

audio("sample.m4a")