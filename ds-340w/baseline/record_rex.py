import argparse
import io
import json
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', required=True)
    parser.add_argument('--block', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.state.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(args.font), 21)
    title_font = ImageFont.truetype(str(args.font), 25)
    encoder = subprocess.Popen([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
        '-f', 'image2pipe', '-framerate', '2', '-i', '-',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output),
    ], stdin=subprocess.PIPE)

    started = time.monotonic()
    frame_number = 0
    (args.state / 'ready').touch()

    try:
        with (args.state / 'capture.jsonl').open('x') as log:
            while not (args.state / 'done').exists():
                elapsed = time.monotonic() - started
                if elapsed > 1800:
                    raise TimeoutError('Recording exceeded 30 minutes')

                capture = subprocess.run([
                    'rex', 'capture', '-s', args.session, '-b', args.block,
                    '--format', 'text', '--trim',
                ], check=True, capture_output=True, text=True).stdout
                log.write(json.dumps({'elapsed': elapsed, 'text': capture}) + '\n')
                log.flush()

                frame = Image.new('RGB', (1600, 1000), '#111820')
                draw = ImageDraw.Draw(frame)
                draw.text((40, 24), 'AutomaTikZ 7B | Live Rex terminal capture',
                          font=title_font, fill='#e7edf3')
                draw.text((40, 64), f'Elapsed {int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}',
                          font=font, fill='#8eacbd')

                preview = args.state / 'result.png'
                if preview.exists():
                    with Image.open(preview) as source:
                        diagram = source.convert('RGB')
                        diagram.thumbnail((1480, 470))
                    frame.paste(diagram, ((1600 - diagram.width) // 2, 170))
                    lines = capture.splitlines()[-10:]
                    top = 680
                else:
                    lines = capture.splitlines()[-30:]
                    top = 130

                for index, line in enumerate(lines):
                    draw.text((40, top + index * 27), line,
                              font=font, fill='#e7edf3')

                data = io.BytesIO()
                frame.save(data, format='PNG')
                encoder.stdin.write(data.getvalue())
                if frame_number % 20 == 0:
                    frame.save(args.state / 'latest-frame.png')

                frame_number += 1
                time.sleep(max(0, started + frame_number / 2 - time.monotonic()))
    finally:
        encoder.stdin.close()
        code = encoder.wait()
        if code:
            raise RuntimeError(f'ffmpeg exited with status {code}')

    print(f'Saved {frame_number} frames to {args.output}', flush=True)


if __name__ == '__main__':
    main()
