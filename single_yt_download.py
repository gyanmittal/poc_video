#!/usr/bin/env python3
"""Download a single YouTube video by ID into a local folder."""
import os
from yt_dlp import YoutubeDL


def _base_opts(outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    tmpl = os.path.join(outdir, "%(id)s [%(id)s].%(ext)s")
    return {
        "outtmpl": tmpl,
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "postprocessors": [{"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"}],
        "continuedl": True,
        "noprogress": False,
        "retries": 5,
        "socket_timeout": 30,
        "concurrent_fragment_downloads": 4,
        "ignoreerrors": "only_download",
        "ignore_no_formats_error": True,
        "trim_file_name": 120,
        "format_sort": [
            "codec:h264:m4a",
            "res:1080",
            "vcodec:h264",
            "acodec:m4a",
            "ext:mp4:m4a",
        ],
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/127.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    }


def download_video(video_id: str, output_dir: str = "downloads") -> str:
    """Download a single YouTube video by ID into output_dir. Returns saved path."""
    if len(video_id) != 11:
        raise ValueError("video_id must be the 11-character YouTube ID.")
    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = _base_opts(output_dir)
    with YoutubeDL(opts) as ydl:
        ydl.download([url])
        info = ydl.extract_info(url, download=False)
        filename = ydl.prepare_filename(info)
    return filename


if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} <youtube_id> [output_dir]")
        sys.exit(1)

    vid = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) == 3 else "single_downloads"
    out_path = download_video(vid, output_dir=out_dir)
    print(f"Saved to: {out_path}")
