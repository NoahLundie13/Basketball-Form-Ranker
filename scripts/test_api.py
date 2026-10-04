import argparse
import http.client
import json
import os
from pathlib import Path
from urllib.parse import urlencode, urlsplit


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = BASE_DIR / "data" / "input" / "test_videos" / "test1.mp4"
DEFAULT_SERVER = os.environ.get("BASKETBALL_API_URL", "http://127.0.0.1:8000")
CHUNK_BYTES = 1024 * 1024


def upload_video(server_url, video_path, reference):
    parsed_url = urlsplit(server_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("Server URL must start with http:// or https://")

    connection_type = http.client.HTTPSConnection if parsed_url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed_url.hostname, parsed_url.port, timeout=600)
    boundary = f"----BasketballShotTest{os.urandom(12).hex()}"
    filename = video_path.name.replace('"', "_")
    endpoint = f"{parsed_url.path.rstrip('/')}/analyze?{urlencode({'reference': reference})}"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="video"; filename="{filename}"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
    ).encode("utf-8")
    suffix = f"\r\n--{boundary}--\r\n".encode("utf-8")
    content_length = len(prefix) + video_path.stat().st_size + len(suffix)

    try:
        connection.putrequest("POST", endpoint)
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(content_length))
        connection.endheaders()
        connection.send(prefix)
        with video_path.open("rb") as video_file:
            while chunk := video_file.read(CHUNK_BYTES):
                connection.send(chunk)
        connection.send(suffix)

        response = connection.getresponse()
        response_body = response.read()
        if not 200 <= response.status < 300:
            raise RuntimeError(
                f"Server returned HTTP {response.status}: {response_body.decode('utf-8', errors='replace')}"
            )
        return json.loads(response_body)
    finally:
        connection.close()


def print_result(result):
    print(f"Analysis: {result.get('analysis_id', '(no ID returned)')}")
    print(f"Video: {result['video']['filename']}")
    print(f"Reference: {result['reference']['name']}")
    print(f"Overall score: {result['overall_score']:.1f}/100")

    print("\nMetric scores")
    print(f"{'Feature':<30} {'Score':>8}  {'Status':<12}")
    print("-" * 54)
    for metric in result["metrics"]:
        score = "--" if metric["score"] is None else f"{metric['score']:.1f}"
        print(f"{metric['label']:<30} {score:>8}  {metric['status']:<12}")

    print("\nMetric point data (shot values, reference mean, reference spread)")
    for metric in result["metrics"]:
        print(f"\n{metric['key']} ({metric['label']})")
        print(f"  shot_values:    {json.dumps(metric['shot_values'], separators=(',', ':'))}")
        print(f"  reference_mean: {json.dumps(metric['reference_mean'], separators=(',', ':'))}")
        print(f"  reference_std:  {json.dumps(metric['reference_std'], separators=(',', ':'))}")

    print("\nFeedback")
    if result["feedback"]:
        for index, item in enumerate(result["feedback"], start=1):
            print(f"{index}. {item['message']}")
    else:
        print("No notable deviations from the reference were detected.")


def main():
    parser = argparse.ArgumentParser(description="Upload a shot video and print its analysis.")
    parser.add_argument("--url", default=DEFAULT_SERVER, help="API base URL (default: %(default)s)")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Video to upload")
    parser.add_argument("--reference", default="zaid", help="Reference profile key")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")

    try:
        result = upload_video(args.url, args.video, args.reference)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.error(str(error))

    print_result(result)


if __name__ == "__main__":
    main()