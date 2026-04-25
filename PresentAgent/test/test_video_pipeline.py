from types import SimpleNamespace

from pptagent.apis import replace_image
from pptagent.presentation import Layout


def test_layout_validate_accepts_video_asset(tmp_path):
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake-video")

    layout = Layout.from_dict(
        "Video Layout",
        {
            "template_id": 1,
            "slides": [1],
            "content_schema": {
                "visual": {
                    "type": "image",
                    "data": ["placeholder.png"],
                    "description": "Visual slot",
                }
            },
        },
    )
    editor_output = {"visual": {"data": [str(video_path)]}}

    layout.validate(editor_output, str(tmp_path))

    assert editor_output["visual"]["data"][0] == str(video_path)


def test_replace_image_delegates_to_replace_video(monkeypatch):
    calls = []

    def fake_replace_video(slide, img_id, video_path):
        calls.append((slide, img_id, video_path))

    monkeypatch.setattr("pptagent.apis.replace_video", fake_replace_video)

    slide = SimpleNamespace()
    replace_image(slide, None, 3, "demo.mp4")

    assert calls == [(slide, 3, "demo.mp4")]
