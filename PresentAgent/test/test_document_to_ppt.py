from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from pptagent.document import Document, OutlineItem
from pptagent.model_utils import ModelManager
from pptagent.pptgen import PPTAgentAsync
from pptagent.presentation import Presentation
from pptagent.utils import Config


def _find_repo_root(script_path: Path) -> Path:
    current = script_path.resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "pptagent").is_dir():
            return candidate
    raise RuntimeError(
        f"Could not locate repository root from {script_path}; expected a parent directory containing 'pptagent/'."
    )


ROOT = _find_repo_root(Path(__file__))
DEFAULT_TEMPLATE_DIR = ROOT / "runs" / "pptx" / "default_template"


def _load_document(document_json_path: Path, image_dir: Path) -> Document:
    return Document.from_dict(
        json.loads(document_json_path.read_text(encoding="utf-8")),
        str(image_dir),
        False,
    )


async def _generate_ppt(
    *,
    document_json_path: Path,
    image_dir: Path,
    template_dir: Path,
    output_dir: Path,
    num_slides: int,
    outline_json_path: Path | None,
) -> None:
    models = ModelManager()
    source_doc = _load_document(document_json_path, image_dir)

    generation_config = Config(str(output_dir))
    template_config = Config(str(template_dir))
    presentation = Presentation.from_file(
        str(template_dir / "source.pptx"),
        template_config,
    )
    slide_induction = json.loads(
        (template_dir / "slide_induction.json").read_text(encoding="utf-8")
    )

    ppt_agent = PPTAgentAsync(
        models.text_model,
        models.language_model,
        models.vision_model,
        error_exit=False,
        retry_times=5,
    )
    ppt_agent.set_reference(
        config=generation_config,
        slide_induction=slide_induction,
        presentation=presentation,
    )

    outline = None
    if outline_json_path is not None:
        outline = [
            OutlineItem.from_dict(item)
            for item in json.loads(outline_json_path.read_text(encoding="utf-8"))
        ]

    prs, history = await ppt_agent.generate_pres(
        source_doc=source_doc,
        num_slides=num_slides,
        outline=outline,
    )
    if prs is None:
        raise RuntimeError("PPT generation failed; no presentation was returned.")

    final_pptx_path = output_dir / "final.pptx"
    prs.save(str(final_pptx_path))

    final_outline = outline or ppt_agent.outline
    (output_dir / "outline.json").write_text(
        json.dumps([item.to_dict() for item in final_outline], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "generation_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "presentation_text.txt").write_text(
        prs.to_text(show_image=True),
        encoding="utf-8",
    )

    notes_dump = []
    for slide in prs.slides:
        notes_dump.append(
            {
                "slide_idx": slide.slide_idx,
                "title": slide.slide_title,
                "layout": slide.slide_layout_name,
                "notes": slide.slide_notes,
            }
        )
    (output_dir / "slide_notes.json").write_text(
        json.dumps(notes_dump, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("success: True")
    print(f"document_json: {document_json_path}")
    print(f"template_dir: {template_dir}")
    print(f"final_pptx: {final_pptx_path}")
    print(f"outline: {output_dir / 'outline.json'}")
    print(f"history: {output_dir / 'generation_history.json'}")
    print(f"slide_notes: {output_dir / 'slide_notes.json'}")
    print(f"presentation_text: {output_dir / 'presentation_text.txt'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PPT and intermediate debug outputs from refined_doc.json."
    )
    parser.add_argument("--document-json", required=True, help="Path to refined_doc.json")
    parser.add_argument(
        "--image-dir",
        default="",
        help="Directory containing media assets referenced by the document; defaults to the document JSON parent directory",
    )
    parser.add_argument(
        "--template-dir",
        default=str(DEFAULT_TEMPLATE_DIR),
        help="Template directory containing source.pptx and slide_induction.json",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save PPT outputs")
    parser.add_argument("--num-slides", type=int, default=8, help="Target number of slides")
    parser.add_argument(
        "--outline-json",
        default="",
        help="Optional path to an existing outline.json to reuse instead of generating a fresh outline",
    )
    args = parser.parse_args()

    document_json_path = Path(args.document_json).resolve()
    if not document_json_path.exists():
        raise SystemExit(f"refined_doc.json not found: {document_json_path}")

    image_dir = Path(args.image_dir).resolve() if args.image_dir else document_json_path.parent
    template_dir = Path(args.template_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outline_json_path = Path(args.outline_json).resolve() if args.outline_json else None
    asyncio.run(
        _generate_ppt(
            document_json_path=document_json_path,
            image_dir=image_dir,
            template_dir=template_dir,
            output_dir=output_dir,
            num_slides=args.num_slides,
            outline_json_path=outline_json_path,
        )
    )


if __name__ == "__main__":
    main()
