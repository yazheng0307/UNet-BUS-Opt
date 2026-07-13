import json
import re
from pathlib import Path
from zipfile import ZipFile

from docx import Document


ROOT = Path(__file__).resolve().parent
MARKDOWN = ROOT / "硕士学位论文初稿_低专业度版.md"
WORD = ROOT / "硕士学位论文初稿_低专业度版.docx"


def main():
    markdown = MARKDOWN.read_text(encoding="utf-8")
    document = Document(WORD)
    document_text = "\n".join(
        [paragraph.text for paragraph in document.paragraphs]
        + [
            cell.text
            for table in document.tables
            for row in table.rows
            for cell in row.cells
        ]
    )
    with ZipFile(WORD) as archive:
        media = [
            name for name in archive.namelist()
            if name.startswith("word/media/") and not name.endswith("/")
        ]
    image_paths = re.findall(r"^!\[[^]]*\]\(([^) ]+)", markdown, re.MULTILINE)
    primary_metrics = (
        ("72.445", "80.718"),
        ("73.622", "81.817"),
        ("74.193", "82.537"),
        ("74.847", "82.943"),
        ("74.932", "83.022"),
        ("75.115", "83.480"),
    )
    plain_names = (
        "多特征增强模块",
        "全局关系增强模块",
        "边界细化模块",
        "易错区域修正模块",
    )
    checks = {
        "markdown_exists": MARKDOWN.is_file(),
        "word_exists": WORD.is_file(),
        "four_plain_names": all(
            name in markdown and name in document_text for name in plain_names
        ),
        "old_names_removed": not any(
            name in markdown for name in ("TriCAR", "CGR-Bridge", "BD-CoRefine", "UDER")
        ),
        "all_primary_metrics_present": all(
            iou in markdown and dice in markdown
            and iou in document_text and dice in document_text
            for iou, dice in primary_metrics
        ),
        "ten_markdown_images_exist": (
            len(image_paths) == 10
            and all((ROOT / path).is_file() for path in image_paths)
        ),
        "ten_word_images": len(media) == 10,
        "toc_heading": any(
            paragraph.text == "目录" for paragraph in document.paragraphs
        ),
        "no_missing_image_marker": "[缺失图片：" not in document_text,
        "equation_delimiters_balanced": markdown.count("$$") % 2 == 0,
        "title_matches": document.core_properties.title == (
            "基于改进 U-Net 的乳腺超声图像分割方法研究"
        ),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "document": {
            "paragraphs": len(document.paragraphs),
            "headings": sum(
                paragraph.style.name.startswith("Heading")
                for paragraph in document.paragraphs
            ),
            "tables": len(document.tables),
            "images": len(media),
            "markdown_chars": len(markdown),
            "word_paragraph_chars": sum(
                len(paragraph.text) for paragraph in document.paragraphs
            ),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
