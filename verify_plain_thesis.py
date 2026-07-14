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
    mainstream_models = (
        "U-Net",
        "Attention U-Net",
        "U-Net++",
        "U-Net3+",
        "TransUnet",
        "MedT",
        "SwinUnet",
        "UNeXt",
        "CMU-Net",
        "CMUNeXt",
        "Mobile U-ViT",
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
        "mainstream_review_contains_eleven_models": (
            "### 1.3.2 主流对比模型介绍" in markdown
            and all(model in markdown for model in mainstream_models)
        ),
        "both_research_chapters_have_comparisons": (
            "## 3.5 与主流模型的参考对比" in markdown
            and "## 4.5 与主流模型的参考对比" in markdown
            and "UnetAB（本文）" in markdown
            and "UABCD（本文）" in markdown
        ),
        "reference_metrics_match_source_table": all(
            value in markdown
            for value in (
                "68.61±2.86",
                "76.97±3.10",
                "72.88±2.72",
                "81.18±3.05",
                "650.48",
                "199.74 GFLOPs",
            )
        ),
        "comparison_protocol_difference_disclosed": (
            "参考模型的标准差来自三次随机数据划分" in markdown
            and "本文的标准差来自同一划分上的三次训练" in markdown
            and "不能解释为完全相同的实验" in markdown
        ),
        "new_references_present": all(
            "[{}]".format(index) in markdown for index in range(31, 37)
        ),
        "at_least_ten_tables": len(document.tables) >= 10,
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
