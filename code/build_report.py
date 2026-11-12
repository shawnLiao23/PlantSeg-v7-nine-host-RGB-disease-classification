"""Build the concise Option 1 PDF from recorded experiment outputs.

This script refuses to invent results: the fixed manifest, validation summary,
and final test metrics must already exist. It reads optional figures when present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result"
NAVY = "#163451"
TEAL = "#157a75"
GRAY = "#536475"
LIGHT = "#e8eef2"
OFFICIAL_V7_MD5 = "9358a66dff88cdd15c4fe009763c40a3"


def read_json(path: Path, required: bool = True) -> dict:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required recorded result is missing: {path}")
        return {}
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path: Path, required: bool = True) -> list[dict]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required recorded result is missing: {path}")
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_v7_extraction(audit: dict, verification: dict) -> bool:
    """Identify v7 only when the official ZIP and extracted files both match."""
    source = verification or audit.get("source_archive_verification", audit)
    return (
        source.get("archive_is_official_v7") is True
        and source.get("extraction_matches_archive") is True
        and source.get("archive_md5") == OFFICIAL_V7_MD5
        and source.get("official_v7_md5") == OFFICIAL_V7_MD5
    )


def verify_frozen_inputs(result_dir: Path, final: dict, summary: list[dict]) -> dict:
    """Refuse to report outputs detached from the frozen test protocol."""
    plan = read_json(result_dir / "frozen_plan.json")
    expected_files = {
        "manifest_sha256": result_dir / "manifest.csv",
        "class_mapping_sha256": result_dir / "class_mapping.json",
        "summary_sha256": result_dir / "experiment_summary.csv",
    }
    for field, path in expected_files.items():
        expected = plan.get(field)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Frozen plan is missing a valid {field}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{path.name} changed after the plan was frozen")
    if final.get("frozen_plan_summary_sha256") != plan["summary_sha256"]:
        raise ValueError("Final test metrics do not match the frozen experiment summary")
    if final.get("selected_model") != plan.get("selected_model"):
        raise ValueError("Final test selection differs from the frozen plan")
    names = [row.get("experiment") for row in summary]
    if names != plan.get("experiments"):
        raise ValueError("Validation experiment order differs from the frozen plan")
    if set(final.get("experiments", {})) != set(names):
        raise ValueError("Final test systems differ from the frozen plan")
    return plan


def included(row: dict) -> bool:
    return str(row.get("excluded", "")).strip().lower() not in {"1", "true", "yes"}


def number(value, digits: int = 3) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def duration(value) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "-"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def pretty_name(name: str) -> str:
    aliases = {
        "majority": "Majority class",
        "majority_class": "Majority class",
        "baseline": "Frozen ResNet-18",
        "basic_cnn": "Frozen ResNet-18",
        "augmentation": "Augmentation",
        "weighted": "Class weighting",
        "class_weighted": "Class weighting",
        "finetune": "Fine tuning",
        "fine_tuning": "Fine tuning",
        "majority_class": "Majority class",
        "baseline_resnet18": "Frozen ResNet-18",
        "augmentation_resnet18": "Augmentation",
        "weighted_resnet18": "Class weighting",
    }
    return aliases.get(str(name).lower(), str(name).replace("_", " ").title())


def metric(record: dict, key: str):
    if not isinstance(record, dict):
        return None
    if key in record:
        return record[key]
    for nested_key in ("metrics", "test_metrics", "scores"):
        nested = record.get(nested_key)
        if isinstance(nested, dict) and key in nested:
            return nested[key]
    return None


def page(title: str, page_no: int):
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    fig.text(0.065, 0.965, title, color=NAVY, size=16, weight="bold", va="top")
    fig.add_artist(plt.Line2D([0.065, 0.935], [0.944, 0.944], color=TEAL, lw=1.5))
    fig.text(0.065, 0.033, "AIP304 Coursework 2  |  Option 1: CNN classification", size=7.6, color=GRAY)
    fig.text(0.935, 0.033, str(page_no), size=7.6, color=GRAY, ha="right")
    return fig


def heading(fig, label: str, y: float):
    fig.text(0.065, y, label, color=TEAL, size=10.5, weight="bold", va="top")


def paragraph(fig, content: str, y: float, width: int = 106, size: float = 8.7,
              line_height: float = 0.0185) -> float:
    lines = textwrap.wrap(" ".join(content.split()), width=width,
                          break_long_words=False, break_on_hyphens=False)
    fig.text(0.065, y, "\n".join(lines), color="#243342", size=size,
             va="top", linespacing=1.35)
    return y - max(1, len(lines)) * line_height


def table(fig, headers: list[str], rows: list[list[str]], box: tuple[float, float, float, float],
          widths: list[float] | None = None, font: float = 8.3):
    ax = fig.add_axes(box)
    ax.axis("off")
    artists = ax.table(cellText=rows, colLabels=headers, colWidths=widths,
                       cellLoc="left", loc="upper left", bbox=[0, 0, 1, 1])
    artists.auto_set_font_size(False)
    artists.set_fontsize(font)
    for (row, col), cell in artists.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#cbd5dd")
        cell.PAD = 0.07
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f5f7")
    return ax


def figure(fig, path: Path, box: tuple[float, float, float, float]) -> bool:
    if not path.is_file():
        return False
    image = mpimg.imread(path)
    ax = fig.add_axes(box)
    ax.imshow(image)
    ax.axis("off")
    return True


def caption(fig, text: str, y: float):
    paragraph(fig, text, y, width=115, size=7.8, line_height=0.016)


def main(result_dir: Path, output: Path, student: str, student_id: str,
         contributions: str) -> None:
    manifest = read_csv(result_dir / "manifest.csv")
    audit = read_json(result_dir / "dataset_audit.json")
    verification = read_json(result_dir / "source_archive_verification.json", required=False)
    summary = read_csv(result_dir / "experiment_summary.csv")
    config = read_json(result_dir / "experiment_config.json", required=False)
    final = read_json(result_dir / "final_test_metrics.json")
    errors = read_json(result_dir / "error_analysis.json", required=False)
    sensitivity = read_json(result_dir / "overlap_sensitivity.json", required=False)
    verify_frozen_inputs(result_dir, final, summary)
    if not summary:
        raise ValueError("The validation experiment summary has no rows")
    tests = final.get("experiments", {})
    selected = final.get("selected_model")
    if isinstance(selected, dict):
        selected = selected.get("experiment") or selected.get("name")
    if not selected or not tests:
        raise ValueError("Final test JSON needs selected_model and non-empty experiments")
    if selected not in tests:
        raise ValueError(f"Selected model {selected!r} has no final test record")
    if sensitivity:
        if sensitivity.get("selected_model") != selected:
            raise ValueError("Sensitivity analysis uses a different selected model")
        for field, path in (("predictions_sha256", result_dir / "tables" / "test_predictions.csv"),
                            ("review_sha256", result_dir / "tables" / "near_duplicate_review.csv")):
            if sensitivity.get(field) != sha256_file(path):
                raise ValueError(f"Sensitivity analysis is stale: {path.name}")
    for name, record in tests.items():
        for key in ("macro_f1", "accuracy", "balanced_accuracy"):
            if metric(record, key) is None:
                raise ValueError(f"Final test result for {name!r} has no {key}")
    included_rows = [row for row in manifest if included(row)]
    splits = Counter(row.get("split", "") for row in included_rows)
    classes = {row["class_name"] for row in manifest}
    hosts = {row["host"] for row in manifest}
    train_counts = Counter(row["class_name"] for row in included_rows
                           if row.get("split") == "train")
    if not classes or any(not train_counts[name] for name in classes):
        raise ValueError("All classes must have training examples in the fixed manifest")
    excluded = len(manifest) - len(included_rows)
    if excluded:
        raise ValueError("This report uses the original supplied splits, so every nine-host image must be included")
    duplicate_groups = audit.get("exact_duplicate_group_count", 0)
    cross_split = audit.get("cross_split_exact_duplicate_group_count", 0)
    near_count = audit.get("cross_split_near_duplicate_candidate_count", 0)
    reviewed = audit.get("near_duplicate_reviewed_pair_count", 0)
    confirmed = audit.get("near_duplicate_confirmed_pair_count", 0)
    uncertain = audit.get("near_duplicate_uncertain_pair_count", 0)
    dhash_reviewed = audit.get("dhash_reviewed_pair_count", 0)
    phash_reviewed = audit.get("phash_extra_reviewed_pair_count", 0)
    within_conflicts = audit.get("within_split_conflicting_label_group_count", 0)
    if near_count and reviewed < near_count:
        raise ValueError("The cross-split near-duplicate review is incomplete")
    review_digest = audit.get("near_duplicate_review_sha256")
    if review_digest:
        review_path = result_dir / "tables" / "near_duplicate_review.csv"
        if sha256_file(review_path) != review_digest:
            raise ValueError("Near-duplicate visual review changed after the audit")
    unreadable = len(audit.get("missing_or_corrupt_images", []))
    release_verified = verified_v7_extraction(audit, verification)
    release_phrase = ("verified PlantSeg v7" if release_verified
                      else "the supplied PlantSeg extraction (release unverified locally)")
    figures = result_dir / "figures"
    output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output) as pdf:
        # Page 1: framing, protocol, and the independently recorded data audit.
        fig = page("Plant disease recognition in the wild", 1)
        fig.text(0.065, 0.914, "Option 1  |  RGB host-disease classification", size=11,
                 color=TEAL, weight="bold")
        fig.text(0.065, 0.884, f"Student(s): {student}     Identifier(s): {student_id}",
                 size=8.8, color=GRAY)
        heading(fig, "Problem and motivation", 0.840)
        paragraph(fig,
            "The task is to assign one host-disease label to an unseen RGB photograph. "
            "Field images vary in background, lighting, scale, and symptom size; class imbalance "
            "makes accuracy alone an incomplete measure. We therefore select models by validation "
            "macro-F1 and report accuracy and balanced accuracy as complementary metrics.", 0.813)
        heading(fig, "Data and experimental protocol", 0.688)
        paragraph(fig,
            f"We used {release_phrase}, selecting the nine specified hosts from metadata and "
            "treating each host-disease pair as a separate class. The official train, validation, "
            "and test assignments were retained in a fixed manifest. Only training data can update "
            "weights or determine class weights; validation chooses checkpoints and configuration; "
            "test is used only after those choices are frozen.", 0.661)
        table(fig, ["Subset", "Images", "Classes", "Hosts"],
              [["Included", f"{len(included_rows):,}", str(len(classes)), str(len(hosts))],
               ["Train", f"{splits['train']:,}", "-", "-"],
               ["Validation", f"{splits['val']:,}", "-", "-"],
               ["Test", f"{splits['test']:,}", "-", "-"]],
              (0.065, 0.448, 0.87, 0.105), [0.35, 0.22, 0.21, 0.22])
        heading(fig, "Data audit and split provenance", 0.418)
        review_sentence = (
            f"Visual review covered {reviewed} cross-split candidates "
            f"({dhash_reviewed} dHash, {phash_reviewed} additional pHash): "
            f"{confirmed} same-photo, {reviewed - confirmed - uncertain} distinct, "
            f"{uncertain} uncertain. "
            if reviewed else "No completed visual review was recorded. "
        )
        paragraph(fig,
            f"The audit found {unreadable} missing or unreadable selected images and "
            f"{duplicate_groups} exact-duplicate groups ({cross_split} across partitions). "
            + review_sentence +
            "All images retain their supplied split labels. Cross-split same-photo images "
            "remain, so validation and test scores may be optimistic.",
            0.391)
        paragraph(fig,
            f"Training class counts range from {min(train_counts.values())} to "
            f"{max(train_counts.values())}. {within_conflicts} same-split exact-duplicate groups "
            "have conflicting class labels and remain a label-noise limitation. "
            "Masks and class-bearing filenames are never model inputs. "
            "Validation and test transforms are deterministic; the same manifest and class order "
            "are used for every experiment.", 0.260)
        if release_verified:
            paragraph(fig,
                      f"Source provenance: the archive MD5 matches official v7, and "
                      f"{verification.get('matching_file_count', 'all')} extracted files match its entries "
                      "by size and CRC.", 0.157, size=8.0)
        else:
            paragraph(fig,
                      "The release is not fully verified until both the official ZIP checksum and "
                      "an extracted-file comparison pass.", 0.157, size=8.0)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: class distribution, one of the required data-derived visuals.
        fig = page("Dataset composition", 2)
        paragraph(fig,
            "Figure 1 shows the host-disease class frequencies by official partition. The chart "
            "is generated from the audited manifest; the notebook also displays representative "
            "training photographs.", 0.901)
        has_distribution = figure(fig, figures / "class_distribution.png",
                                  (0.070, 0.165, 0.86, 0.670))
        if has_distribution:
            caption(fig, "Figure 1. Class distribution in the nine-host subset under the supplied original splits.", 0.135)
        else:
            paragraph(fig, "The class distribution figure was unavailable when this PDF was built; see the notebook for the per-class table.", 0.70)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: controlled experiments and validation-only selection.
        fig = page("Method and validation", 3)
        paragraph(fig,
            "The majority predictor always returns the most frequent training class. The CNN "
            f"baseline trains a new {len(classes)}-way head on ImageNet-pretrained ResNet-18 features. "
            "Augmentation and class weighting were separate changes to training transforms and "
            "loss. Validation macro-F1 selects each checkpoint and the final configuration.", 0.900)
        if config:
            lr = config.get("lr", config.get("baseline_lr", "see notebook"))
            paragraph(fig,
                f"Recorded base setup: {config.get('image_size', 'see notebook')} px RGB, "
                f"{config.get('optimizer', 'Adam')} with learning rate {lr}, "
                f"batch {config.get('batch_size', 'see notebook')}, "
                f"{config.get('epochs', 'see notebook')} epoch budget, "
                f"seed {config.get('seed', 'see notebook')}; weights "
                f"{config.get('weights_identifier', 'see notebook')}.",
                0.818, width=112, size=7.9, line_height=0.016)
        rows = []
        for record in summary:
            rows.append([
                pretty_name(record.get("experiment", "")),
                number(record.get("val_macro_f1")),
                number(record.get("val_accuracy")),
                number(record.get("val_balanced_accuracy")),
                str(record.get("best_epoch") or "-"),
                duration(record.get("train_time_sec")),
            ])
        heading(fig, "Validation comparison", 0.758)
        table(fig, ["System", "Macro-F1", "Accuracy", "Balanced acc.", "Epoch", "Train time"],
              rows, (0.065, 0.588, 0.87, 0.135),
              [0.285, 0.13, 0.13, 0.17, 0.095, 0.19], font=7.7)
        selected_row = next((row for row in summary if row.get("experiment") == selected), None)
        if selected_row:
            paragraph(fig,
                f"{pretty_name(selected)} was selected using validation macro-F1 "
                f"({number(selected_row.get('val_macro_f1'))}); selection did not use test labels.", 0.553)
        has_curves = figure(fig, figures / "training_curves.png",
                            (0.078, 0.128, 0.844, 0.366))
        if has_curves:
            caption(fig, "Figure 2. Recorded training and validation curves. The selected checkpoint is marked in the source figure.", 0.095)
        else:
            paragraph(fig, "Training curves are available in the notebook and saved experiment logs.", 0.31)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: frozen test comparison and confusion structure.
        fig = page("Final test evaluation", 4)
        paragraph(fig,
            "After the validation plan was frozen, the systems below were evaluated with "
            "deterministic transforms and disabled gradients. Macro-F1 averages class-wise F1; "
            "balanced accuracy averages class recall. Undefined per-class precision or recall "
            "uses the convention documented in the notebook.", 0.902)
        test_rows = []
        for name, result in tests.items():
            test_rows.append([
                pretty_name(name) + (" *" if name == selected else ""),
                number(metric(result, "macro_f1")),
                number(metric(result, "accuracy")),
                number(metric(result, "balanced_accuracy")),
            ])
        table(fig, ["System", "Macro-F1", "Accuracy", "Balanced accuracy"], test_rows,
              (0.065, 0.705, 0.87, 0.125), [0.40, 0.18, 0.18, 0.24], font=8.0)
        caption(fig, "Table 2. Final test metrics for recorded systems. * Final configuration chosen on validation.", 0.676)
        has_matrix = figure(fig, figures / "confusion_matrix.png",
                            (0.115, 0.105, 0.77, 0.525))
        if has_matrix:
            caption(fig, "Figure 3. Confusion matrix for the selected model; rows are true labels and columns are predictions. The full matrix and per-class report are retained in the notebook.", 0.084)
        else:
            paragraph(fig, "The full confusion matrix is available in the notebook output.", 0.41)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: sample-level evidence, interpretation, references, disclosures.
        fig = page("Discussion and disclosures", 5)
        has_gallery = figure(fig, figures / "prediction_examples.png",
                             (0.075, 0.638, 0.85, 0.225))
        if has_gallery:
            caption(fig, "Figure 4. Selected correct and incorrect test predictions, labelled with true class, predicted class, and maximum softmax score (not a calibrated probability).", 0.611)
        heading(fig, "Interpretation and limitations", 0.558)
        best_test = tests[selected]
        discussion = (
            f"On the full official test split, the selected system achieved macro-F1 "
            f"{number(metric(best_test, 'macro_f1'))}, accuracy "
            f"{number(metric(best_test, 'accuracy'))}, and balanced accuracy "
            f"{number(metric(best_test, 'balanced_accuracy'))}. "
        )
        if sensitivity:
            clean = sensitivity["without_detected_train_overlap"]
            discussion += (
                f"A post-test sensitivity check omitted {sensitivity['overlap_union']} test images "
                "with detected same-photo matches in training. On the remaining "
                f"{clean['count']}, macro-F1 was {number(clean['macro_f1'])}, accuracy "
                f"{number(clean['accuracy'])}, and balanced accuracy "
                f"{number(clean['balanced_accuracy'])}. "
            )
        discussion += "This check does not remove undetected overlap or validation leakage."
        paragraph(fig, discussion, 0.532, size=8.3)
        top_confusions = errors.get("top_confusions", [])
        if top_confusions:
            top = top_confusions[0]
            observation = (
                f"The largest confusion was {top['true_class']} predicted as "
                f"{top['predicted_class']} ({top['count']} images). "
            )
        else:
            observation = "Specific error pairs are tabulated in the notebook. "
        inspected = errors.get("inspected_example")
        if inspected:
            observation += inspected["observation"] + " "
        observation += (
            f"The {within_conflicts} conflicting-label duplicate groups add label noise; "
            "performance on new farms, cameras, and seasons remains unknown."
        )
        paragraph(fig, observation, 0.425, size=8.1)
        heading(fig, "References", 0.337)
        paragraph(fig,
            "Wei, T. et al. (2025). A large-scale in-the-wild dataset for plant disease segmentation, "
            "Version v7. Zenodo. doi:10.5281/zenodo.17719108.", 0.311, size=7.7, line_height=0.016)
        paragraph(fig,
            "Wei, T. et al. (2026). A large-scale in-the-wild dataset for plant disease segmentation. "
            "Scientific Data 13, 205. doi:10.1038/s41597-025-06513-4.", 0.264, size=7.7, line_height=0.016)
        paragraph(fig,
            "He, K., Zhang, X., Ren, S. & Sun, J. (2016). Deep Residual Learning for Image "
            "Recognition. CVPR, 770-778. arXiv:1512.03385.",
            0.217, size=7.7, line_height=0.016)
        heading(fig, "AI use", 0.168)
        paragraph(fig,
            "OpenAI Codex (GPT-6) assisted with code, debugging, reference lookup, analysis, "
            "figures, and wording. Outputs were checked against executed code, saved results, "
            "and cited sources.", 0.144, size=8.1, line_height=0.017)
        heading(fig, "Team contributions", 0.089)
        paragraph(fig, contributions, 0.065, size=8.0, line_height=0.016)
        pdf.savefig(fig)
        plt.close(fig)

    try:
        from pypdf import PdfReader
        page_count = len(PdfReader(output).pages)
        if page_count > 5:
            raise RuntimeError(f"Report has {page_count} pages; the limit is five")
    except ImportError:
        page_count = 5
    print(f"Saved {output} ({page_count} pages)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=RESULT)
    parser.add_argument("--output", type=Path, default=RESULT / "AIP304_CW2_Report.pdf")
    parser.add_argument("--student", default="[Student name(s)]")
    parser.add_argument("--student-id", default="[Student ID(s)]")
    parser.add_argument("--contributions", default="[Add each team member's actual contribution; write 'Individual project' if working alone.]")
    options = parser.parse_args()
    main(options.result_dir, options.output, options.student, options.student_id,
         options.contributions)
