from pathlib import Path

path = Path("main.py")
lines = path.read_text(encoding="utf-8").splitlines()

helper_lines = [
    "",
    "def evaluate_video_level_auc(vid2abnormality, anno_file):",
    "    from sklearn.metrics import roc_auc_score",
    "    import os",
    "    import numpy as np",
    "",
    "    y_true = []",
    "    y_score = []",
    "    pred_map = {}",
    "",
    "    for vid, value in vid2abnormality.items():",
    "        base = os.path.basename(vid)",
    "        arr = np.array(value).reshape(-1)",
    "",
    "        if arr.size == 0:",
    "            continue",
    "",
    "        pred_map[base] = float(np.max(arr))",
    "",
    "    for line in open(anno_file, 'r', encoding='utf-8'):",
    "        parts = line.strip().split()",
    "",
    "        if len(parts) < 3:",
    "            continue",
    "",
    "        video_name = os.path.basename(parts[0])",
    "",
    "        try:",
    "            label = int(parts[2])",
    "        except:",
    "            try:",
    "                label = int(parts[-1])",
    "            except:",
    "                continue",
    "",
    "        if video_name not in pred_map:",
    "            print('The video %s is excluded on the result!' % video_name)",
    "            continue",
    "",
    "        y_true.append(0 if label == 0 else 1)",
    "        y_score.append(pred_map[video_name])",
    "",
    "    if len(y_true) == 0:",
    "        raise RuntimeError('No matched videos between prediction and annotation file.')",
    "",
    "    if len(set(y_true)) < 2:",
    "        raise RuntimeError('AUC needs both Normal and Anomaly videos, but only one class was found.')",
    "",
    "    auc = roc_auc_score(y_true, y_score)",
    "    return auc, auc",
    "",
]

# Insert helper function after imports, only if it does not already exist.
if not any("def evaluate_video_level_auc(" in line for line in lines):
    insert_index = None

    for i, line in enumerate(lines):
        if line.startswith("from utils.tools import"):
            insert_index = i + 1
            break

    if insert_index is None:
        raise RuntimeError("Could not find import line: from utils.tools import")

    lines = lines[:insert_index] + helper_lines + lines[insert_index:]

text = "\n".join(lines) + "\n"

text = text.replace(
    "auc_all_p, auc_ano_p = evaluate_result(tmp_dict, config.DATA.VAL_FILE)",
    "auc_all_p, auc_ano_p = evaluate_video_level_auc(tmp_dict, config.DATA.VAL_FILE)"
)

path.write_text(text, encoding="utf-8")

print("DONE: video-level AUC enabled.")