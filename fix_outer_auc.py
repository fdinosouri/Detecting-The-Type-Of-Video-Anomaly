from pathlib import Path

path = Path("main.py")
lines = path.read_text(encoding="utf-8").splitlines()

# Replace the OUTER AUC block around lines 116-128.
# Line numbers in Python list are zero-based.
start_line = 116
end_line = 128

indent = " " * 16

new_block = [
    indent + "tmp_dict = {}",
    indent + "for v_name in scores_dict['prd'].keys():",
    indent + "    video_name = os.path.basename(v_name)",
    indent + "    prd = np.array(scores_dict['prd'][v_name])",
    "",
    indent + "    if prd.ndim == 3:",
    indent + "        prd = prd.reshape(-1, prd.shape[-1])",
    indent + "    elif prd.ndim == 2:",
    indent + "        prd = prd.reshape(-1, prd.shape[-1])",
    indent + "    elif prd.ndim == 1:",
    indent + "        prd = prd.reshape(1, -1)",
    "",
    indent + "    if prd.shape[1] > 2:",
    indent + "        anomaly_score = np.max(prd[:, 1:], axis=1)",
    indent + "    else:",
    indent + "        anomaly_score = prd[:, 1]",
    "",
    indent + "    tmp_dict[video_name] = [anomaly_score]",
    "",
    indent + "try:",
    indent + "    auc_all_p, auc_ano_p = evaluate_result(tmp_dict, config.DATA.VAL_FILE)",
    indent + "    logger.info(f'AUC: [{auc_all_p:.3f}/{auc_ano_p:.3f}]\\t')",
    indent + "except Exception as e:",
    indent + "    logger.info(f\"Skipping AUC evaluation because: {e}\")",
]

# Convert user-facing line numbers to zero-based indexes.
start_index = start_line - 1
end_index = end_line

lines = lines[:start_index] + new_block + lines[end_index:]

path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("DONE: outer AUC block fixed.")