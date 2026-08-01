from pathlib import Path

path = Path("main.py")
lines = path.read_text(encoding="utf-8").splitlines()

start_index = None

for i, line in enumerate(lines):
    if line.strip() == "if config.TEST.ONLY_TEST:":
        start_index = i
        break

if start_index is None:
    raise RuntimeError("Could not find: if config.TEST.ONLY_TEST:")

end_index = None

for i in range(start_index + 1, len(lines)):
    line = lines[i]

    if line.startswith("    for epoch ") or line.startswith("    for epoch in "):
        end_index = i
        break

if end_index is None:
    raise RuntimeError("Could not find the training epoch loop after ONLY_TEST block.")

new_block = [
    "    if config.TEST.ONLY_TEST:",
    "        out_path = os.path.join(config.OUTPUT, \"test_scores.pkl\")",
    "",
    "        if os.path.exists(out_path):",
    "            scores_dict = mmcv.load(out_path)",
    "        else:",
    "            scores_dict = validate(val_loader, text_labels, model, config, out_path)",
    "",
    "        tmp_dict = {}",
    "",
    "        for v_name in scores_dict[\"prd\"].keys():",
    "            video_name = os.path.basename(v_name)",
    "            prd = np.array(scores_dict[\"prd\"][v_name])",
    "",
    "            if prd.ndim == 3:",
    "                prd = prd.reshape(-1, prd.shape[-1])",
    "            elif prd.ndim == 2:",
    "                prd = prd.reshape(-1, prd.shape[-1])",
    "            elif prd.ndim == 1:",
    "                prd = prd.reshape(1, -1)",
    "",
    "            if prd.shape[1] > 2:",
    "                anomaly_score = np.max(prd[:, 1:], axis=1)",
    "            else:",
    "                anomaly_score = prd[:, 1]",
    "",
    "            tmp_dict[video_name] = [anomaly_score]",
    "",
    "        try:",
    "            auc_all_p, auc_ano_p = evaluate_result(tmp_dict, config.DATA.VAL_FILE)",
    "            logger.info(f'AUC: [{auc_all_p:.3f}/{auc_ano_p:.3f}]\\t')",
    "        except Exception as e:",
    "            logger.info(f\"Skipping AUC evaluation because: {e}\")",
    "",
    "        return",
    "",
]

lines = lines[:start_index] + new_block + lines[end_index:]

path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("DONE: ONLY_TEST block cleaned.")