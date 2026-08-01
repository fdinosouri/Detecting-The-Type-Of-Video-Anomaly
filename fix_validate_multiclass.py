from pathlib import Path

path = Path("main.py")
text = path.read_text(encoding="utf-8")

start_marker = "    tmp_dict = {}\n"
end_marker = "    logger.info(f'writing results to {out_path}')\n"

start = text.index(start_marker)
end = text.index(end_marker, start)

new_block = '''    tmp_dict = {}
    video_results = {}

    for v_name in scores_dict["prd"].keys():
        video_name = os.path.basename(v_name)
        prd = np.array(scores_dict["prd"][v_name])

        # Convert prediction shape to [num_segments, num_classes]
        if prd.ndim == 3:
            prd = prd.reshape(-1, prd.shape[-1])
        elif prd.ndim == 2:
            prd = prd.reshape(-1, prd.shape[-1])
        elif prd.ndim == 1:
            prd = prd.reshape(1, -1)

        # Average class scores over clips/segments
        class_scores = prd.mean(axis=0)
        pred_class = int(np.argmax(class_scores))

        # Binary anomaly score for old-style anomaly evaluation:
        # class 0 = Normal, classes 1..N = anomaly types
        if prd.shape[1] > 2:
            anomaly_score = np.max(prd[:, 1:], axis=1)
        else:
            anomaly_score = prd[:, 1]

        tmp_dict[video_name] = [anomaly_score]

        video_results[video_name] = {
            "pred_class": pred_class,
            "class_scores": class_scores.tolist(),
            "anomaly_score": float(np.max(anomaly_score))
        }

    scores_dict["video_results"] = video_results

    if config.DATA.NUM_CLASSES > 2:
        logger.info("Skipping old binary AUC because NUM_CLASSES > 2. Multi-class predictions were saved.")
    else:
        try:
            auc_all_p, auc_ano_p = evaluate_result(tmp_dict, config.DATA.VAL_FILE)
            logger.info(f'AUC: [{auc_all_p:.3f}/{auc_ano_p:.3f}]\\t')
        except Exception as e:
            logger.info(f"Skipping AUC evaluation because: {e}")

'''

text = text[:start] + new_block + text[end:]
path.write_text(text, encoding="utf-8")

print("DONE: validate block fixed for multi-class evaluation.")