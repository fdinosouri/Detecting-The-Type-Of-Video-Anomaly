from pathlib import Path

path = Path("main.py")
text = path.read_text(encoding="utf-8")

old_block = '''        prd = np.array(scores_dict["prd"][v_name])

        if prd.ndim == 2 and prd.shape[1] > 2:
            anomaly_score = np.max(prd[:, 1:], axis=1)
        else:
            anomaly_score = prd[:, 1]

        tmp_dict[video_name] = [anomaly_score]
'''

new_block = '''        prd = np.array(scores_dict["prd"][v_name])

        # Prediction shape can be:
        # [num_views, num_clips, num_classes]
        # or [num_clips, num_classes]
        # We reshape it to [num_segments, num_classes]
        if prd.ndim == 3:
            prd = prd.reshape(-1, prd.shape[-1])
        elif prd.ndim == 2:
            prd = prd.reshape(-1, prd.shape[-1])
        elif prd.ndim == 1:
            prd = prd.reshape(1, -1)

        # For binary AUC from multi-class output:
        # class 0 is Normal
        # classes 1..N are anomaly classes
        if prd.shape[1] > 2:
            anomaly_score = np.max(prd[:, 1:], axis=1)
        else:
            anomaly_score = prd[:, 1]

        tmp_dict[video_name] = [anomaly_score]
'''

if old_block not in text:
    raise RuntimeError("Target block was not found. main.py may already be changed.")

text = text.replace(old_block, new_block)
path.write_text(text, encoding="utf-8")

print("DONE: multi-class AUC block fixed.")