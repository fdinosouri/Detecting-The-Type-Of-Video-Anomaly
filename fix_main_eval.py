from pathlib import Path

path = Path("main.py")
text = path.read_text(encoding="utf-8")

start_marker = '    tmp_dict = {}\n'
end_marker = '    try:\n'

start = text.index(start_marker)
end = text.index(end_marker, start)

new_block = '''    tmp_dict = {}
    for v_name in scores_dict["prd"].keys():
        video_name = os.path.basename(v_name)
        prd = np.array(scores_dict["prd"][v_name])

        # prd may be shaped like:
        # [num_views, num_clips, num_classes]
        # or [num_clips, num_classes]
        # We convert it to [num_segments, num_classes]
        if prd.ndim == 3:
            prd = prd.reshape(-1, prd.shape[-1])
        elif prd.ndim == 2:
            prd = prd.reshape(-1, prd.shape[-1])
        elif prd.ndim == 1:
            prd = prd.reshape(1, -1)

        # Multi-class anomaly score:
        # class 0 = Normal
        # classes 1..13 = anomaly classes
        if prd.shape[1] > 2:
            anomaly_score = np.max(prd[:, 1:], axis=1)
        else:
            anomaly_score = prd[:, 1]

        tmp_dict[video_name] = [anomaly_score]

'''

text = text[:start] + new_block + text[end:]
path.write_text(text, encoding="utf-8")

print("DONE: main.py evaluation block fixed.")