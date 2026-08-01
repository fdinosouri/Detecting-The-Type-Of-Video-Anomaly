from pathlib import Path

path = Path("main.py")
text = path.read_text(encoding="utf-8")

old_block = '''    try:
        auc_all_p, auc_ano_p = evaluate_video_level_auc(tmp_dict, config.DATA.VAL_FILE)
        logger.info(f'AUC: [{auc_all_p:.3f}/{auc_ano_p:.3f}]\\t')
    except Exception as e:
        logger.info(f"Skipping AUC evaluation because: {e}")
'''

if old_block not in text:
    raise RuntimeError("Target duplicate AUC block was not found.")

pos = text.rfind(old_block)
text = text[:pos] + text[pos + len(old_block):]

path.write_text(text, encoding="utf-8")

print("DONE: duplicate AUC inside validate removed.")