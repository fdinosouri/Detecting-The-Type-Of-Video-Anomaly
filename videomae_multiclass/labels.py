"""UCF-Crime multiclass label definitions (14 classes, Normal at index 0)."""

CLASSES = [
    "Normal",
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]

NUM_CLASSES = len(CLASSES)
NORMAL_INDEX = CLASSES.index("Normal")

LABEL2ID = {name: i for i, name in enumerate(CLASSES)}
ID2LABEL = {i: name for i, name in enumerate(CLASSES)}


def label_from_path(path: str) -> str:
    """Infer the class name from a UCF-Crime style file path.

    Anomaly videos live under a folder named after their class
    (e.g. ``Anomaly-Videos-Part-1/Abuse/Abuse036_x264.mp4``); anything
    else (``Training-Normal-Videos-Part-1/...``, ``Testing_Normal_Videos...``)
    is Normal.
    """
    normalized = path.replace("\\", "/")
    for part in normalized.split("/"):
        if part in LABEL2ID and part != "Normal":
            return part
    filename = normalized.rsplit("/", 1)[-1]
    for name in CLASSES:
        if name != "Normal" and filename.startswith(name):
            return name
    return "Normal"
