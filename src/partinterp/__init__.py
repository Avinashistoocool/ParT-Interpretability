"""partinterp: mechanistic interpretability tools for the Particle Transformer (ParT)."""
CLASS_LIST = ["QCD", "Hbb", "Hcc", "Hgg", "H4q", "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl"]
CLASS_FILE_PREFIX = {
    "QCD": "ZJetsToNuNu", "Hbb": "HToBB", "Hcc": "HToCC", "Hgg": "HToGG", "H4q": "HToWW4Q",
    "Hqql": "HToWW2Q1L", "Zqq": "ZToQQ", "Wqq": "WToQQ", "Tbqq": "TTBar", "Tbl": "TTBarLep",
}
__version__ = "0.1.0"
