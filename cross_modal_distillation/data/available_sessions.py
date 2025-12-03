from enum import Enum

import numpy as np


class AvailableSessions(Enum):
    MonkeyI: list = [
        "20160622_01",
        "20160624_03",
        "20160627_01",
        "20160630_01",
        "20160915_01",
        "20160916_01",
        "20160921_01",
        "20160927_04",
        "20160927_06",
        "20160930_02",
        "20160930_05",
        "20161005_06",
        "20161006_02",
        "20161007_02",
        "20161011_03",
        "20161013_03",
        "20161014_04",
        "20161017_02",
        "20161024_03",
        "20161025_04",
        "20161026_03",
        "20161027_03",
        "20161206_02",
        "20161207_02",
        "20161212_02",
        "20161220_02",
        "20170123_02",
        "20170124_01",
        "20170127_03",
        "20170131_02",
    ]
    MonkeyC: list = [
        "e1_1",
        "e2_1",
        "e2_2",
        "e3_1",
        "e3_2",
        "e3_3",
        "e3_4",
        "e4_1",
        "e4_2",
        "e4_3",
        "e5_1",
        "e5_2",
    ]


def get_all_sessions():
    sessions = []
    for subject in AvailableSessions:
        sessions_s = [f"{subject.name}_{v}" for v in subject.value]
        sessions.extend(sessions_s)

    return np.unique(sessions).tolist()
