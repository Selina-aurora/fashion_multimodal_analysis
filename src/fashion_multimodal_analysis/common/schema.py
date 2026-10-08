"""项目公共接口：参数约定与返回结构供其他模块复用。

Frozen PRD garment category IDs; background is zero.
"""

from __future__ import annotations

CLASS_TO_ID = {
    "top": 1,
    "pants": 2,
    "skirt": 3,
    "outerwear": 4,
    "dress": 5,
    "shoe": 6,
    "bag": 7,
    "accessory": 8,
}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
NUM_CLASSES = 9
