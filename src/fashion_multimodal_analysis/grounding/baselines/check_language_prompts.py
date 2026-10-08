"""3.1.2 文本引导区域定位：检测覆盖率、粗框可用率和人工定位准确率分开解释。"""

from __future__ import annotations

from fashion_multimodal_analysis.common.paths import (
    data_root,
)
from fashion_multimodal_analysis.common.paths import project_root as get_project_root
from fashion_multimodal_analysis.common.paths import (
    resolve_path as resolve_artifact_path,
)


def main() -> None:
    """解析运行参数并执行本模块的实验入口。"""
    prompts = {
        "part": [
            "the collar of the shirt",
            "the zipper of the jacket",
        ],
        "attribute": [
            "the floral pattern on the shirt",
            "the logo on the chest",
        ],
        "spatial": [
            "the left sleeve",
            "the right pocket",
        ],
        "relationship": ["the area where jacket overlaps with inner shirt"],
    }


if __name__ == "__main__":
    main()
