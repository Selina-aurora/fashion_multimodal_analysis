# PRD Compliance Matrix

_Last updated: 2026-09-14_

This matrix distinguishes **current engineering progress** from the **final PRD acceptance target**. A requirement marked "In progress" is not claimed as satisfied.

## 3.1 Fine-grained vision foundation

| PRD item | PRD requirement | Current implementation / evidence | Status | Gap before acceptance |
| --- | --- | --- | --- | --- |
| 3.1.1 Instance segmentation | RGB image input; output per-garment mask, bbox and category | Mask2Former baseline, DeepFashion2 parser, independent instance masks, person/garment ROI and adaptive-margin diagnostics | In progress | Current COCO-pretrained baseline is workflow validation, not the final fashion-category model; final 8-category mapping, IoU and latency acceptance are pending |
| 3.1.1 Category coverage | Tops, pants, skirts, coats, dresses, shoes, bags, accessories | DeepFashion2 annotations are parsed; final PRD 8-category prediction layer is not frozen | In progress | Define/freeze category mapping and evaluate category outputs |
| 3.1.1 Performance | IoU >= 0.85; <= 50 ms/image | No final acceptance benchmark is claimed | Not yet verified | Run controlled IoU and latency benchmark on the frozen 3.1.1 configuration |
| 3.1.2 Language-guided localization | Image + natural language query -> local region mask + bbox | Grounding DINO bbox/score pipeline; prompt studies; ROI/LOCAL diagnostics; 88-case verified-positive benchmark | In progress | Fine-grained mask output is not yet integrated; final prompt/crop parameters and manual localization metrics are still being finalized |
| 3.1.2 Region coverage | Collar, cuff, hem, pocket, shoulder, waist, pattern, decoration, etc. | Current controlled benchmark covers sleeve, collar, button, zipper | Partial | Expand representative PRD region coverage after the current diagnostic stage |
| 3.1.2 Performance | Localization accuracy >= 92%; <= 30 ms | Automatic prediction coverage is reported separately from manual localization quality; no 92% claim is made | Not met / not yet verified | Complete manual accuracy evaluation and optimize the frozen baseline; measure latency on the deployment target |
| 3.1.3 Attribute extraction | Image + target mask -> fine-grained attribute label + confidence | Attribute schema and implementation plan are documented | Planned this week | Implement baseline, structured output, batch evaluation and integration with 3.1.1/3.1.2 |
| 3.1.3 Attribute scope | 14 categories / 200+ attributes, including material, craft and design | Initial schema focuses on color, pattern/texture, sleeve, collar/neckline and closure | Early stage | Expand toward the PRD taxonomy after the first baseline is stable |
| 3.1.3 Performance | Accuracy >= 88%; <= 20 ms | No completed production baseline is claimed | Not yet verified | Establish benchmark, field-level accuracy and latency measurement |

## Downstream modules

PRD 3.2/3.3 and deployment requirements (multimodal QA, content generation, Agent/RAG, FastAPI, Docker and TensorRT optimization) remain **outside the current 3.1 milestone**. They are intentionally not claimed as implemented in this repository snapshot.

## Current milestone definition

For the present development week, “3.1 complete” means that 3.1.1, 3.1.2 and 3.1.3 can run as a coherent baseline with recorded parameters, evaluation artifacts, error analysis and a reproducible end-to-end example. This milestone is **not equivalent to final PRD acceptance** until the PRD performance thresholds are independently verified.
