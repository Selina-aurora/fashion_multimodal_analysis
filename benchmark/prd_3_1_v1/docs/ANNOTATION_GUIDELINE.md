# Annotation Guideline v1

## General principles

1. Annotate what is visibly supported by the image.
2. Do not infer hidden material, composition, manufacturing process, or brand facts from appearance alone.
3. Mark uncertainty explicitly instead of forcing a label.
4. Do not use model predictions as the source of ground truth.
5. Every benchmark-test annotation must have a review status.

## Review status

Allowed values:
- `unreviewed`
- `reviewed`
- `adjudicated`

For a release-grade benchmark, use `reviewed` or `adjudicated`.

## Ambiguity

Use `ambiguous=1` when:
- the target is visually occluded;
- image resolution is insufficient;
- multiple allowed labels are equally defensible;
- the local region has no stable visible boundary.

Ambiguous cases are excluded from primary attribute-accuracy denominators but remain in the dataset for coverage/error analysis.

## 3.1.1 instance annotation

Required:
- source image
- garment ID
- one of the frozen eight classes
- GT bbox
- GT instance mask
- source dataset
- object size / area
- review status

## 3.1.2 local-region annotation

Required:
- garment ID
- query ID and exact query text
- requested region
- GT region bbox
- GT region mask when a reliable mask can be drawn
- ambiguity flag
- reviewer note

Region labels:
- collar
- cuff
- hem
- pocket
- shoulder
- waist
- pattern
- decoration

Do not label the entire garment as the local region unless the region itself genuinely occupies that area.

## 3.1.3 attribute annotation

Each attribute row must include:
- applicability
- human GT label
- ambiguity
- annotation note
- review status

Do not force an attribute onto unsupported garment categories.

Current v1 focus is visible/design attributes. Material/composition and craftsmanship/process are outside the primary v1 acceptance scope unless separately annotated with reliable evidence.
