# Core-region SPATIAL_HR manual audit — final

## Manual audit rubric

- **correct**: bbox substantially matches the requested local region.
- **coarse**: target is inside the bbox, but the bbox is substantially broader than the target.
- **wrong**: bbox lands mainly on another region/object.
- **missed**: no bbox is returned.

## Final interpretation

SPATIAL_HR clearly reduced effective box area for collar/cuff, but did not convert the reviewed cases into strict region-level localization. The dominant failure mode remains **coarse localization**: Grounding DINO tends to cover the available local crop rather than isolate the precise part.

- **Collar:** baseline 5/5 coarse; SPATIAL_HR 4/5 coarse + 1/5 missed. Search area shrank, but strict accuracy did not improve.
- **Cuff:** baseline 4/5 coarse + 1/5 missed; SPATIAL_HR 5/5 coarse. This is a useful recall improvement, but still no strict cuff localization.
- **Hem:** baseline 3/5 coarse + 2/5 missed; SPATIAL_HR 5/5 missed. The current lower-band strategy should be rejected.
- **Overall:** no `coarse -> correct` transition was observed in the 15-case diagnostic.

## Engineering conclusion

Target-specific cropping is useful as a **region proposal / attention restriction** mechanism, but Grounding DINO alone is not sufficient for the required fine-grained bbox/mask output. The next stage should keep the useful spatial prior for collar/cuff, discard the current hem lower-band setting, and add a local refinement/segmentation stage rather than continuing to lengthen prompts or increase input resolution.
