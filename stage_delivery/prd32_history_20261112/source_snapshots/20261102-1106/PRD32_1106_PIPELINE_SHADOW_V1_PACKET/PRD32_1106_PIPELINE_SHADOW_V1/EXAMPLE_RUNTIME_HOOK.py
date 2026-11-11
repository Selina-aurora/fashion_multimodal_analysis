"""Illustration only; adapt to actual GPU Qwen runner after auditing entrypoint.

The pipeline must obtain mask_source_instance_id from V5 segmentation output,
not copy its expected value from the request. No models are invoked here.
"""
from runtime_identity import InstanceKey,make_request,attach_host_response

def demonstrate():
    segmentation_instance={'case_id':'fresh_dev_001','instance_id':'pred_0000',
                           'image_sha256':'a'*64, 'bbox':(1.0,2.0,30.0,40.0)}
    target=InstanceKey(segmentation_instance['case_id'],segmentation_instance['instance_id'],segmentation_instance['image_sha256'])
    # Real callers must use the actual distinct ROI/Mask metadata they observed.
    request=make_request(target,task='tags',variant='baseline',
                         roi_instance_id='pred_0000',mask_instance_id='pred_0000',
                         marked_image_bytes=b'illustration',bbox_xyxy=segmentation_instance['bbox'])
    # Correlation must come from host scheduler's fulfilled request id, not from model text.
    response=attach_host_response(request,completed_request_id=request.request_id,
                                  raw_text='{"category":"top"}',
                                  response_origin_instance_id='pred_0000')
    assert response['publish_allowed'] is False
    return response

if __name__=='__main__': print(demonstrate()['guard_status'])
