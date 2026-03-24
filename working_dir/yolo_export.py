from ultralytics.utils.benchmarks import ProfileModels
from ultralytics import RTDETR


# model = RTDETR("ultralytics/cfg/models/26/yolo26n-rtdetr_p4_l3_efms.yaml")
# model.save("yolo26n_rtdetr_480.pt")
# ProfileModels(paths=["yolo26n_rtdetr_480.pt"], imgsz=480).run()

# model = RTDETR("ultralytics/cfg/models/26/yolo26s-rtdetr_p4_l3_efms.yaml")
# model.save("yolo26ns_rtdetr_512.pt")
# ProfileModels(paths=["yolo26ns_rtdetr_512.pt"], imgsz=512).run()

# model = RTDETR("ultralytics/cfg/models/26/yolo26s-rtdetr.yaml")
# model.save("yolo26s_rtdetr_640.pt")
# ProfileModels(paths=["yolo26s_rtdetr_640.pt"], imgsz=640).run()

# model = RTDETR("ultralytics/cfg/models/26/yolo26m-rtdetr.yaml")
# model.save("yolo26m_rtdetr_640.pt")
# ProfileModels(paths=["yolo26m_rtdetr_640.pt"], imgsz=640).run()

# model = RTDETR("ultralytics/cfg/models/26/yolo26l-rtdetr.yaml")
# model.set_head_attr(disable_topk=True)
# model.save("yolo26l_rtdetr_640_disable_topk.pt")
# ProfileModels(paths=["yolo26l_rtdetr_640_disable_topk.pt"], imgsz=640).run()
# ProfileModels(paths=["yolo26l_rtdetr_640_disable_topk.onnx"], imgsz=640).run()

# ProfileModels(paths=["/home/esat/workspace/dinov3_rtdetr/dinov3_rtdetr_512.onnx"], imgsz=512).run()
# ProfileModels(paths=["/home/esat/workspace/dinov3_rtdetr/rtdetr_dinov3sta_detrl3_640.onnx"], imgsz=640).run()
# ProfileModels(paths=["/home/esat/workspace/dinov3_rtdetr/rtdetr_dinov3sta_detrl6_640.onnx"], imgsz=640).run()
# ProfileModels(paths=["/home/esat/workspace/dinov3_rtdetr/rtdetr_deimv2L_v3_PreEpc25_lrf05_deimaug_24epc.onnx"], imgsz=640).run()
# ProfileModels(paths=["/home/esat/workspace/deim/deim_dfine_hgnetv2_l_coco.onnx"], imgsz=640).run()
ProfileModels(paths=["/home/esat/workspace/deim/deim_dfine_hgnetv2_x_coco.onnx"], imgsz=640).run()


# model = RTDETR("ultralytics/cfg/models/26/yolo26x-rtdetr.yaml")
# model.save("yolo26x_rtdetr_640.pt")
# ProfileModels(paths=["yolo26x_rtdetr_640.pt"], imgsz=640).run()

# ProfileModels(paths=["yolo26l_rtdetr_640.onnx"], imgsz=640).run()

# ProfileModels(paths=["yolo26n.pt"]).run()
# ProfileModels(paths=["yolo26s.pt"]).run()
# ProfileModels(paths=["yolo26m.pt"]).run()
# ProfileModels(paths=["yolo26l.pt"]).run()
# ProfileModels(paths=["yolo26x.pt"]).run()

# ProfileModels(paths=["yolo26n.onnx"]).run()
# ProfileModels(paths=["yolo26s.onnx"]).run()
# ProfileModels(paths=["yolo26m.onnx"]).run()
# ProfileModels(paths=["yolo26l.onnx"]).run()
# ProfileModels(paths=["yolo26x.onnx"]).run()

# ProfileModels(paths=["working_dir/yolo26-detr-V1/yolo26_detr_n_obj_480.onnx"], imgsz=480).run()
# ProfileModels(paths=["working_dir/yolo26-detr-V1/yolo26_detr_ns_coco_512.onnx"], imgsz=512).run()
# ProfileModels(paths=["working_dir/yolo26-detr-V1/yolo26_detr_s_coco_640.onnx"], imgsz=640).run()
# ProfileModels(paths=["working_dir/yolo26-detr-V1/yolo26_detr_m_coco_640.onnx"], imgsz=640).run()
# ProfileModels(paths=["working_dir/yolo26-detr-V1/yolo26_detr_l_obj_640.onnx"], imgsz=640).run()
# ProfileModels(paths=["working_dir/yolo26-detr-V1/yolo26_detr_x_coco_640.onnx"], imgsz=640).run()

# ProfileModels(paths=["/home/esat/workspace/yolo26detr_v2/yolo26_detr_n_obj_480.onnx"], imgsz=480).run()
# ProfileModels(paths=["/home/esat/workspace/yolo26detr_v2/yolo26_detr_ns_obj_512.onnx"], imgsz=512).run()
# ProfileModels(paths=["/home/esat/workspace/yolo26detr_v2/yolo26_detr_s_obj_640.onnx"], imgsz=640).run()
# ProfileModels(paths=["/home/esat/workspace/yolo26detr_v2/yolo26_detr_m_obj_640.onnx"], imgsz=640).run()
# ProfileModels(paths=["/home/esat/workspace/yolo26detr_v2/yolo26_detr_l_obj_640.onnx"], imgsz=640).run()
# ProfileModels(paths=["/home/esat/workspace/yolo26detr_v2/yolo26_detrl4_l_obj_704.onnx"], imgsz=704).run()
# ProfileModels(paths=["/home/esat/workspace/yolo26detr_v2/yolo26_detr_x_obj_640.onnx"], imgsz=640).run()

# ProfileModels(paths=["working_dir/yolo26detr_opset17/yolo26_detr_n_obj_480_opset17.onnx"], imgsz=480).run()
# ProfileModels(paths=["working_dir/yolo26detr_opset17/yolo26_detr_ns_coco_512_opset17.onnx"], imgsz=512).run()
# ProfileModels(paths=["working_dir/yolo26detr_opset17/yolo26_detr_s_coco_640_opset17.onnx"], imgsz=640).run()
# ProfileModels(paths=["working_dir/yolo26detr_opset17/yolo26_detr_m_coco_640_opset17.onnx"], imgsz=640).run()
# ProfileModels(paths=["working_dir/yolo26detr_opset17/yolo26_detr_l_obj_640_opset17.onnx"], imgsz=640).run()
# ProfileModels(paths=["working_dir/yolo26detr_opset17/yolo26_detr_x_coco_640_opset17.onnx"], imgsz=640).run()

# ProfileModels(paths=["working_dir/yolo26nms_weights/yolo26n_nms.onnx"]).run()
# ProfileModels(paths=["working_dir/yolo26nms_weights/yolo26s_nms.onnx"]).run()
# ProfileModels(paths=["working_dir/yolo26nms_weights/yolo26m_nms.onnx"]).run()
# ProfileModels(paths=["working_dir/yolo26nms_weights/yolo26l_nms.onnx"]).run()
# ProfileModels(paths=["working_dir/yolo26nms_weights/yolo26x_nms.onnx"]).run()

# ProfileModels(paths=["onnx_exports/rfdetr-nano/rfdetr-nano.onnx"], imgsz=384).run()
# ProfileModels(paths=["onnx_exports/rfdetr-small/rfdetr-small.onnx"], imgsz=512).run()
# ProfileModels(paths=["onnx_exports/rfdetr-medium/rfdetr-medium.onnx"], imgsz=576).run()
# ProfileModels(paths=["onnx_exports/rfdetr-large/rfdetr-large.onnx"], imgsz=704).run()
# ProfileModels(paths=["onnx_exports/rfdetr-xlarge/rfdetr-xlarge.onnx"], imgsz=700).run()
# ProfileModels(paths=["onnx_exports/rfdetr-xxlarge/rfdetr-xxlarge.onnx"], imgsz=880).run()

# ProfileModels(paths=["output/lwdetr_tiny_coco/lwdetr_tiny.onnx"], imgsz=640).run()
# ProfileModels(paths=["output/lwdetr_small_coco/lwdetr_small.onnx"], imgsz=640).run()
# ProfileModels(paths=["output/lwdetr_medium_coco/lwdetr_medium.onnx"], imgsz=640).run()
# ProfileModels(paths=["output/lwdetr_large_coco/lwdetr_large.onnx"], imgsz=640).run()
# ProfileModels(paths=["output/lwdetr_xlarge_coco/lwdetr_xlarge.onnx"], imgsz=640).run()

# ProfileModels(paths=["deim_onnx_outputs/deim_dfine_hgnetv2_n_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_dfine_hgnetv2_s_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_dfine_hgnetv2_m_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_dfine_hgnetv2_l_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_dfine_hgnetv2_x_coco.onnx"], imgsz=640).run()

# ProfileModels(paths=["deim_onnx_outputs/deim_rtdetrv2_r18vd_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_rtdetrv2_r34vd_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_rtdetrv2_r50vd_m_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_rtdetrv2_r50vd_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["deim_onnx_outputs/deim_rtdetrv2_r101vd_coco.onnx"], imgsz=640).run()

# ProfileModels(paths=["/home/esat/workspace/deimv2/deimv2_hgnetv2_pico_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["/home/esat/workspace/deimv2/deimv2_hgnetv2_n_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["/home/esat/workspace/deimv2/deimv2_dinov3_s_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["/home/esat/workspace/deimv2/deimv2_dinov3_m_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["/home/esat/workspace/deimv2/deimv2_dinov3_l_coco.onnx"], imgsz=640).run()
# ProfileModels(paths=["/home/esat/workspace/deimv2/deimv2_dinov3_x_coco.onnx"], imgsz=640).run()
