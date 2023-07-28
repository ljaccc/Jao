
# 导出模型
import torch
from detect_show import LPCDet
from models.LPCDet import LPCDet
model_configs = {
    'backbone': {'type': 'resnet50', 'pretrained': False, "in_channels": 3},
    'neck': {'type': 'FPEM_FFM', 'fpem_repeat': 2},  # 特征融合，FPN or FPEM_FFM
    'head': {'type': 'SRCHead', 'num_classes': 1},
}
# 实例化
LPCDet_pt = LPCDet(model_config=model_configs)##模型预测代码的载入模型部分
# 从训练文件中加载
LPCDet_pt.load_state_dict(torch.load(r'F:\6Model_deployment\LPCDet_L\run\20230510-12_58\weight\LPCDet_last_acc0.984127_loss0.4267.pth'))##pth训练模型载入地址
LPCDet_pt.eval()
sm = torch.jit.script(LPCDet_pt)
sm.save("LPCDet_JX.pt")