import os
import torch
from torch import nn
import torch.nn.functional as F

from model.HRposeNew import get_pose_net as hr_pose_net

class FusedNet(nn.Module):   
  def __init__(self, fuseType,mod_src,fuse_stage=3):
    if(fuse_stage>3 or fuse_stage<2):
        print ("fuse stage not between correct range.")
        return 

    super(FusedNet, self).__init__()
     
    self.mod_src = mod_src
    self.dropout = nn.Dropout2d(p=0.2)
    self.fuse_stage = fuse_stage 
    self.fuseType = fuseType
    
    model_list = []
    for mod in self.mod_src:
        # path = model_paths[mod]
        if(mod == "RGB"):
            model = hr_pose_net(in_ch=3, out_ch=14, end_stage = fuse_stage+1).cuda()
        else:
            model =  hr_pose_net(in_ch=1, out_ch=14, end_stage = fuse_stage+1).cuda()
        model_list.append(model)
        
    conv_list=[]
    if (self.fuseType == "concat"):
        for branch_num in range(fuse_stage):
            conv_list.append(nn.Conv2d((2**(5+branch_num))*len(self.mod_src), 2**(5+branch_num), 1,1).cuda())  # 1x1 Conv to reduce channel dimension
    fusion_batchnorm_list = []

    for branch_num in range(fuse_stage):
        fusion_batchnorm_list.append(nn.BatchNorm2d(2**(5+branch_num)))
    self.fusion_batchnorms =  nn.ModuleList(fusion_batchnorm_list)
    
    self.model_list = nn.ModuleList(model_list)
    self.conv_list = nn.ModuleList(conv_list)
    self.channel_dropout = nn.Dropout2d(p=0.2)
    self.last = hr_pose_net(in_ch=4, out_ch=14,start_stage=fuse_stage+1)   #in_ch not required 
      
      
  # Defining the forward pass    
  def forward(self, x):
    data_list = []
    c = 0
    # print(x.shape)
    for mod in self.mod_src:
        if (mod == "RGB"):
            data = x[:,c:c+3,:,:]
            c += 3
        else:
            data = torch.unsqueeze(x[:,c,:,:], 1)
            c += 1
        data_list.append(data)
        
    batch_size = data_list[0].shape[0]
    
    pred_list = []
    for i in range(len(self.model_list)):
        pred = self.model_list[i](data_list[i])["stage_outputs"][self.fuse_stage-1]
        pred_list.append(pred)  # stage-2: pred_list-[[ib1,ib2],[rb1,rb2]]
     
    
    pred_branch_dict = {}
    for branch_num in range(len(pred_list[0])):
        t = []
        for pred in pred_list:
            t.append(pred[branch_num])
        pred_branch_dict[branch_num] = t    # stage-2: pred_branch_dict-{0:[rgb-b1,de-b1],1:[ib2,rb2]}
        
          
    if (self.fuseType == "concat"):
        x=[]
        for branch_num in range(len(pred_list[0])):
            l = [self.channel_dropout(pred_branch_dict[branch_num][0]) , self.channel_dropout(pred_branch_dict[branch_num][1])]
            concated = torch.cat(l, dim=1)
            y = self.conv_list[branch_num](concated)
            y = self.fusion_batchnorms[branch_num](y)
            y = F.relu(y)
            x.append(y)
    else:
        x=[]
        for branch_num in range(len(pred_list[0])) :
            pred_branch_dict[branch_num][0] = self.channel_dropout(pred_branch_dict[branch_num][0])
            pred_branch_dict[branch_num][1] = self.channel_dropout(pred_branch_dict[branch_num][1])
            added = torch.sum(torch.stack(pred_branch_dict[branch_num]), dim=0)
            added = self.fusion_batchnorms[branch_num](added)
            added = F.relu(added)
            x.append(added)
            
    x=self.last(x)
    return x
  
def get_pose_net(in_ch,out_ch,fuse_type, mod_src,fuse_stage):
    model = FusedNet(fuseType=fuse_type, mod_src=mod_src, fuse_stage=int(fuse_stage))
    return model