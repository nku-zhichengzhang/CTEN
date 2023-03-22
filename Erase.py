import torch.nn.functional as F
import random

def get_erase_segment(output):
    output_softmax=F.softmax(output,dim=2)
    output_softmax_max=output_softmax.min(dim=2,keepdim=True)[1]
    print(output_softmax_max)

def get_erase_segment1(base_out,target,index):
    base_out_wide_softmax=F.softmax(base_out,dim=2)
    base_out_wide_softmax_max_dim_1=base_out_wide_softmax.max(dim=2,keepdim=True)[1]
    print(base_out_wide_softmax_max_dim_1.shape)
    print(base_out_wide_softmax_max_dim_1.squeeze(dim=2).shape)
    base_out_dim1_min=base_out.min(dim=1,keepdim=True)[0]
    base_out_dim1_max=base_out.max(dim=1,keepdim=True)[0]
    base_out_dim1_minmax=(base_out-base_out_dim1_min)/(base_out_dim1_max-base_out_dim1_min)
    result=base_out_wide_softmax*base_out_dim1_minmax
    result=result.max(dim=1,keepdim=False)[1]
    erase_dict={}
    for i,t in enumerate(target):
        erase_dict[index[i].item()]=result[i][t].item()
    print(erase_dict)
    return erase_dict

def get_random_erase_segment(base_out,target,index):
    erase_dict = {}
    for i, t in enumerate(target):
        erase_dict[index[i].item()] = random.randrange(0,5,1)
    return erase_dict

def get_collect_best_info(base_out,base_out_two,target,index):
    base_out_wide_softmax=F.softmax(base_out,dim=2)
    base_out_two_wide_softmax=F.softmax(base_out_two,dim=2)
    base_out_max_dim_1=base_out_wide_softmax.max(dim=1,keepdim=True)[0]
    base_out_two_max_dim_1=base_out_two_wide_softmax.max(dim=1,keepdim=True)[0]
    print(base_out_max_dim_1,base_out_two_max_dim_1)

def collect_info_segment(base_out,base_out_two,target):
    base_out_wide_softmax = F.softmax(base_out, dim=2)
    base_out_two_wide_softmax=F.softmax(base_out_two,dim=2)
    base_out_segment=base_out_wide_softmax.max(dim=2,keepdim=True)[1].squeeze(dim=2)
    base_out_two_segment=base_out_two_wide_softmax.max(dim=2,keepdim=True)[1].squeeze(dim=2)
    with open("/home/ubuntu5/wlj/result/tsn.output",'a') as f:
        for i in range(base_out_segment.shape[0]):
            for j in range(base_out_segment.shape[1]):
                f.write(str(base_out_segment[i][j].item())+" ")
            '''for k in range(base_out_two_segment.shape[1]):
                f.write(base_out_two_segment[i][k] + " ")'''
            #for l in range(target.shape[1]):
            f.write("target:"+str(target[i].item())+" ")
            f.write('\n')
