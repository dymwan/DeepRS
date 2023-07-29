import torch




def getLineCrossPoint(skeleton,device='cpu'):
    '''
        get cross points and endpoints of all lines with GPU implementation
            all cross-points are labeled as 1
            all end-points are labeled as 2
    
    '''
    w, h = skeleton.shape

    skeleton_t = torch.from_numpy(skeleton).float().to(device).unsqueeze(0).unsqueeze(0)
    cross_point_map = torch.zeros(skeleton.shape, dtype=torch.uint8).to(device)

    helo_index = torch.LongTensor([[0],[1],[2],[5],[8],[7],[6],[3],[4]]).unsqueeze(0).repeat(1,1, w*h).to(device)

    shift_window = torch.nn.Unfold(3,1,1,1)
    ats = shift_window(skeleton_t)

    helos = ats.gather(1, helo_index)[:,:-1,:]

    helos -= torch.roll(helos, 1, dims=1)
    helos[helos == -1] = 0
    helos_ct =  torch.sum(helos, 1).view(*skeleton.shape)

    cross_point_map[helos_ct >= 3] = 1 

    window_sum = torch.sum(ats, 1).view([skeleton_t.shape[-2],skeleton_t.shape[-1]])
    window_sum[skeleton_t[0,0,:,:]!=1] = 0

    cross_point_map[window_sum == 2] = 2
    cross_point_map[skeleton_t[0,0,:,:] == 0] = 0
    
    return cross_point_map.detach().cpu().numpy()