import torch

def chamfer_distance(p, q, return_raw=False):
    """Compute chamfer distance between two point clouds.
    
    Parameters
    ----------
    p : Tensor of shape [N, C] or [B, N, C]
        First point cloud
    
    q : Tensor of shape [M, C] or [B, M, C]
        Second point cloud
    
    return_raw : bool, default=False
        Whether to return raw distances and indices
    
    Returns
    -------
    cd_p : Tensor of shape [B]
        Chamfer distance with sqrt
    cd_t : Tensor of shape [B]
        Chamfer distance without sqrt
    p_min : Tensor of shape [B, N], optional
        Minimum distances from p to q
    q_min : Tensor of shape [B, M], optional
        Minimum distances from q to p
    p_idx : Tensor of shape [B, N], optional
        Indices of nearest points in q for each point in p
    q_idx : Tensor of shape [B, M], optional
        Indices of nearest points in p for each point in q
    """
    # Ensure inputs are 3D tensors
    if len(p.shape) == 2:
        p = p.unsqueeze(0)
    if len(q.shape) == 2:
        q = q.unsqueeze(0)
    
    # Compute pairwise distances
    pq_dists = torch.cdist(p, q)
    pq_dists = torch.clamp(pq_dists, min=1e-8)
    
    # Find minimum distances and indices
    p_min, p_idx = torch.min(pq_dists, dim=2)  # Min dist from each p to any q
    q_min, q_idx = torch.min(pq_dists, dim=1)  # Min dist from each q to any p
    
    # Compute Chamfer distances
    cd_t = p_min.mean(dim=-1) + q_min.mean(dim=-1)  # Original Chamfer distance
    cd_p = torch.sqrt(p_min).mean(dim=-1) + torch.sqrt(q_min).mean(dim=-1)  # With sqrt
    
    if return_raw:
        return cd_p, cd_t, p_min, q_min, p_idx, q_idx
    return cd_p, cd_t

def chamfer_dist(output, gt, return_raw=False):
    """Alias for chamfer_distance with swapped arguments.
    
    Parameters
    ----------
    output : Tensor of shape [B, N, C]
        Output point cloud
    
    gt : Tensor of shape [B, M, C]
        Ground truth point cloud
    
    return_raw : bool, default=False
        Whether to return raw distances and indices
    
    Returns
    -------
    cd_p : Tensor of shape [B]
        Chamfer distance with sqrt
    cd_t : Tensor of shape [B]
        Chamfer distance without sqrt
    dist1, dist2, idx1, idx2 : optional
        Raw distances and indices
    """
    if return_raw:
        cd_p, cd_t, dist2, dist1, idx2, idx1 = chamfer_distance(gt, output, return_raw=True)
        return [cd_p, cd_t, dist1, dist2, idx1, idx2]
    else:
        cd_p, cd_t = chamfer_distance(gt, output, return_raw=False)
        return [cd_p, cd_t]

def density_chamfer_dist(
    x, gt, alpha=200, n_lambda=0.5, return_raw=False, non_reg=False
):
    """Compute density-aware chamfer distance between two point clouds.
    Based on https://github.com/wutong16/Density_aware_Chamfer_Distance
    
    Parameters
    ----------
    x : Tensor of shape [B, N, C]
        Output point cloud
    
    gt : Tensor of shape [B, M, C]
        Ground truth point cloud
    
    alpha : float, default=200
        Parameter for exponential weighting
    
    n_lambda : float, default=0.5
        Parameter for density normalization
    
    return_raw : bool, default=False
        Whether to return raw distances and indices
    
    non_reg : bool, default=False
        Whether to use non-regularized version
    
    Returns
    -------
    loss : Tensor of shape [B]
        Density-aware chamfer distance
    cd_p : Tensor of shape [B]
        Chamfer distance with sqrt
    cd_t : Tensor of shape [B]
        Chamfer distance without sqrt
    dist1, dist2, idx1, idx2 : optional
        Raw distances and indices
    """
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    batch_size, n_gt, _ = gt.shape
    assert x.shape[0] == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = chamfer_distance(x, gt, return_raw=True)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    loss = (loss1 + loss2) / 2

    res = (loss, cd_p, cd_t)
    if return_raw:
        res = res + (dist1, dist2, idx1, idx2)
    return res