import torch

def get_cluster_data_set(num_samples=1000, center_gap=9):
    # Sample from 2 dimensinal normal distribution
    X = torch.normal(mean=0, std=1, size=(num_samples,2)) - 0.5 * center_gap

    # Sample cluster ids randomly from {0,1,2,3}
    cluster_id = torch.randint(low=0,high=4, size=(num_samples,))

    # Transform data point to receive four clusters
    #      odd cluster_id   -->  shift x value by center_gap
    #      cluster_id >= 2  -->  shift y value by center_gap
    X = torch.stack([X[:,0] + center_gap * (cluster_id % 2), 
                    X[:,1] + center_gap * (cluster_id // 2)], axis=-1);

    # Map cluster_ids to class labels: 
    #      cluster_ids 0 and 3 --> 0
    #      cluster_ids 1 and 4 --> 1
    Y = torch.where((cluster_id == 0) + (cluster_id == 3) > 0, 
                    torch.zeros_like(cluster_id), 
                    torch.ones_like(cluster_id))
    Y = Y.unsqueeze(dim=-1)

    return X,Y