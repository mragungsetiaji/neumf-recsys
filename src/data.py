import pandas as pd
from torch.utils.data import Dataset

class UserItemRatingDataset(Dataset):
    """
    Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset
    """

    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair

        """
        self.user_tensor   = user_tensor
        self.item_tensor   = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return (self.user_tensor[index], 
                self.item_tensor[index],
                self.target_tensor[index])

    def __len__(self):
        return  self.user_tensor.size(0)

class SampleGenerator(object):
    """
    Construct dataset for NCF (Neural Network Collaborative Filtering)
    """

    def __init__(self, ratings):
        """
        args:
            ratings: pd.DataFrame, contains 4 columns = ["userId", "itemId",
                                                         "rating", "timestamp"]
        """
        assert "userId" in ratings.columns
        assert "itemId" in ratings.columns
        assert "rating" in ratings.columns

        self.ratings = ratings
        self.normalize_ratings = self._normalize(ratings)

        self.user_pool = set(self.ratings["userId"].unique())
        self.item_pool = set(self.ratings["itemId"].unique())

        # Create negative item samples for NCF learning
        self.negatives                        = self._sample_negative(ratings)
        self.train_ratings, self.test_ratings = self._split_loo(self.normalize_ratings)