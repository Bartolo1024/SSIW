from torch import nn


class HDLoss(nn.module):
    def __init__(self):
        super.__init__(self, HDLoss)
        self.cosine_similarity = nn.CosineSimilarity()
        self.log_softmax = nn.LogSoftmax()

    def __call__(self, output, target, one_hot_labels):
        output = output
        target = target
        cos_sim = self.cosine_similarity(output, target)
        # TODO cos_sim = cos_sim / temperature
        loss = self.log_softmax(cos_sim) * one_hot_labels
        loss = -loss.mean()
        return loss
