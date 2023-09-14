import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.compresion import *


class clientCE(Client):
    def __init__(self, args, id, train_samples, test_samples, ckks, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate)

        self.init = False
        self.r = args.r  
        self.ckks_tools = ckks

        self.alpha = 1
        self.mu = 0.001

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        # if self.init:
        #     self.compressed_model.package_decompresion(self.r)
        #     self.compressed_model.unpack(self.model,self.device)

        # self.init = True
        # self.compressed_model.package_de(self.ckks_tools)
        if self.compressed_model.is_Compressed is True:
            self.compressed_model.package_decompresion(self.r)
        self.compressed_model = copy.deepcopy(
            self.compressed_model.unpack(copy.deepcopy(self.model), self.device))

        # self.model.to(self.device)
        self.model.train()

        fixed_model = copy.deepcopy(self.model)

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                ce_loss = self.loss(output, y)

                reg_loss = 0
                fixed_params = {n: p for n, p in fixed_model.named_parameters()}
                for n, p in self.model.named_parameters():
                    reg_loss += ((p - fixed_params[n].detach()) ** 2).sum()

                loss = self.alpha * ce_loss + 0.5 * self.mu * reg_loss
                loss.backward()
                self.optimizer.step()

        self.compressed_model = Packages()
        self.compressed_model.pack_up(copy.deepcopy(self.model))
        self.compressed_model.package_compresion(self.r)
        self.compressed_model.package_en(self.ckks_tools)
        

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
