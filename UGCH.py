import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import datasets
import os.path as osp
from models import ImgNet, TxtNet, AttImgNet, AttTexNet
from utils import compress, calculate_top_map


class UGCH:
    def __init__(self, log, config):
        self.logger = log
        self.config = config

        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.cuda.set_device(self.config.GPU_ID)

        if self.config.DATASET == "MIRFlickr":
            self.train_dataset = datasets.MIRFlickr(train=True, transform=datasets.mir_train_transform)
            self.test_dataset = datasets.MIRFlickr(train=False, database=False, transform=datasets.mir_test_transform)
            self.database_dataset = datasets.MIRFlickr(train=False, database=True, transform=datasets.mir_test_transform)

        if self.config.DATASET == "NUSWIDE":
            self.train_dataset = datasets.NUSWIDE(train=True, transform=datasets.nus_train_transform)
            self.test_dataset = datasets.NUSWIDE(train=False, database=False, transform=datasets.nus_test_transform)
            self.database_dataset = datasets.NUSWIDE(train=False, database=True, transform=datasets.nus_test_transform)

        # Data Loader (Input Pipeline)
        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=self.config.BATCH_SIZE,
                                       shuffle=True,
                                       num_workers=self.config.NUM_WORKERS,
                                       drop_last=True)

        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.config.BATCH_SIZE,
                                      shuffle=False,
                                      num_workers=self.config.NUM_WORKERS)

        self.database_loader = DataLoader(dataset=self.database_dataset,
                                          batch_size=self.config.BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=self.config.NUM_WORKERS)

        self.ImgNet = ImgNet(code_len=self.config.HASH_BIT)

        txt_feat_len = datasets.txt_feat_len

        self.TxtNet = TxtNet(code_len=self.config.HASH_BIT, txt_feat_len=txt_feat_len)

        self.AttImgNet = AttImgNet()
        self.AttTexNet = AttTexNet(txt_feat_len=txt_feat_len)

        self.opt_I = torch.optim.SGD(self.ImgNet.parameters(), lr=self.config.LR_IMG, momentum=self.config.MOMENTUM,
                                      weight_decay=self.config.WEIGHT_DECAY)
        #self.opt_I = torch.optim.Adam(self.CodeNet_I.parameters(), lr=self.config.LR_IMG, weight_decay=self.config.WEIGHT_DECAY)
        self.opt_T = torch.optim.SGD(self.TxtNet.parameters(), lr=self.config.LR_TXT, momentum=self.config.MOMENTUM,
                                      weight_decay=self.config.WEIGHT_DECAY)
        #self.opt_T = torch.optim.Adam(self.CodeNet_T.parameters(), lr=self.config.LR_TXT, weight_decay=self.config.WEIGHT_DECAY)

        self.opt_AI = torch.optim.Adam(self.AttImgNet.parameters(), lr=self.config.LR_IMG, weight_decay=self.config.WEIGHT_DECAY)

        self.opt_AT = torch.optim.Adam(self.AttTexNet.parameters(), lr=self.config.LR_TXT, weight_decay=self.config.WEIGHT_DECAY)

        self.best_it = 0
        self.best_ti = 0

    def train(self, epoch):

        self.ImgNet.set_alpha(epoch)
        self.TxtNet.set_alpha(epoch)

        self.ImgNet.cuda().train()
        self.TxtNet.cuda().train()
        self.AttImgNet.cuda().train()
        self.AttTexNet.cuda().train()



        for idx, (img, txt, _, _) in enumerate(self.train_loader):
            img = torch.FloatTensor(img).cuda()
            txt = torch.FloatTensor(txt.numpy()).cuda()

            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            self.opt_AI.zero_grad()
            self.opt_AT.zero_grad()

            F_I, code_I = self.ImgNet(img)
            F_T, code_T = self.TxtNet(txt)
            G_I = self.AttImgNet(F_I)
            G_T = self.AttTexNet(F_T)

            S = self.cal_similarity_matrix(F_I, F_T, G_I, G_T)

            loss = self.cal_loss(code_I, code_T, S)

            loss.backward()

            self.opt_I.step()
            self.opt_T.step()
            self.opt_AI.step()
            self.opt_AT.step()

            if (idx + 1) % (len(self.train_dataset) // self.config.BATCH_SIZE / self.config.EPOCH_INTERVAL) == 0:
                self.logger.info('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                                 % (epoch + 1, self.config.NUM_EPOCH, idx + 1, len(self.train_dataset) // self.config.BATCH_SIZE,
                                     loss.item()))


    def eval(self):

        self.logger.info('--------------------Evaluation: mAP@5000-------------------')

        self.ImgNet.cuda().eval()
        self.TxtNet.cuda().eval()

        re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L = compress(self.database_loader, self.test_loader, self.ImgNet,
                                                          self.TxtNet, self.database_dataset, self.test_dataset)

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)

        if (self.best_it + self.best_ti) < (MAP_I2T + MAP_T2I):
            self.best_it = MAP_I2T
            self.best_ti = MAP_T2I

        self.logger.info('mAP@50 I->T: %.4f, mAP@50 T->I: %.4f' % (MAP_I2T, MAP_T2I))
        self.logger.info('Best MAP of I->T: %.4f, Best mAP of T->I: %.4f' % (self.best_it, self.best_ti))
        self.logger.info('--------------------------------------------------------------------')


    def cal_similarity_matrix(self, F_I, F_T, G_I, G_T):

        F_I = F.normalize(F_I, dim=1)
        S_I = F_I.mm(F_I.t())
        F_T = F.normalize(F_T, dim=1)
        S_T = F_T.mm(F_T.t())

        G_I = F.normalize(G_I, dim=1)
        A_I = G_I.mm(G_I.t())
        G_T = F.normalize(G_T, dim=1)
        A_T = G_T.mm(G_T.t())

        S_fusion = self.config.gamma * (S_I.mm(A_I.t())) / self.config.BATCH_SIZE + (1 - self.config.gamma) * (S_T.mm(A_T.t())) / self.config.BATCH_SIZE

        S = self.config.lamb * S_fusion + (1 - self.config.lamb) * S_fusion.mm(S_fusion.t()) / self.config.BATCH_SIZE

        # S_high = F.normalize(S_I).mm(F.normalize(S_T).t())
        # S_ = self.config.alpha * S_I + self.config.beta * S_T + self.config.gamma * (S_high + S_high.t()) / 2
        # S = S_  * self.config.eta

        return S

    def cal_loss(self, code_I, code_T, S):

        B_I = F.normalize(code_I, dim=1)
        B_T = F.normalize(code_T, dim=1)

        BI_BI = B_I.mm(B_I.t())
        BT_BT = B_T.mm(B_T.t())
        BI_BT = B_I.mm(B_T.t())
        # BT_BI = B_T.mm(B_I.t())

        # loss1 = F.mse_loss(BI_BI, S)
        # loss2 = F.mse_loss(BI_BT, S) + F.mse_loss(BT_BI, S) -(B_I * B_T).sum(dim=1).mean()
        # loss3 = F.mse_loss(BT_BT, S)
        # loss = self.config.lamb * loss1 + loss2 + self.config.lamb * loss3
        loss1 = F.mse_loss(BI_BT, S)
        loss2 = F.mse_loss(BI_BI, S)
        loss3 = F.mse_loss(BT_BT, S)

        loss = self.config.alpha * loss1 + self.config.beta * loss2 + self.config.mu * loss3

        return loss

    def save_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        obj = {
            'ImgNet': self.ImgNet.state_dict(),
            'TxtNet': self.TxtNet.state_dict(),
        }
        torch.save(obj, ckp_path)
        self.logger.info('**********Save the trained model successfully.**********')


    def load_checkpoints(self, file_name='latest.pth'):
        ckp_path = osp.join(self.config.MODEL_DIR, file_name)
        try:
            obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
            self.logger.info('**************** Load checkpoint %s ****************' % ckp_path)
        except IOError:
            self.logger.error('********** Fail to load checkpoint %s!*********' % ckp_path)
            raise IOError

            self.ImgNet.load_state_dict(obj['ImgNet'])
            self.TxtNet.load_state_dict(obj['TxtNet'])







