import torch
import os
import csv
import time
import logging
import warnings
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from utils.print_args import print_args
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from nets.SiameseTimesNet import SiameseNetwork

# 忽略警告信息
warnings.filterwarnings('ignore')

class Exp_SiameseClassification(Exp_Basic):
    def __init__(self, args):
        self.result_file = os.path.join(args.checkpoints, 'results', 'result.csv')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(Exp_SiameseClassification, self).__init__(args)
        # 确保结果文件夹存在
        os.makedirs(os.path.join(args.checkpoints, 'results'), exist_ok=True)
        self.model = self._build_model()
        self.model.to(self.device)  # 将模型移动到设备


    # def _get_data(self, flag):
    #     """
    #     获取数据集和数据加载器，并根据需要划分为训练集和验证集。
    #     """
    #     data_set, _ = data_provider(self.args, flag)

    #     if flag == 'TRAIN':
    #         # 按照比例划分训练集和验证集
    #         train_size = int(0.8 * len(data_set))  # 80% 的数据作为训练集
    #         vali_size = len(data_set) - train_size  # 剩余数据作为验证集

    #         train_set, vali_set = random_split(data_set, [train_size, vali_size], generator=torch.Generator().manual_seed(42))

    #         # 创建数据加载器
    #         train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
    #         vali_loader = DataLoader(vali_set, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

    #         return train_set, train_loader, vali_set, vali_loader

    #     elif flag == 'TEST':
    #         test_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
    #         return data_set, test_loader


    # def _build_model(self):
    #     """
    #     构建Siamese网络模型并设置相关参数。
    #     """
    #     self.args.num_class = 50

    #     # 获取数据集
    #     train_data, train_loader, vali_data, vali_loader = self._get_data(flag='TRAIN')
    #     test_data, test_loader = self._get_data(flag='TEST')

    #     # 从原始数据集中获取最大序列长度和特征维度
    #     original_train_data = train_data.dataset if isinstance(train_data, torch.utils.data.Subset) else train_data
    #     original_test_data = test_data.dataset if isinstance(test_data, torch.utils.data.Subset) else test_data

    #     self.args.seq_len = max(original_train_data.max_seq_len, original_test_data.max_seq_len)
    #     self.args.pred_len = 0
    #     self.args.enc_in = original_train_data.feature_df.shape[1]

    #     # 构建模型
    #     model = SiameseNetwork(self.args).float()
    #     if torch.cuda.device_count() > 1:
    #         model = nn.DataParallel(model)
    #     return model

    def _build_model(self):
        """
        构建Siamese网络模型并设置相关参数。
        """
        self.args.num_class = 50
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        model = SiameseNetwork(self.args).float()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        选择Adam优化器。
        """
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        选择BCEWithLogitsLoss损失函数。
        """
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def train(self, setting):
        """
        模型训练过程。
        """
        # 创建并配置日志记录器
        logging.basicConfig(
            filename=os.path.join(self.args.checkpoints, 'results', 'training_log.txt'), 
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger()

        # 加载训练数据和验证数据
        train_data, train_loader = self._get_data(flag='TRAIN')
        

        # 设置检查点保存路径
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        # 加载检查点
        checkpoint_path = os.path.join(self.args.checkpoints, 'results', 'checkpoint.pth')
        start_epoch = 0
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            model_optim = self._select_optimizer()
            model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            iter_count = checkpoint.get('iter_count', 0)
            print(f"Loaded checkpoint from epoch {start_epoch}")
        else:
            model_optim = self._select_optimizer()
            iter_count = 0
            print("No checkpoint found, starting training from scratch...")

        # 获取当前时间
        time_now = time.time()
        train_steps = len(train_loader)

        # 初始化早停机制
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        criterion = self._select_criterion()

        # 设置学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=10, eta_min=1e-5)

        for epoch in range(start_epoch, self.args.train_epochs):
            train_loss = []
            correct_predictions = 0  # 累计正确预测的样本数
            total_predictions = 0    # 累计预测的样本数

            self.model.train()
            for i, ((batch_x1, label1, mask1), (batch_x2, label2, mask2)) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # 将数据移动到设备
                batch_x1 = batch_x1.float().to(self.device)
                batch_x2 = batch_x2.float().to(self.device)
                label1 = label1.to(self.device)
                label2 = label2.to(self.device)
                mask1 = mask1.to(self.device)
                mask2 = mask2.to(self.device)

                # 前向传播
                outputs = self.model(batch_x1, batch_x2, mask1, mask2).squeeze()
                labels = (label1 == label2).float()  # 正样本对标签为1，负样本对标签为0

                # 计算损失
                loss = criterion(outputs, labels)
                train_loss.append(loss.item())

                # 计算预测准确率
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                # 反向传播和优化
                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print(f"Epoch: {epoch+1}, Step: {i+1}, Loss: {loss.item():.4f}")

            # 计算平均训练损失和准确率
            epoch_loss = np.mean(train_loss)
            epoch_accuracy = correct_predictions / total_predictions

            print(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")
            logger.info(f"Epoch {epoch+1} | Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")

            # 验证模型
            # vali_loss, vali_accuracy = self.vali(vali_data, vali_loader, criterion)
            # print(f"Validation Loss: {vali_loss:.4f}, Validation Accuracy: {vali_accuracy:.4f}")
            # logger.info(f"Validation Loss: {vali_loss:.4f}, Validation Accuracy: {vali_accuracy:.4f}")

            # 保存检查点
            save_checkpoint(self.model, model_optim, epoch, iter_count, checkpoint_path, logger)

            # # 检查早停
            # early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping triggered.")
            #     break

            # 更新学习率
            scheduler.step()

        # 加载最佳模型
        best_model_path = os.path.join(self.args.checkpoints, 'results', 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model


    
    # def vali(self, vali_data, vali_loader, criterion):
    #     """
    #     模型验证过程。
    #     """
    #     total_loss = []
    #     correct_predictions = 0  # 正确预测的总数
    #     total_predictions = 0    # 总的样本数

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, ((batch_x1, label1, mask1), (batch_x2, label2, mask2)) in enumerate(vali_loader):
    #             # 将数据移动到 GPU（如果可用）
    #             batch_x1 = batch_x1.float().to(self.device)
    #             batch_x2 = batch_x2.float().to(self.device)
    #             label1 = label1.to(self.device)
    #             label2 = label2.to(self.device)
    #             mask1 = mask1.to(self.device)
    #             mask2 = mask2.to(self.device)
                
    #             # 正样本对的前向传播
    #             outputs = self.model(batch_x1, batch_x2, mask1, mask2)
    #             outputs = outputs.squeeze()
    #             labels = torch.ones_like(outputs, device=self.device)  # 正样本对标签为1
                
    #             # 计算损失
    #             loss = criterion(outputs, labels)
    #             total_loss.append(loss.item())
                
    #             # 计算准确率
    #             predictions = (torch.sigmoid(outputs) > 0.5).float()
    #             correct_predictions += torch.sum(predictions == labels).item()
    #             total_predictions += labels.size(0)

    #     # 计算平均验证损失和准确率
    #     total_loss = np.average(total_loss)
    #     total_accuracy = correct_predictions / total_predictions  # 总体准确率

    #     self.model.train()
    #     return total_loss, total_accuracy

    def test(self, setting):
        """
        测试阶段：从训练集中选择一个样本，与测试集中的所有样本生成正样本对和负样本对，计算准确率。
        """
        # 创建并配置日志记录器
        logging.basicConfig(filename=os.path.join(self.args.checkpoints, 'results', 'test_log.txt'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger()

        # 加载训练数据和测试数据
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')

        # 构建检查点路径
        checkpoint_path = os.path.join(self.args.checkpoints, 'results', 'checkpoint.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点未找到：{checkpoint_path}")

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise KeyError("检查点中未找到 model_state_dict。")

        # 设置测试结果保存目录
        folder_path = os.path.join(self.args.checkpoints, 'results', 'test_results', setting)
        os.makedirs(folder_path, exist_ok=True)

        # 模型评估模式
        self.model.to(self.device).eval()

        preds = []
        trues = []

        with torch.no_grad():
            # 从训练集中随机选择一个样本
            for train_idx, ((train_sample, train_label, train_mask),_) in enumerate(train_loader):
                train_sample = train_sample.float().to(self.device)
                train_mask = train_mask.to(self.device)
                train_label = train_label.to(self.device)

                sample_preds = []
                sample_trues = []

                # 遍历测试集中的样本
                for test_idx, ((test_sample, test_label, test_mask),_) in enumerate(test_loader):
                    test_sample = test_sample.float().to(self.device)
                    test_mask = test_mask.to(self.device)
                    test_label = test_label.to(self.device)

                    # 前向传播，计算相似性分数
                    output = self.model(train_sample, test_sample, train_mask, test_mask)
                    output = output.squeeze()

                    # 标签：1 表示同类，0 表示不同类
                    expected_label = (train_label == test_label).float()
                    sample_trues.append(expected_label.cpu().numpy())
                    sample_preds.append(torch.sigmoid(output).cpu().numpy())

                # 记录当前训练样本的测试结果
                preds.append(np.concatenate(sample_preds))
                trues.append(np.concatenate(sample_trues))

        # 转换为 NumPy 数组
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        # 计算准确率
        predictions = (preds > 0.5).astype(np.float32)
        accuracy = np.mean(predictions == trues)

        # 保存测试结果
        logger.info('accuracy: {}'.format(accuracy))
        print('accuracy: {}'.format(accuracy))
        save_test_results(setting, accuracy, folder_path)

        return accuracy


# 保存函数
def save_checkpoint(model, optimizer, epoch, iter_count, checkpoint_path, logger):
    """
    保存模型检查点。
    """
    if isinstance(model, nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'iter_count': iter_count,
    }, checkpoint_path)
    logger.info(f"Checkpoint saved at epoch {epoch}, iter {iter_count}")
    print(f"Checkpoint saved at epoch {epoch}, iter {iter_count}")
    
def save_results(epoch, train_loss, train_accuracy, vali_loss, val_accuracy, result_file):
    """
    保存训练和验证结果。
    """
    with open(result_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'vali_loss', 'val_accuracy'])  # 添加表头
        writer.writerow([epoch + 1, train_loss, train_accuracy, vali_loss, val_accuracy])

def save_test_results(setting, accuracy, folder_path):
    """
    保存测试结果。
    """
    file_name = 'result_classification.txt'
    with open(os.path.join(folder_path, file_name), 'a') as f:
        f.write(setting + "  \n")
        f.write('accuracy: {}'.format(accuracy))
        f.write('\n\n')
