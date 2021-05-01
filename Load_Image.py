# 数据读取
# 对每个样本赋予一个标签
import os.path
import cv2


class LoadImage:

    def __init__(self, root_path="E:\graduate\QBroi\ROIs", rate=0.7):
        self.root_path = root_path
        self.rate = rate
        # 存放数据地址和标签
        self.data = []
        self.label = []

    def load_image(self):
        labels = 0

        for person in os.listdir(self.root_path):
            dir_finger = self.root_path + '\\' + person
            # 访问手指
            for finger in os.listdir(dir_finger):
                # 将每个手指作为一个个体
                dir_filename = dir_finger + '\\' + finger
                # 删除文件中的DS_store
                # if finger==".DS_Store":
                # dir_filename=dir_finger+'\\'+finger
                # os.remove(dir_filename)
                # 访问每个文件
                for filename in os.listdir(dir_filename):
                    # 每个文件的路径名
                    self.data.append(dir_filename + '\\' + filename)
                    self.label.append(labels)
                labels += 1

    # 数据集划分
    def split_data(self):
        # 导入数据
        self.load_image()

        train_data = []
        train_label = []
        test_data = []
        test_label = []

        sample_num = 10
        count = 0
        train_num = sample_num * self.rate

        for new_data, new_label in zip(self.data, self.label):
            if count % 10 < train_num:
                train_data.append(new_data)
                train_label.append(new_label)
            else:
                test_data.append(new_data)
                test_label.append(new_label)
            count += 1
        return train_data, train_label, test_data, test_label
