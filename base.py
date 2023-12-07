import csv, torch, os
import glob, random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, _mutual_info, f_regression


# best_mRMR_score, best_id, select_index = 0, 0, 0
class mRMR_feature():
    def __init__(self, k):
        self.k = k

    def data_load(self):
        # 각 class feature(training set)
        bn = np.load('/home/compu/LJC/data/voting/features/class_0.npy')
        wd = np.load('/home/compu/LJC/data/voting/features/class_1.npy')
        md = np.load('/home/compu/LJC/data/voting/features/class_2.npy')
        pd = np.load('/home/compu/LJC/data/voting/features/class_3.npy')

        # 각 class feature id(training set)
        bn_id = np.load('/home/compu/LJC/data/voting/features/class_id_0.npy')
        wd_id = np.load('/home/compu/LJC/data/voting/features/class_id_1.npy')
        md_id = np.load('/home/compu/LJC/data/voting/features/class_id_2.npy')
        pd_id = np.load('/home/compu/LJC/data/voting/features/class_id_3.npy')

        # return bn, wd, md, pd, bn_id, wd_id, md_id, pd_id

        class_data = np.concatenate((bn, wd, md, pd))
        class_id = np.concatenate((bn_id, wd_id, md_id, pd_id))

        target = np.zeros((4 * self.k))

        for i in range(4):
            target[i * self.k: (i + 1) * self.k] = i + 1

        # First redundancy(simlarity matrix row_sum)
        similarity_matrix = np.load('/home/compu/LJC/data/voting/features/similarity_matrix.npy')

        # 전체 sample에 대한 redundancy score
        redundancy_score = np.load('/home/compu/LJC/data/voting/features/similarity_score.npy')

        return class_data, class_id, target, similarity_matrix, redundancy_score

    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum(((x / 1024) - (y / 1024)) ** 2))

    def similarity_matrix(self, data, sigma=0.1):
        size = len(data)

        similarity_pixel_raw_sum = np.zeros((size, 1))

        similarity_pixel = np.zeros((size, size))
        for y in range(size):
            for x in range(size):
                if y != x:
                    similarity_pixel[y, x] = np.exp(-(self.euclidean_distance(data[y], data[x]) / sigma))

            similarity_pixel_raw_sum[y] = np.sum(similarity_pixel[y]) / (size - 1)

        # for i in range(size):
        #     similarity_pixel_raw_sum[i] = np.sum(similarity_pixel[i]) / (size - 1)

        return similarity_pixel, similarity_pixel_raw_sum

    def update_redundancy_score(self, similarity_matrix, choiced_index, big_redundancy_index, index, k):

        index = index.reshape(4 * k)
        change_index = np.delete(index, big_redundancy_index)
        change_similarity_matrix = np.sum(similarity_matrix[int(choiced_index)][change_index]) / (len(index) - 1)
        index = np.insert(change_index, big_redundancy_index, int(choiced_index))
        return change_similarity_matrix, index


    def new_similarity(self, similarity_score, index, k):

        # data_length = [0, 773, 773+1866, 773+1866+2997]
        data_length = [0, 773, 773 + 1035, 773 + 1035 + 1903]
        for i in range(4):
            index[i] = index[i] + data_length[i]
        index = index.reshape(4 * k)
        local_sample_redundancy_score = np.zeros((len(index), 1))
        for i in range(len(index)):
            choiced_index = np.delete(index, i)
            local_sample_redundancy_score[i] = np.sum(similarity_score[index[i]][choiced_index]) / (len(index) - 1)

        return local_sample_redundancy_score

    def lda_redundancy(self, similarity_score, index, k):
        data_length = [0, 773, 773+1866, 773+1866+2997]

        for i in range(4):
            index[i] = index[i] + data_length[i]

        index = index.reshape(4 * k)

        local_sample_lad_redundancy = np.zeros((4, k))

        for num_class in range(4):
            for i in range(k):
                # within_choiced_index = np.delete(index, k*num_class + i)
                within_choiced_index = np.delete(index, np.s_[k*int(i/k):k*(int(i/k) + 1)])
                within_choiced_index = np.delete(within_choiced_index, i)

                # within class
                within_score = np.sum(similarity_score[index[k*num_class + i]][within_choiced_index]) / (len(index) - 1)

                between_choiced_index = np.delete(index, np.s_[k*int(i/k):k*(int(i/k) + 1)])

                # between class
                between_score = np.sum(
                    similarity_score[index[k*num_class + i]][between_choiced_index]) / (
                                             len(between_choiced_index) - 1)

                local_sample_lad_redundancy[num_class, i] = between_score / (within_score + 1)

        return local_sample_lad_redundancy.reshape(4 * k), between_score, within_score

    def mutual_information_classif(self, values_x, values_y):

        mi_value = 0.0
        # discrete_features: 'auto' or [False]: continuous and discrete variable MI to determine
        # mutual_info_classif: target is discrete & regression: target is continuous
        mi_value = mutual_info_classif(values_x.reshape(-1, 1), values_y, discrete_features=[False], n_neighbors=3,
                               copy=True, random_state=None)

        return mi_value
    def mutual_information(self, x_arr, y_arr, log_base=2):

        # Variable to return MI
        mi_value = 0.0

        # values_x = set(x_arr)
        # values_y = set(y_arr)
        # For each random
        for value_x in x_arr:
            for value_y in y_arr:
                px = np.shape(np.where(x_arr == value_x))[1] / len(x_arr)
                py = np.shape(np.where(y_arr == value_y))[1] / len(y_arr)
                pxy = len(np.where(np.in1d(np.where(x_arr == value_x)[0],
                                     np.where(y_arr == value_y)[0]) == True)[0]) / len(y_arr)
                if pxy > 0.0:
                    mi_value += pxy * np.math.log((pxy / (px * py)), log_base)
        return mi_value

    # Updata 시 각 class마다 하나씩 넣어서 비교해보기
    # class_cnt: 남은 class 데이터 수 & class_num: 0, 1, 2, 3 의미
    redundancy_data, feature_data, id, zero_id = 0, 0, 0, 0

    def Updata(self, start_class_cnt, end_class_cnt, target, fist_total_score, k, choice_redundancy_data,
               choice_feature_data, choice_id, select_index, remaining_index, similarity_score, best_redundancy,
               remaining_redundancy_data, remaining_feature_data, remaining_id, break_true, best_mRMR_score, best_id):

        global redundancy_data, feature_data, id, zero_id
        redundancy_data, feature_data, id, zero_id = choice_redundancy_data, choice_feature_data, choice_id, choice_id
        terminate_cnt = 0
        class_num2 = 0
        tt = 0
        while True:
            for class_num in range(4):

                choice_redundancy_data, choice_feature_data, choice_id = redundancy_data, feature_data, id
                not_pass_cnt = 0
                # 가장 큰 redundancy 빼기
                big_redundancy_index = np.argsort(choice_redundancy_data[class_num * k: (class_num + 1) * k], 0)[-1]
                big_redundancy_index = big_redundancy_index + (class_num * k)
                best_id = big_redundancy_index


                for cnt in start_class_cnt[class_num][class_num2]:
                    tt += 1
                    terminate_cnt += 1
                    choice_redundancy_data = np.delete(choice_redundancy_data, big_redundancy_index)
                    choice_feature_data = np.delete(choice_feature_data, big_redundancy_index, axis=0)
                    choice_id = np.delete(choice_id, big_redundancy_index)

                    # 빠진 sample에 대해 나머지 sample 끼리의 redundancy score 다시 구하여 채우기
                    new_redundancy, _ = self.update_redundancy_score(similarity_score, remaining_index[cnt],
                                                                  big_redundancy_index, select_index, k)
                    select_index = _
                    # 빠진 index 정보 remaining 배열에 다시 넣기 X: 이유 하나씩 비교 하기 위해
                    # 그리고 빠진 것 채우기 위해 나머지 각 class remaining 배열 하나씩 넣어서 비교함
                    # 그 중 가장 score가 큰 것 선정함(id 저장하기: 채우기 위해)
                    # choice_redundancy_data = np.insert(choice_redundancy_data, class_num*k, remaining_redundancy_data[cnt])
                    choice_redundancy_data = np.insert(choice_redundancy_data, big_redundancy_index,
                                                       new_redundancy)
                    choice_feature_data = np.insert(choice_feature_data, big_redundancy_index, remaining_feature_data[cnt], axis = 0)
                    choice_id = np.insert(choice_id, big_redundancy_index, remaining_id[cnt])

                    best_id = cnt

                    if cnt == 2474:
                        break

                    if terminate_cnt == 1:
                        terminate_cnt = 0
                        break


                    # # redundancy가 0.5이상인 경우 pass
                    # if (choice_redundancy_data < 0.5).all():
                    #     # relevancy_score & redundancy_score 구하기
                    #     not_pass_cnt += 1
                    #     relevance_score = 0
                    #     for ii in range(1024):
                    #         relevance_score += mRMR.mutual_information_classif(choice_feature_data[:, ii], target)
                    #     relevance_score = relevance_score / 1024
                    #
                    #     # new_tatal_score = relevance_score - (np.sum(choice_redundancy_data)/(4*k))
                    #     new_tatal_score = relevance_score - (np.sum(choice_redundancy_data) / (4*k))
                    #
                    #     # 선택된 sample
                    #     if best_mRMR_score < new_tatal_score:
                    #         best_mRMR_score = new_tatal_score
                    #         best_id = cnt
                    #         best_redundancy = new_redundancy


                print(best_id, not_pass_cnt)

                if best_id != big_redundancy_index:
                    # 가장 큰 score 일 때 sample 정보
                    class_wise_big_smaple_info = np.zeros((2))
                    class_wise_big_smaple_info[0] = redundancy_data[big_redundancy_index]
                    class_wise_big_smaple_info[1] = id[big_redundancy_index]
                    class_wise_big_feature_info = feature_data[big_redundancy_index]

                    # index 정리
                    select_index[big_redundancy_index] = remaining_index[best_id]
                    remaining_index[best_id] = big_redundancy_index

                    # 가장 큰 스코어 일때의 샘플 넣기
                    redundancy_data = np.delete(redundancy_data, big_redundancy_index)
                    # 가장 큰 스코어 일 때의 sample 넣기(위에서 구한 저장된 정보 이용하기)
                    redundancy_data = np.insert(redundancy_data, big_redundancy_index, best_redundancy)

                    feature_data = np.delete(feature_data, big_redundancy_index,  axis=0)
                    feature_data = np.insert(feature_data, big_redundancy_index, remaining_feature_data[best_id], axis=0)

                    # 0 일 때의 ID 저장
                    zero_id = id

                    id = np.delete(id, big_redundancy_index)
                    id = np.insert(id, big_redundancy_index, remaining_id[best_id])

                    # 가장 큰 스코어 자리에 전에 있던 가장 큰 redundancy value 넣기
                    remaining_redundancy_data[best_id] = class_wise_big_smaple_info[0]
                    remaining_id[best_id] = class_wise_big_smaple_info[1]
                    remaining_feature_data[best_id] = class_wise_big_feature_info
                if cnt == 2474:
                    break

            class_num2 += 1
                    # # 정렬 이유: 항상 마지막이 가장 큰 수로 만들기 위해(각 class 마다)
                    # big_index = np.argsort(redundancy_data[class_num*k : (class_num+1)*k], 0)
                    # redundancy_data[class_num*k : (class_num+1)*k] = redundancy_data[class_num*k : (class_num+1)*k][big_index]
                    # feature_data[class_num*k : (class_num+1)*k] = feature_data[class_num*k : (class_num+1)*k][big_index]
                    # id[class_num*k : (class_num+1)*k] = id[class_num*k : (class_num+1)*k][big_index]
                    # select_index[class_num * k: (class_num + 1) * k] = select_index[class_num * k: (class_num + 1) * k][
                    # big_index]

                # relevance_score = 0
                # for ii in range(1024):
                #     relevance_score += mRMR.mutual_information_classif(feature_data[:, ii], target)
                # relevance_score = relevance_score / 1024
                #
                # redundancy = np.sum(redundancy_data) / (4 * k)
                # new_tatal_score = relevance_score - redundancy
                #
                # print(relevance_score, redundancy, new_tatal_score, fist_total_score, new_tatal_score - fist_total_score)
                #
                # if np.abs(new_tatal_score - fist_total_score) < 1.0e-5:
                #     break_true = 1
                #
                # elif np.abs(new_tatal_score - fist_total_score) < 1.0e-5 and int(np.abs(new_tatal_score - fist_total_score)) == 0:
                #     id = zero_id
                #     break_true = 1
                #
                # if break_true == 1:
                #     break
                #
                # fist_total_score = new_tatal_score
            if cnt == 2474:
                break_true = 1
            if break_true == 1:
                break

        return id, break_true



if __name__ == '__main__':

    path = f"/home/compu/LJC/data/voting/preds/colon_vit_large_r50_s32_384_1.0_seed_42.npy"
    data = np.load(path)


    from glob import glob
    from sklearn.model_selection import train_test_split

    # def splitting(dataset):  # train val test 80/10/10
    #     train, rest = train_test_split(
    #         dataset, train_size=0.8, shuffle=False, random_state=42
    #     )
    #     valid, test = train_test_split(rest, test_size=0.5, shuffle=False, random_state=42)
    #     return train, valid, test
    #
    # base_path = "/home/compu/LJC/data/prostate_miccai_2019_patches_690_80_step05/"
    # files = glob(f"{base_path}/*/*.jpg")
    #
    # data_class0 = [
    #     data for data in files if int(data.split("_")[-1].split(".")[0]) == 0
    # ]
    # data_class2 = [
    #     data for data in files if int(data.split("_")[-1].split(".")[0]) == 2
    # ]
    # data_class3 = [
    #     data for data in files if int(data.split("_")[-1].split(".")[0]) == 3
    # ]
    # data_class4 = [
    #     data for data in files if int(data.split("_")[-1].split(".")[0]) == 4
    # ]
    #
    # train_data0, validation_data0, test_data0 = splitting(data_class0)
    # train_data2, validation_data2, test_data2 = splitting(data_class2)
    # train_data3, validation_data3, test_data3 = splitting(data_class3)
    # train_data4, validation_data4, test_data4 = splitting(data_class4)


    # data_path = '/data2/jh_data/data/colon/train/*/*.jpg'
    #
    # with open("{0}train_384.csv".format('/home/compu/LJC/data/colon_tma/COLON_MANUAL_512/'), 'w', newline="") as csvfile:
    #     fieldnames = ['img', 'label']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for load in glob.glob(data_path):
    #         class_name = load.split('/')[-2]
    #         if class_name == 'bn':
    #             label = 0
    #         elif class_name == 'wd':
    #             label = 1
    #         elif class_name == 'md':
    #             label = 2
    #         elif class_name == 'pd':
    #             label = 3
    #         writer.writerow({'img': load, 'label': label})
    #
    # data_path = '/data2/jh_data/data/colon/valid/*/*.jpg'
    # with open("{0}valid_384.csv".format('/home/compu/LJC/data/colon_tma/COLON_MANUAL_512/'), 'w', newline="") as csvfile:
    #     fieldnames = ['img', 'label']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for load in glob.glob(data_path):
    #         class_name = load.split('/')[-2]
    #         if class_name == 'bn':
    #             label = 0
    #         elif class_name == 'wd':
    #             label = 1
    #         elif class_name == 'md':
    #             label = 2
    #         elif class_name == 'pd':
    #             label = 3
    #         writer.writerow({'img': load, 'label': label})
    #
    # data_path = '/data2/jh_data/data/colon/test/*/*.jpg'
    # with open("{0}test_384.csv".format('/home/compu/LJC/data/colon_tma/COLON_MANUAL_512/'), 'w', newline="") as csvfile:
    #     fieldnames = ['img', 'label']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for load in glob.glob(data_path):
    #         class_name = load.split('/')[-2]
    #         if class_name == 'bn':
    #             label = 0
    #         elif class_name == 'wd':
    #             label = 1
    #         elif class_name == 'md':
    #             label = 2
    #         elif class_name == 'pd':
    #             label = 3
    #         writer.writerow({'img': load, 'label': label})


    import cv2, pandas as pd

    # train_colon_path = pd.read_csv('/home/compu/LJC/data/colon_tma/COLON_PATCHES_1024/train.csv')
    # cnt = 0
    # for i in range(len(train_colon_path)):
    #     image_path = train_colon_path['path'][i]
    #     img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #     class_kind = int(os.path.basename(image_path).split('_')[-1].split('.jpg')[0])
    #     img_name = os.path.basename(image_path).split('.jpg')[0]
    #     center_crop_img = img[128:896, 128:896]
    #
    #     if class_kind == 0:
    #         class_name = 'bn'
    #     elif class_kind == 1:
    #         class_name = 'wd'
    #     elif class_kind == 2:
    #         class_name = 'md'
    #     elif class_kind == 3:
    #         class_name = 'pd'
    #
    #     for y in range(2):
    #         for x in range(2):
    #             center_crop_img_cut = center_crop_img[y*384:(y+1)*384, x*384:(x+1)*384]
    #             plt.savefig('/data2/jh_data/data/colon/train/{0}/{1}_{2}_{3}.jpg'.format(class_name, img_name, y, x))
    #
    # valid_colon_path = pd.read_csv('/home/compu/LJC/data/colon_tma/COLON_PATCHES_1024/valid.csv')
    # cnt = 0
    # for i in range(len(valid_colon_path)):
    #     image_path = valid_colon_path['path'][i]
    #     img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #     class_kind = int(os.path.basename(image_path).split('_')[-1].split('.jpg')[0])
    #     img_name = os.path.basename(image_path).split('.jpg')[0]
    #     center_crop_img = img[128:896, 128:896]
    #
    #     if class_kind == 0:
    #         class_name = 'bn'
    #     elif class_kind == 1:
    #         class_name = 'wd'
    #     elif class_kind == 2:
    #         class_name = 'md'
    #     elif class_kind == 3:
    #         class_name = 'pd'
    #
    #     for y in range(2):
    #         for x in range(2):
    #             center_crop_img_cut = center_crop_img[y*384:(y+1)*384, x*384:(x+1)*384]
    #             plt.savefig('/data2/jh_data/data/colon/valid/{0}/{1}_{2}_{3}.jpg'.format(class_name, img_name, y, x))
    #
    # test_colon_path = pd.read_csv('/home/compu/LJC/data/colon_tma/COLON_PATCHES_1024/test.csv')
    # cnt = 0
    # for i in range(len(test_colon_path)):
    #     image_path = test_colon_path['path'][i]
    #     img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #     class_kind = int(os.path.basename(image_path).split('_')[-1].split('.jpg')[0])
    #     img_name = os.path.basename(image_path).split('.jpg')[0]
    #     center_crop_img = img[128:896, 128:896]
    #
    #     if class_kind == 0:
    #         class_name = 'bn'
    #     elif class_kind == 1:
    #         class_name = 'wd'
    #     elif class_kind == 2:
    #         class_name = 'md'
    #     elif class_kind == 3:
    #         class_name = 'pd'
    #
    #     for y in range(2):
    #         for x in range(2):
    #             center_crop_img_cut = center_crop_img[y*384:(y+1)*384, x*384:(x+1)*384]
    #             plt.savefig('/data2/jh_data/data/colon/test/{0}/{1}_{2}_{3}.jpg'.format(class_name, img_name, y, x))


    # cv2.imshow(img_file, img)
    # cv2.imwrite(save_file, img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # data1 = pd.read_csv('/data1/ljc/AGGC22/class1.csv')
    # data2 = pd.read_csv('/data1/ljc/AGGC22/class2.csv')
    # data3 = pd.read_csv('/data1/ljc/AGGC22/class3.csv')
    # data4 = pd.read_csv('/data1/ljc/AGGC22/class4.csv')
    # data5 = pd.read_csv('/data1/ljc/AGGC22/class5.csv')
    #
    # data1_train = data1.sample(n=32546, random_state=42)
    # data1_valid = data1.drop(data1_train.index).sample(frac=0.5, random_state=42)
    # data1_test = data1.drop(data1_train.index).drop(data1_valid.index)
    #
    # data2_train = data2.sample(n=22201, random_state=42)
    # data2_valid = data2.drop(data2_train.index).sample(frac=0.5, random_state=42)
    # data2_test = data2.drop(data2_train.index).drop(data2_valid.index)
    #
    # data3_train = data3.sample(n=63161, random_state=42)
    # data3_valid = data3.drop(data3_train.index).sample(frac=0.5, random_state=42)
    # data3_test = data3.drop(data3_train.index).drop(data3_valid.index)
    #
    # data4_train = data4.sample(n=103882, random_state=42)
    # data4_valid = data4.drop(data4_train.index).sample(frac=0.5, random_state=42)
    # data4_test = data4.drop(data4_train.index).drop(data4_valid.index)
    #
    # data5_train = data5.sample(n=24815, random_state=42)
    # data5_valid = data5.drop(data5_train.index).sample(frac=0.5, random_state=42)
    # data5_test = data5.drop(data5_train.index).drop(data5_valid.index)
    #
    # train = pd.concat([data1_train, data2_train, data3_train, data4_train, data5_train])
    # valid = pd.concat([data1_valid, data2_valid, data3_valid, data4_valid, data5_valid])
    # test = pd.concat([data1_test, data2_test, data3_test, data4_test, data5_test])
    #
    # train.to_csv('/data1/ljc/AGGC22/train.csv', sep=',')
    # valid.to_csv('/data1/ljc/AGGC22/valid.csv', sep=',')
    # test.to_csv('/data1/ljc/AGGC22/test.csv', sep=',')
    # data = pd.read_csv('/data1/ljc/AGGC22/data.csv')
    # class1, class2, class3, class4, class5 = [], [], [], [], []
    # class1_label, class2_label, class3_label, class4_label, class5_label = [], [], [], [], []
    #
    # for i in range(len(data)):
    #     if data['label'][i] == 1:
    #         class1.append(data['img'][i])
    #         # class1_label.append(0)
    #     elif data['label'][i] == 2:
    #         class2.append(data['img'][i])
    #         # class2_label.append(1)
    #     elif data['label'][i] == 3:
    #         class3.append(data['img'][i])
    #         # class3_label.append(2)
    #     elif data['label'][i] == 4:
    #         class4.append(data['img'][i])
    #         # class4_label.append(3)
    #     elif data['label'][i] == 5:
    #         class5.append(data['img'][i])
    #         # class5_label.append(4)
    #
    # with open("{0}class1.csv".format('/data1/ljc/AGGC22/'), 'w', newline="") as csvfile:
    #     fieldnames = ['img', 'label']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for num, load in enumerate(class1):
    #         writer.writerow({'img': load, 'label': 1})
    #
    # with open("{0}class2.csv".format('/data1/ljc/AGGC22/'), 'w', newline="") as csvfile:
    #     fieldnames = ['img', 'label']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for num, load in enumerate(class2):
    #         writer.writerow({'img': load, 'label': 2})
    #
    # with open("{0}class3.csv".format('/data1/ljc/AGGC22/'), 'w', newline="") as csvfile:
    #     fieldnames = ['img', 'label']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for num, load in enumerate(class3):
    #         writer.writerow({'img': load, 'label': 3})
    #
    # with open("{0}class4.csv".format('/data1/ljc/AGGC22/'), 'w', newline="") as csvfile:
    #     fieldnames = ['img', 'label']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for num, load in enumerate(class4):
    #         writer.writerow({'img': load, 'label': 0})
    #
    # with open("{0}class5.csv".format('/data1/ljc/AGGC22/'), 'w', newline="") as csvfile:
    #     fieldnames = ['img', 'label']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for num, load in enumerate(class5):
    #         writer.writerow({'img': load, 'label': 0})

    # dir_path = '/data1/ljc/AGGC22/patch_data/Class_corrected/'
    # path_save = []
    # for (root, directories, files) in os.walk(dir_path):
    #     for file in files:
    #         if '.jpg' in file:
    #             file_path = os.path.join(root, file)
    #             path_save.append(file_path)
    #
    #
    # with open("{0}data.csv".format('/data1/ljc/AGGC22/'), 'w', newline="") as csvfile:
    #     fieldnames = ['img', 'label']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for num, load in enumerate(path_save):
    #         class_num = int(os.path.basename(load).split('_')[-1].split('.')[0])
    #         writer.writerow({'img': load, 'label': class_num})

    # for file_path in total_path:
    #     if os.path(file_path).st_size == 0:
    #
    #     else:
    #         with open("{0}eeg_non_ictal_train2.csv".format('/data1/lee/file_csv/'), 'w', newline="") as csvfile:
    #             fieldnames = ['img', 'label']
    #             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #             writer.writeheader()
    #             normal_list = glob.glob("/data1/lee/chb_mit_20s/non_ictal/train/*.npy")
    #             for n in normal_list:
    #                 writer.writerow({'img': n, 'label': 0})


    # mRMR = mRMR_feature(10)
    # class_data, class_id, target, _, redundancy_score = mRMR.data_load()
    #
    # similarity_matrix, similarity_score = mRMR.similarity_matrix(class_data, 0.1)
    # np.save('/home/compu/LJC/data/voting/features/similarity_matrix.npy', similarity_matrix)
    # np.save('/home/compu/LJC/data/voting/features/similarity_score.npy', similarity_score)

    # break_true = 0
    #
    # # cluster_list = [10, 30, 50, 70, 90, 100, 200, 300, 400, 500]
    # cluster_list = [10]
    #
    # for k in cluster_list:
    #     mRMR = mRMR_feature(k)
    #     class_data, class_id, target, similarity_score, redundancy_score = mRMR.data_load()
    #
    #     data_length = len(class_data)
    #     redundancy = np.zeros((data_length, 1))
    #     choice_redundancy_data, choice_feature_data, choice_id = np.zeros((4 * k, 1)), np.zeros(
    #         (4 * k, 1024)), np.zeros((4 * k, 1))
    #     remaining_redundancy_data, remaining_feature_data, remaining_id = np.zeros((data_length - (4 * k), 1)), np.zeros(
    #         (data_length - (4 * k), 1024)), np.zeros((data_length - (4 * k), 1))
    #
    #     data_section = [0, 773, 773 + 1866, 773 + 1866 + 2997, 773 + 1866 + 2997 + 1391]
    #     remaining_data_section = [0, 773 - k, 773 + 1866 - (k*2), 773 + 1866 + 2997 - (k*3), 773 + 1866 + 2997 + 1391 - (k*4)]
    #     # data_section = [0, 773, 773 + 1035, 773 + 1035 + 1903, 773 + 1035 + 1903 + 1163]
    #     # remaining_data_section = [0, 773 - k, 773 + 1035 - (k*2), 773 + 1035 + 1903 - (k*3), 773 + 1035 + 1903 + 1163 - (k*4)]
    #     choice_index = np.zeros((4, k)).astype(int)
    #
    #     # 각 class마다 redundancy score가 가장 작은 것을 선정(이 때 redundancy score: 전체 sample에 대한 정보)
    #     for i in range(4):
    #         redundancy = redundancy_score[data_section[i]:data_section[i+1]]
    #         class_feature_data = class_data[data_section[i]:data_section[i+1]]
    #         redundancy_id = class_id[data_section[i]:data_section[i+1]]
    #
    #         choice_index[i] = np.argsort(redundancy, 0)[:k].reshape(-1)
    #         # choice_index[i] = np.argsort(redundancy, 0)[len(redundancy) - k:len(redundancy)].reshape(-1)
    #
    #         choice_redundancy_data[k*i:k*(i+1)] = redundancy[choice_index[i]]
    #         choice_feature_data[k*i:k*(i+1)] = class_feature_data[choice_index[i]]
    #         choice_id[k*i:k*(i+1)] = redundancy_id[choice_index[i]]
    #
    #         remaining_redundancy_data[remaining_data_section[i]:remaining_data_section[i+1]] = \
    #             np.delete(redundancy, (choice_index)[i]).reshape(-1, 1)
    #         remaining_feature_data[remaining_data_section[i]:remaining_data_section[i + 1]] = \
    #             np.delete(class_feature_data, (choice_index[i]), axis=0)
    #         remaining_id[remaining_data_section[i]:remaining_data_section[i+1]] = \
    #             np.delete(redundancy_id, (choice_index[i])).reshape(-1, 1)
    #
    #     # 선정된 sample 끼리의 redundancy score 계산
    #     local_sample_redundancy_score, between_score, within_score = mRMR.lda_redundancy(similarity_score, choice_index,
    #                                                                                      k)
    #     # local_sample_redundancy_score = mRMR.new_similarity(similarity_score, choice_index, k)
    #
    #     # # 선정된 sample에 대해 다시 sort
    #     # for ii in range(4):
    #     #     order = np.argsort(local_sample_redundancy_score[ii*k : (ii+1)*k], 0)
    #     #     local_sample_redundancy_score[ii * k: (ii + 1) * k] = local_sample_redundancy_score[ii * k: (ii + 1) * k][order].reshape(-1, 1)
    #     #     choice_feature_data[ii * k: (ii + 1) * k] = choice_feature_data[ii * k: (ii + 1) * k][order].squeeze()
    #     #     choice_id[ii * k: (ii + 1) * k] = choice_id[ii * k: (ii + 1) * k][order].reshape(-1, 1)
    #
    #     relevance_score = 0
    #     for ii in range(1024):
    #         relevance_score += mRMR.mutual_information_classif(choice_feature_data[:, ii], target)
    #     relevance_score = relevance_score / 1024
    #
    #     # total_score = relevance_score - (np.sum(choice_redundancy_data)/(4*k))
    #     total_score = relevance_score - (np.sum(local_sample_redundancy_score)/(4*k))
    #
    #     ###################################################### Update ######################################################
    #     global best_mRMR_score, best_id, select_index
    #     best_mRMR_score = total_score
    #     best_id = 0
    #     select_index = choice_index
    #     remaining_index = np.zeros((data_length - (4 * k)))
    #
    #     np.argsort(local_sample_redundancy_score[0: k], 0)
    #
    #     # index_cnt, best_redundancy = 0, local_sample_redundancy_score[9]
    #     index_cnt, best_redundancy = 0, local_sample_redundancy_score[np.argsort(local_sample_redundancy_score[0: k], 0)[-1]]
    #
    #     for iii in range(data_length):
    #         if (iii != choice_index).all():
    #             remaining_index[index_cnt] = iii
    #             index_cnt += 1
    #
    #     # remaining_data_section: 각 class마다 4 * k개를 제외한 배열 길이
    #     # target: shape: (4, k) 각 class 정보 [1, ..., 2, ..., 3, ..., 4, ...]
    #     # total_score: 선정된 4 * k개의 R - S & local_sample_redundancy_score: 선정된 4 * k개의 redundancy score
    #     # choice_feature_data: 선정된 4 * k개의 feature vector
    #     # choice_id: 선정된 4 * K개의 image id & select_index: 선정된 4 * K개의 index 정보
    #     # remaining_index: 남아 있는 index 정보 & similarity_score: similarity_matrix
    #     # best_redundancy: 가장 좋은 sample을 찾기 위한 redundancy 정보(sample이 바뀔 때마다 특정 조건마다 바뀜)
    #     # break_true: while문 제거 조건 & best_mRMR_score: 가장 좋은 mRMR score 저장
    #     # best_id: image id
    #
    #     remaining_data_section = [[[312], [327], [186], [592]], [[2541], [2395], [2554], [2474]], [[3082], [3694], [4544]], [[6497], [6541], [6477]]]
    #     choice_samlpe_id, break_true = mRMR.Updata(remaining_data_section, remaining_data_section,
    #                     target, total_score, k, local_sample_redundancy_score,  # choice_redundancy_data
    #                     choice_feature_data, choice_id, select_index, remaining_index, similarity_score,
    #                     best_redundancy, remaining_redundancy_data, remaining_feature_data, remaining_id,
    #                     break_true, best_mRMR_score, best_id)
    #
    #
    #
    #     np.save('/home/compu/LJC/data/voting/0.9_feature/lda_choice_id_{0}.npy'.format(k), choice_samlpe_id)
    #     break_true = 0
    #     print('{0}'.format(k))














