import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import MultipleLocator


if __name__ == "__main__":

    dir_name = '../../results/'
    Jaccard_b_pred = []
    F1_b_pred = []
    Recall_b_pred = []
    Precision_b = []
    Acc_b_pred = []
    AP_pred = []
    RMSE_real = []


    sample_r = []
    txt_name = []

    names = ['RadioYnet', 'RadioCycle', 'RadioYnet_NRM']

    # txt_dir = './results/' + dir_name + '.txt'

    for name in names:
       txt_dir = os.path.join(dir_name, name, 'real_datasetstatistics.txt')
       with open(txt_dir, encoding='utf-8') as file:
          content = file.read()
          a = content.split('\n')
          Jaccard_b_pred_mean = list(map(float,((a[2].split('['))[1].split(']')[0]).split(',')))
          Jaccard_b_pred.append(Jaccard_b_pred_mean)

          F1_b_pred_mean = list(map(float,((a[4].split('['))[1].split(']')[0]).split(',')))
          F1_b_pred.append(F1_b_pred_mean)

          Recall_b_pred_mean = list(map(float,((a[6].split('['))[1].split(']')[0]).split(',')))
          Recall_b_pred.append(Recall_b_pred_mean)

          Precision_b_mean = list(map(float,((a[8].split('['))[1].split(']')[0]).split(',')))
          Precision_b.append(Precision_b_mean)

          Acc_b_pred_mean = list(map(float,((a[10].split('['))[1].split(']')[0]).split(',')))
          Acc_b_pred.append(Acc_b_pred_mean)

          AP_pred_mean = list(map(float, ((a[12].split('['))[1].split(']')[0]).split(',')))
          AP_pred.append(AP_pred_mean)

          RMSE_real_mean = list(map(float,((a[14].split('['))[1].split(']')[0]).split(',')))
          RMSE_real.append(RMSE_real_mean)

          sample_ratio = list(map(float,((a[16].split('['))[1].split(']')[0]).split(',')))
          sample_r.append(sample_ratio)


    plt.figure()
    plt.grid(True)
    # plt.title("F1", fontsize=10)
    ax = plt.gca()
    x_major_locator = MultipleLocator(5)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.05)
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    for i in range(len(names)):
        if names[i] == 'RadioYnet_NRM':
            names[i] = 'RadioYnet w/o RM'

        plt.plot(sample_r[i], AP_pred[i], label=names[i])
      # ax[0].plot(sample_ratio, F1_b_pred_mean, label='F1')
      # ax[0].plot(sample_ratio, Recall_b_pred_mean, label='Recall')
      # ax[0].plot(sample_ratio, Precision_b_mean, label='Precision')
      # ax[0].plot(sample_ratio, Acc_b_pred_mean, label='Accuracy')
    plt.legend()


    plt.figure()
    plt.grid(True)
    plt.title("RMSE", fontsize=10)
    ax = plt.gca()
    x_major_locator = MultipleLocator(5)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.02)
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    for i in range(len(names)):
      # plt.plot(sample_ratio, RMSE_real_mean, "o:", label='RMSE_real')
      # plt.plot(sample_ratio, RMSE_rec_mean, "o:", label='RMSE_rec')
      # plt.plot(sample_ratio, RMSE_rec_with_b_mean, "o:", label='RMSE_rec_with_b')
      # plt.plot(sample_ratio, unsample_r_rec_mean, "o:", label='unsample_r_rec')
      if names[i] != 'RadioYnet w/o RM':
        plt.plot(sample_r[i], RMSE_real[i], "o:", label=names[i], markersize=1)
      # ax[1].plot(sample_ratio, RMSE_real_mean, "o:", label='RMSE_real',markersize=1)
      # ax[1].plot(sample_ratio, RMSE_rec_mean, "o:", label='RMSE_rec',markersize=1)
      # ax[1].plot(sample_ratio, RMSE_rec_with_b_mean, "o:", label='RMSE_rec_with_b',markersize=1)
      # ax[1].plot(sample_ratio, unsample_r_rec_mean, "o:", label='unsample_r_rec',markersize=1)

    plt.legend()

    plt.show()

    # plt.figure()
    # plt.grid(True)
    # plt.title("NMSE", fontsize=10)
    # ax = plt.gca()
    # x_major_locator = MultipleLocator(5)
    # # 把x轴的刻度间隔设置为1，并存在变量里
    # # y_major_locator = MultipleLocator(0.02)
    # ax.xaxis.set_major_locator(x_major_locator)
    # # 把x轴的主刻度设置为1的倍数
    # # ax.yaxis.set_major_locator(y_major_locator)
    # for i in range(len(txt_name)):
    #     # plt.plot(sample_ratio, RMSE_real_mean, "o:", label='RMSE_real')
    #     # plt.plot(sample_ratio, RMSE_rec_mean, "o:", label='RMSE_rec')
    #     # plt.plot(sample_ratio, RMSE_rec_with_b_mean, "o:", label='RMSE_rec_with_b')
    #     # plt.plot(sample_ratio, unsample_r_rec_mean, "o:", label='unsample_r_rec')
    #     plt.plot(sample_r[i], NMSE_rec[i], "o:", label=name[txt_name[i][0:-4]], markersize=1)
    #     # ax[1].plot(sample_ratio, RMSE_real_mean, "o:", label='RMSE_real',markersize=1)
    #     # ax[1].plot(sample_ratio, RMSE_rec_mean, "o:", label='RMSE_rec',markersize=1)
    #     # ax[1].plot(sample_ratio, RMSE_rec_with_b_mean, "o:", label='RMSE_rec_with_b',markersize=1)
    #     # ax[1].plot(sample_ratio, unsample_r_rec_mean, "o:", label='unsample_r_rec',markersize=1)
    #
    # plt.legend()

    # plt.figure()
    # plt.grid(True)
    # # plt.title("cycleDWA_s20", fontsize=10)
    # ax = plt.gca()
    # x_major_locator = MultipleLocator(5)
    # # 把x轴的刻度间隔设置为1，并存在变量里
    # # y_major_locator = MultipleLocator(0.02)
    # ax.xaxis.set_major_locator(x_major_locator)
    # # 把x轴的主刻度设置为1的倍数
    # # ax.yaxis.set_major_locator(y_major_locator)
    # for i in range(len(txt_name)):
    #     if txt_name[i] == 'cycleDWA_s20.txt':
    #         plt.plot(sample_r[i], Jaccard_b_pred[i], label='Jaccard')
    #         plt.plot(sample_r[i], F1_b_pred[i], label='F1-score')
    #         plt.plot(sample_r[i], Recall_b_pred[i], label='Recall')
    #         plt.plot(sample_r[i], Precision_b[i], label='Precision')
    #         plt.plot(sample_r[i], Acc_b_pred[i], label='Accuracy')
    # plt.legend()
    # plt.show()




