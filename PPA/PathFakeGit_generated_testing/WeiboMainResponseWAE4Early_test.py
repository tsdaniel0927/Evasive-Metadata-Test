# 相比MainPathVoting的不同之处：random使用全部路径文本；wae使用response path，即除去了源文本
# 二者使用不同的id编码，即每一条路径对应random path ids和wae response path ids
from models.PathBased import ResponseWAE, ResponseCatWAE, ResponseWAECat
from torch import optim
import datetime
from evaluate import *
from get_args import _args, print_args
from data_io import *
import sys
from torch.optim.lr_scheduler import StepLR


def main():
    print_args(_args)
    # 固定随机数种子
    setup_seed(_args.seed)
    print("Step1:processing data")

    x_train_random, y_train_random, x_test_random, y_test_random = \
        load_path_data_weibo(train_path, test_path, label_path,
                             path_random_id2paths_dict_path,
                             path_random_npz)

    x_train_response, y_train_response, x_test_response, y_test_response = \
        load_path_data_weibo(train_path, test_path, label_path,
                             response_id2paths_dict_path,
                             response_wae_npz)

    print('Step2:build model')
    model = ResponseWAE(_args.random_vocab_dim, _args.response_vocab_dim,
                        wae_best_encoder_path, _args.random_dim,
                        _args.vae_dim, _args.class_num)
    model.to(device)

    model.load_state_dict(torch.load("weibo.pt"))
    with torch.no_grad():
        prediction = []
        # 因为每棵树都不同，所以测试的训练和测试的batch都为1；后续有待改进
        for j in range(len(y_test_random)):
            prediction.append(
                model.predict_up(torch.Tensor(x_test_random[j]).cuda(device).long(),
                                 torch.Tensor(x_test_response[j]).cuda(device).long())
                .cpu().data.numpy().tolist())
        res = evaluation_2class(prediction, y_test_random)
        print(res)


if __name__ == '__main__':
    main()
