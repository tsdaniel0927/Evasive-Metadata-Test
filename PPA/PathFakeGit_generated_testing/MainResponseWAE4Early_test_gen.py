# 相比MainPathVoting的不同之处：random使用全部路径文本；wae使用response path，即除去了源文本
# 二者使用不同的id编码，即每一条路径对应random path ids和wae response path ids
from models.PathBased import ResponseWAE
from evaluate import *
from get_args import _args, print_args
from data_io import *
from os.path import join
import os
import json


def main():
    labelid_2_name = {
        '0': 'false',
        '1': 'true',
        '2': 'unverified',
        '3': 'non-rumor'
    }

    print_args(_args)
    # 固定随机数种子
    setup_seed(_args.seed)
    print("Step1:processing data")

    test_path = join('220913__new_data', 'gen_test1516.json')
    label_path = join('220913__new_data', 'label1516.txt')

    if not os.path.exists(label_path):
        with open(test_path, 'r') as f:
            train_datas = [json.loads(line.strip()) for line in f.readlines()]
        with open(label_path, 'w') as f:
            for i in train_datas:
                f.write(f'{i["id_"]}\t{labelid_2_name[str(i["label"])]}\n')

    x_test_random, y_test_random = load_path_data_with_1516_test_with_gen(
        test_path, label_path, '220913__new_data/gen_test15_dict_random.json')

    x_test_response, y_test_response = load_path_data_with_1516_test_with_gen(
        test_path, label_path, '220913__new_data/gen_test15_dict_response.json')

    print('Step2:build model')
    model = ResponseWAE(_args.random_vocab_dim, _args.response_vocab_dim,
                        wae_best_encoder_path, _args.random_dim,
                        _args.vae_dim, _args.class_num)
    model.to(device)
    model.load_state_dict(torch.load("twitter1516.pt"))

    with torch.no_grad():
        prediction = []
        # 因为每棵树都不同，所以测试的训练和测试的batch都为1；后续有待改进
        for j in range(len(y_test_random)):
            prediction.append(
                model.predict_up(torch.Tensor(x_test_random[j]).cuda(device).long(),
                # model.predict_up(torch.Tensor(x_test_random[j])[:1,:].cuda(device).long(),
                                 torch.Tensor(x_test_response[j]).cuda(device).long())
                .cpu().data.numpy().tolist())
        res = evaluation_4class(prediction, y_test_random)
        print(res)


if __name__ == '__main__':
    main()
