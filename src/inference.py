import os
import argparse
from typing import List, Dict
from yacs.config import CfgNode
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from collections import OrderedDict

from utils import load_config
from data.unified_loader import unified_loader
from models.build_model import Build_Model
from metrics.build_metrics import Build_Metrics
from visualization.build_visualizer import Build_Visualizer

# track_id별 trajectory(궤적) prediction을 통한 density map을 그림으로 출력해서 내보내는 inference code


# argument : gpu, mode, visualize / default로 설정해둠
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pytorch training & testing code for task-agnostic time-series prediction")
    parser.add_argument("--config_file", type=str, default='config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth.yml',
                        metavar="FILE", help='path to config file')
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument(
        "--mode", type=str, default="test")
    parser.add_argument(
        "--visualize", action="store_true", help="flag for whether visualize the results in mode:test")

    return parser.parse_args()


def evaluate_model(cfg: CfgNode, model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, visualize=False):
    model.eval()
    metrics = Build_Metrics(cfg)
    visualizer = Build_Visualizer(cfg)
    # timesteps 1번째만 시각화시킴 (0을 기반으로 1 update 시키는 방식)
    update_timesteps = [1]

    # visualize 코드 -> method가 visualize를 진행한다음에, 해당 시각화된 경로를 통해서 거리 계산
    if visualize:
        with torch.no_grad():
            result_list = []
            print("timing the computation, evaluating probability map, and visualizing... ")
            data_loader_one_each = unified_loader(
                cfg, rand=False, split="test", batch_size=1)


            # for i, data_dict in enumerate(tqdm(data_loader_one_each, leave=False, total=10)):
            #     data_dict = {k: data_dict[k].cuda()
            #                  if isinstance(data_dict[k], torch.Tensor)
            #                  else data_dict[k]
            #                  for k in data_dict}
            #     dict_list = []

            # 모든 시점, 인스턴스에 대한 시각화를 진행하도록 하는 코드
            # 위의 주석처리한 코드는 total =10 을 선택하면, 10개만 시각화된 결과물을 출력할 수 있다.

            for i, data_dict in enumerate(tqdm(data_loader_one_each, leave=False)):
                data_dict = {k: data_dict[k].cuda()
                             if isinstance(data_dict[k], torch.Tensor)
                             else data_dict[k]
                             for k in data_dict}
                dict_list = []

                # 모델 예측 부분
                result_dict = model.predict(
                    deepcopy(data_dict), return_prob=True)  # warm-up
                result_dict = model.predict(
                    deepcopy(data_dict), return_prob=True)


                for t in update_timesteps:
                    result_dict = model.predict_from_new_obs(result_dict, t)
                dict_list.append(deepcopy(result_dict))

                dict_list = metrics.denormalize(
                    dict_list)  # denormalize the output
                if cfg.TEST.KDE:
                    dict_list = kde(dict_list)
                dict_list = visualizer.prob_to_grid(dict_list)
                result_list.append(metrics(deepcopy(dict_list)))
                if visualize:
                    # if dict_list[0]["index"][0][2] == "PEDESTRIAN/37":
                    visualizer(dict_list)
                if i == 10:
                    break



def test(cfg: CfgNode, visualize) -> None:
    # dataloader -> unified loader

    data_loader = unified_loader(cfg, rand=False, split="test")
    model = Build_Model(cfg)
    try:
        model.load()
    except FileNotFoundError:
        print("no model saved")
    result_info = evaluate_model(cfg, model, data_loader, visualize)

    import json
    with open(os.path.join(cfg.OUTPUT_DIR, "metrics.json"), "w") as fp:
        json.dump(result_info, fp)

# 밀도 기반 시각화를 위한, KDE(Kernel Density Estimation) 코드
def kde(dict_list: List):
    from src.utils import GaussianKDE
    for data_dict in dict_list:
        for k in list(data_dict.keys()):
            if k[0] == "prob":
                prob = data_dict[k]
                batch_size, _, timesteps, _ = prob.shape
                prob_, gt_traj_log_prob = [], []
                for b in range(batch_size):
                    prob__, gt_traj_prob__ = [], []
                    for i in range(timesteps):
                        kernel = GaussianKDE(prob[b, :, i, :-1])
                        # estimate the prob of predicted future positions for fair comparison of inference time
                        kernel(prob[b, :, i, :-1])
                        prob__.append(deepcopy(kernel))
                        gt_traj_prob__.append(
                            kernel(data_dict["gt"][b, None, i].float()))
                    prob_.append(deepcopy(prob__))
                    gt_traj_log_prob.append(
                        torch.cat(gt_traj_prob__, dim=-1).log())
                gt_traj_log_prob = torch.stack(gt_traj_log_prob, dim=0)
                gt_traj_log_prob = torch.nan_to_num(
                    gt_traj_log_prob, neginf=-10000)
                data_dict[k] = prob_
                data_dict[("gt_traj_log_prob", k[1])] = gt_traj_log_prob

    return dict_list


def main() -> None:
    args = parse_args()
    cfg = load_config(args)

    if args.mode == "test":
        test(cfg, args.visualize)



if __name__ == "__main__":
    main()