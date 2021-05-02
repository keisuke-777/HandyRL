# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import os
import sys
import yaml


import torch
import random
from handyrl.agent import Agent
from handyrl.model import ModelWrapper
from handyrl.envs.ci_geister import test_cigeister, Environment
from game import State, random_action
import numpy as np

# gameのstate→handyRLのobs(完全情報)
def convert_state_to_obs(state):
    obs = {"scalar": None, "board": None}
    blue_c_cnt = red_c_cnt = blue_o_cnt = red_o_cnt = 0
    # boardの作成
    # BRbrのnp配列をそれぞれ作成して、それらを足し合わせてなんとかする

    # 自分の駒
    blue_c = np.zeros((6, 6), dtype=np.float32)
    red_c = np.zeros((6, 6), dtype=np.float32)
    for index, piece_color in enumerate(state.pieces):
        # blue
        if piece_color == 1:
            blue_c[index // 6][index % 6] = 1
            blue_c_cnt += 1
        # red
        if piece_color == 2:
            red_c[index // 6][index % 6] = 1
            red_c_cnt += 1
    # 時計回りに90度回転
    blue_c = np.rot90(blue_c, k=3)
    red_c = np.rot90(red_c, k=3)

    # 敵の駒
    blue_o = np.zeros((6, 6), dtype=np.float32)
    red_o = np.zeros((6, 6), dtype=np.float32)
    for index, piece_color in enumerate(state.enemy_pieces):
        if piece_color == 1:
            blue_o[index // 6][index % 6] = 1
            blue_o_cnt += 1
        if piece_color == 2:
            red_o[index // 6][index % 6] = 1
            red_o_cnt += 1
    # 反時計回りに90度回転
    blue_o = np.rot90(blue_o, k=1)
    red_o = np.rot90(red_o, k=1)

    # 駒の情報を統合し、obsで使われていた形式に変換
    board = np.stack(
        [
            np.ones((6, 6), dtype=np.float32),
            blue_c + red_c,
            blue_o + red_o,
            blue_c,
            red_c,
            blue_o,
            red_o,
        ]
    ).astype(np.float32)

    obs["board"] = board

    # scalarの作成
    scalar = np.zeros(18, dtype=np.float32)

    # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    # 先手かどうかの部分どうすれば良いのかわからんので後程修正(先手か後手かでミスると見たことない盤面が出ちゃうかも？？)
    scalar[0] = state.depth % 2  # 先手かどうか
    scalar[1] = 1

    # 駒の数をobsで用いられていた形式に変換
    scalar[6 - blue_c_cnt] = 1
    scalar[10 - red_c_cnt] = 1
    scalar[14 - blue_o_cnt] = 1
    scalar[18 - red_o_cnt] = 1

    obs["scalar"] = scalar
    return obs


# 内容的にはconvert_state_to_obsと同じだが、GuessEnemyPieceで使える形式にしている
def convert_iip_to_obs(ii_pieces_array):
    # ii_pieces_arrayの中身 -> [[[自青],[自赤]],[[敵青],[敵赤]]]
    obs = {"scalar": None, "board": None}
    blue_c_cnt = red_c_cnt = blue_o_cnt = red_o_cnt = 0

    # 自分の駒
    blue_c = np.zeros((6, 6), dtype=np.float32)
    red_c = np.zeros((6, 6), dtype=np.float32)
    for index, my_blue in enumerate(ii_pieces_array[0][0]):
        if my_blue == 1:
            blue_c[index // 6][index % 6] = 1
            blue_c_cnt += 1

    for index, my_red in enumerate(ii_pieces_array[0][1]):
        if my_red == 1:
            red_c[index // 6][index % 6] = 1
            red_c_cnt += 1

    # 時計回りに90度回転
    blue_c = np.rot90(blue_c, k=3)
    red_c = np.rot90(red_c, k=3)

    blue_o = np.zeros((6, 6), dtype=np.float32)
    red_o = np.zeros((6, 6), dtype=np.float32)
    for index, enemy_blue in enumerate(ii_pieces_array[1][0]):
        if enemy_blue == 1:
            blue_o[index // 6][index % 6] = 1
            blue_o_cnt += 1

    for index, enemy_red in enumerate(ii_pieces_array[1][1]):
        if enemy_red == 1:
            red_o[index // 6][index % 6] = 1
            red_o_cnt += 1

    # 時計回りに90度回転(こっちは敵の駒も時計回り)
    blue_o = np.rot90(blue_o, k=3)
    red_o = np.rot90(red_o, k=3)

    # 駒の情報を統合し、obsで使われていた形式に変換
    board = np.stack(
        [
            np.ones((6, 6), dtype=np.float32),
            blue_c + red_c,
            blue_o + red_o,
            blue_c,
            red_c,
            blue_o,
            red_o,
        ]
    ).astype(np.float32)

    obs["board"] = board

    # scalarの作成
    scalar = np.zeros(18, dtype=np.float32)

    # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    scalar[0] = 1  # 先手かどうか
    scalar[1] = 1

    # 駒の数をobsで用いられていた形式に変換
    scalar[6 - blue_c_cnt] = 1
    scalar[10 - red_c_cnt] = 1
    scalar[14 - blue_o_cnt] = 1
    scalar[18 - red_o_cnt] = 1

    obs["scalar"] = scalar
    return obs


# Competeのii_state→handyRLのobs(不完全情報)
def convert_state_to_obs(ii_state):
    obs = {"scalar": None, "board": None}
    blue_c_cnt = red_c_cnt = blue_o_cnt = red_o_cnt = 0
    # boardの作成
    # BRbrのnp配列をそれぞれ作成して、それらを足し合わせてなんとかする

    # 自分の駒
    blue_c = np.zeros((6, 6), dtype=np.float32)
    red_c = np.zeros((6, 6), dtype=np.float32)
    for index, piece_color in enumerate(state.pieces):
        # blue
        if piece_color == 1:
            blue_c[index // 6][index % 6] = 1
            blue_c_cnt += 1
        # red
        if piece_color == 2:
            red_c[index // 6][index % 6] = 1
            red_c_cnt += 1
    # 時計回りに90度回転
    blue_c = np.rot90(blue_c, k=3)
    red_c = np.rot90(red_c, k=3)

    # 敵の駒
    blue_o = np.zeros((6, 6), dtype=np.float32)
    red_o = np.zeros((6, 6), dtype=np.float32)
    for index, piece_color in enumerate(state.enemy_pieces):
        if piece_color == 1:
            blue_o[index // 6][index % 6] = 1
            blue_o_cnt += 1
        if piece_color == 2:
            red_o[index // 6][index % 6] = 1
            red_o_cnt += 1
    # 反時計回りに90度回転
    blue_o = np.rot90(blue_o, k=1)
    red_o = np.rot90(red_o, k=1)

    # 駒の情報を統合し、obsで使われていた形式に変換
    board = np.stack(
        [
            np.ones((6, 6), dtype=np.float32),
            blue_c + red_c,
            blue_o + red_o,
            blue_c,
            red_c,
            blue_o,
            red_o,
        ]
    ).astype(np.float32)

    obs["board"] = board

    # scalarの作成
    scalar = np.zeros(18, dtype=np.float32)

    # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    # 先手かどうかの部分どうすれば良いのかわからんので後程修正(先手か後手かでミスると見たことない盤面が出ちゃうかも？？)
    scalar[0] = state.depth % 2  # 先手かどうか
    scalar[1] = 1

    # 駒の数をobsで用いられていた形式に変換
    scalar[6 - blue_c_cnt] = 1
    scalar[10 - red_c_cnt] = 1
    scalar[14 - blue_o_cnt] = 1
    scalar[18 - red_o_cnt] = 1

    obs["scalar"] = scalar
    return obs


# enemy_ii_predictで用いられるii_pieces_arrayから方策のリストを生成
def convert_func_use_in_guess(path):
    env = Environment()
    env.reset()
    agent = make_agent(env, path)
    state = State()

    def convert_func_use_in_guess(ii_pieces_array, legal_actions):
        obs = convert_iip_to_obs(ii_pieces_array)
        ap_list = action_sort_obs_to_policy(agent, obs, legal_actions)
        policies = []
        for tup in ap_list:
            policies.append(tup[1])

        # 方策が負の値を取ると色々困るので補正
        min_p = min(policies)
        policies -= min_p
        policies /= sum(policies) if sum(policies) else 1
        return policies

    return convert_func_use_in_guess


# gameのactions→handyRLのactions
def convert_gameAction_to_handyAction(game_actions):
    handy_actions = []
    for game_act in game_actions:
        pos = game_act // 4
        direction = game_act % 4

        # 座標を時計回りに90度回転させる作業
        row = pos // 6
        line = pos % 6
        handy_pos = 5 - row + line * 6

        # 方向に関しては互換性がないので場当たり的にやるしかない
        # game→handyの方向変換(回転を考慮)
        handy_dir = 100
        if direction == 0:
            handy_dir = 1
        elif direction == 1:
            handy_dir = 0
        elif direction == 2:
            handy_dir = 2
        elif direction == 3:
            handy_dir = 3
        else:
            print("エラー：convert_gameAction_to_handyAction")

        # 行動番号をhandyRLの形に計算しなおす
        handy_act = handy_pos + 36 * handy_dir
        handy_actions.append(handy_act)
    return handy_actions


# handyRLのaction→gameのaction
def convert_handyAction_to_gameAction(handy_action):
    pos = handy_action % 36
    direction = handy_action // 36

    # 座標を反時計回りに90度回転させる作業
    row = pos // 6
    line = pos % 6
    game_pos = (5 - line) * 6 + row

    # 方向を変換
    game_dir = 100
    if direction == 0:
        game_dir = 1
    elif direction == 1:
        game_dir = 0
    elif direction == 2:
        game_dir = 2
    elif direction == 3:
        game_dir = 3
    else:
        print("エラー：convert_handyAction_to_gameAction")

    game_action = game_pos * 4 + game_dir
    return game_action


# modelを持っているagentと自作のobsからaction(game.pyに適応)とpolicyのセットを抽出
def obs_to_policy_to_use_game(agent, obs, state):
    # 方策を取得
    outputs = agent.plan(obs)

    game_actions = state.legal_actions()
    handy_actions = convert_gameAction_to_handyAction(game_actions)

    p = outputs["policy"]
    ap_list = sorted(
        # 行動番号をgameで使用しているものに直しつつ、必要なポリシーを抽出
        [(convert_handyAction_to_gameAction(a), p[a]) for a in handy_actions],
        key=lambda x: -x[1],
    )
    return ap_list


# action順になるようにobs_to_policy_to_use_gameを実行
def action_sort_obs_to_policy(agent, obs, legal_actions):
    # 方策を取得
    outputs = agent.plan(obs)

    game_actions = legal_actions
    handy_actions = convert_gameAction_to_handyAction(game_actions)

    p = outputs["policy"]
    ap_list = sorted(
        [(convert_handyAction_to_gameAction(a), p[a]) for a in handy_actions],
        key=lambda x: x[0],
    )
    return ap_list


# モデルファイルからモデルを入手
def get_model(env, model_path):
    model = env.net()()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return ModelWrapper(model)


def make_agent(env, path):
    model_path = path
    agent = Agent(get_model(env, model_path))
    return agent


def test_predict():
    e = Environment()
    e.reset()

    for _ in range(10):
        actions = e.legal_actions()
        e.play(random.choice(actions))

    print(e.observation(0))

    for _ in range(1):
        print(e.legal_actions())
        print(e)
        print("行動を入力")
        num = input()
        e.play(int(num))
        actions = e.legal_actions()
        e.play(random.choice(actions))

    agent = make_agent(e)
    agent.action(e, 1, True)
    outputs = predict_using_obs(e.observation(0), agent)
    # print(outputs)
    # policy = outputs["policy"]
    # print(policy)
    # print(len(policy))


# 最も利得の高い行動を選択
# "models/10000.pth"
def HandyAction(path):
    env = Environment()
    env.reset()
    agent = make_agent(env, path)

    def HandyAction(state):
        obs = convert_state_to_obs(state)
        ap_list = obs_to_policy_to_use_game(agent, obs, state)
        return ap_list[0][0]

    return HandyAction


# def EvalHandyRL(number_of_matches, path):
#     env = Environment()
#     env.reset()
#     agent = make_agent(env, path)
#     for _ in range(number_of_matches):
#         # 状態の生成
#         state = State()
#         while True:
#             # ゲーム終了
#             if state.is_done():
#                 break
#             if state.depth % 2 == 0:
#                 obs = convert_state_to_obs(state)
#             else:
#                 pass


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"

    with open("config.yaml") as f:
        args = yaml.safe_load(f)
    # print(args)

    # ここに実験用のコードを書く

    state = State()
    while True:
        print(state.legal_actions())
        state = state.next(random_action(state))

    # path = "models/10000.pth"
    # EvalHandyRL(100, path)
    # policies = obs_to_policy_to_use_game(agent, obs, state)

    # print(policies)

    # convert_state_to_obs(state)

    # test_predict()
    # test_cigeister()

    # 方策を持ってくる

    # 接続部分
    # 受け取った方策を既存のコードで使える形に変換
