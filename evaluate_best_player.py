from game import State, random_action, human_player_action, mcts_action
from pathlib import Path
import numpy as np
from AlphaBetaSearch import alpha_beta_action

# パラメータの準備
EP_GAME_COUNT = 100  # 1評価あたりのゲーム数

# 先手プレイヤーのポイント
def first_player_point(ended_state):
    # 1:先手勝利, 0:先手敗北, 0.5:引き分け
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    print("引き分け")
    return 0.5


# 1ゲームの実行
def play(next_actions):
    # 状態の生成
    state = State()

    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 行動の取得
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # 次の状態の取得
        state = state.next(action)

    # 先手プレイヤーのポイントを返す
    return first_player_point(state)


# 任意のアルゴリズムの評価
def evaluate_algorithm_of(label, next_actions):
    # 複数回の対戦を繰り返す
    total_point = 0
    for i in range(EP_GAME_COUNT):
        # 1ゲームの実行
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        # 出力
        print("\rEvaluate {}/{}".format(i + 1, EP_GAME_COUNT), end="")
    print("")

    print("total_point", total_point)
    # 平均ポイントの計算
    average_point = total_point / EP_GAME_COUNT
    print(label, average_point)


# ベストプレイヤーの評価
def evaluate_best_player():
    # HandyRLの方策を短絡的に選択するエージェント
    from test import HandyAction

    # handy_action = HandyAction("models/40000.pth")

    from test import IIHandyAction

    # handy_action = IIHandyAction("ii_models/10000.pth")

    # # VS過去の自分
    # handy_action_bf = HandyAction("models/20000.pth")
    # next_actions = (handy_action, handy_action_bf)
    # evaluate_algorithm_of("VS_過去の自分", next_actions)

    # # VSランダム
    # next_actions = (handy_action, random_action)
    # evaluate_algorithm_of("VS_Random", next_actions)

    # # VSモンテカルロ木探索
    # next_actions = (handy_action, mcts_action)
    # evaluate_algorithm_of("VS_MCTS", next_actions)

    # # VSモデルを変更して比較するHandyRL
    # handy_action_two = HandyAction("ii_models/20000.pth")
    # next_actions = (handy_action, handy_action_two)
    # evaluate_algorithm_of("VS_Handy", next_actions)

    # VS不完全情報で行動を選択するHandyRLエージェント
    # from test import IIHandyAction

    # ii_handy_action = IIHandyAction("ii_models/20000.pth")
    # next_actions = (handy_action, ii_handy_action)
    # evaluate_algorithm_of("VS_iiHandy", next_actions)

    # VS人間
    # next_actions = (handy_action, human_player_action)
    # evaluate_algorithm_of("VS_Human", next_actions)

    # ホゲホゲ
    # next_actions = (alpha_beta_action, mcts_action)
    # evaluate_algorithm_of("alphabeta_VS_MCTS", next_actions)

    # vsαβ
    handy_action = IIHandyAction("ii_models/40000.pth")
    # handy_action = HandyAction("models/108000.pth")
    next_actions = (alpha_beta_action, handy_action)
    evaluate_algorithm_of("alphabeta_VS_iiHandy", next_actions)

    # 自己対戦
    # next_actions = (next_pv_mcts_action, first_next_pv_mcts_action)
    # evaluate_algorithm_of("VS_過去の自分", next_actions)

    # for i in range(15):
    #     path = "models/" + str(i * 5000 + 5000) + ".pth"
    #     handy_action = HandyAction(path)
    #     print(path)
    #     next_actions = (handy_action, mcts_action)
    #     evaluate_algorithm_of("VS_MCTS", next_actions)


# 動作確認
if __name__ == "__main__":
    evaluate_best_player()
