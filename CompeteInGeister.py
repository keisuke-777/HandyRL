import random
import math

# ゲームの状態
class State:
    # 初期化
    def __init__(self, pieces=None, enemy_pieces=None, depth=0):

        self.is_goal = False

        # 駒の配置
        if pieces != None:
            self.pieces = pieces
        else:
            self.pieces = [0] * 36

        if enemy_pieces != None:
            self.enemy_pieces = enemy_pieces
        else:
            self.enemy_pieces = [0] * 36

        # ターンの深さ(ターン数)
        self.depth = depth

        # 駒の初期配置
        if pieces == None or enemy_pieces == None:
            piece_list = [1, 1, 1, 1, 2, 2, 2, 2]  # 青4赤4

            random.shuffle(piece_list)  # 配置をランダムに
            self.pieces[25] = piece_list[0]
            self.pieces[26] = piece_list[1]
            self.pieces[27] = piece_list[2]
            self.pieces[28] = piece_list[3]
            self.pieces[31] = piece_list[4]
            self.pieces[32] = piece_list[5]
            self.pieces[33] = piece_list[6]
            self.pieces[34] = piece_list[7]

            random.shuffle(piece_list)  # 配置をランダムに
            self.enemy_pieces[25] = piece_list[0]
            self.enemy_pieces[26] = piece_list[1]
            self.enemy_pieces[27] = piece_list[2]
            self.enemy_pieces[28] = piece_list[3]
            self.enemy_pieces[31] = piece_list[4]
            self.enemy_pieces[32] = piece_list[5]
            self.enemy_pieces[33] = piece_list[6]
            self.enemy_pieces[34] = piece_list[7]

            # 実験用(データとりではランダムにする)
            # self.pieces[25] = self.enemy_pieces[25] = 1
            # self.pieces[26] = self.enemy_pieces[26] = 2
            # self.pieces[27] = self.enemy_pieces[27] = 1
            # self.pieces[28] = self.enemy_pieces[28] = 2
            # self.pieces[31] = self.enemy_pieces[31] = 2
            # self.pieces[32] = self.enemy_pieces[32] = 1
            # self.pieces[33] = self.enemy_pieces[33] = 2
            # self.pieces[34] = self.enemy_pieces[34] = 1

    # 負けかどうか
    def is_lose(self):
        if not any(elem == 1 for elem in self.pieces):  # 自分の青駒が存在しないなら負け
            return True
        if not any(elem == 2 for elem in self.enemy_pieces):  # 敵の赤駒が存在しない(全部取っちゃった)なら負け
            return True
        # 前の手でゴールされてたらis_goalがTrueになってる(ような仕様にする)
        if self.is_goal:
            return True
        return False

    # プリントする用途のis_lose
    def print_is_lose(self):
        if not any(elem == 1 for elem in self.pieces):  # 自分の青駒が存在しないなら負け
            print("青喰い")
            return True
        if not any(elem == 2 for elem in self.enemy_pieces):  # 敵の赤駒が存在しない(全部取っちゃった)なら負け
            print("赤喰い")
            return True
        if self.is_goal:
            print("ゴール")
            return True
        return False

    # 引き分けかどうか
    def is_draw(self):
        return self.depth >= 150  # 300手

    # ゲーム終了かどうか
    def is_done(self):
        return self.is_lose() or self.is_draw()

    # デュアルネットワークの入力の2次元配列の取得
    def pieces_array(self):
        # プレイヤー毎のデュアルネットワークの入力の2次元配列の取得
        def pieces_array_of(pieces):
            table_list = []
            # 青駒(1)→赤駒(2)の順に取得
            for j in range(1, 3):
                table = [0] * 36
                table_list.append(table)
                # appendは参照渡しなのでtable書き換えればtable_listも書き換わる
                for i in range(36):
                    if pieces[i] == j:
                        table[i] = 1

            return table_list

        # デュアルネットワークの入力の2次元配列の取得(自分と敵両方)
        return [pieces_array_of(self.pieces), pieces_array_of(self.enemy_pieces)]

    # position->0~35
    # direction->下:0,左:1,上:2,右:3

    # 駒の移動元と移動方向を行動に変換
    def position_to_action(self, position, direction):
        return position * 4 + direction

    # 行動を駒の移動元と移動方向に変換
    def action_to_position(self, action):
        return (int(action / 4), action % 4)  # position,direction

    # 合法手のリストの取得
    def legal_actions(self):
        actions = []
        for p in range(36):
            # 駒の存在確認
            if self.pieces[p] != 0:
                # 存在するなら駒の位置を渡して、その駒の取れる行動をactionsに追加
                actions.extend(self.legal_actions_pos(p))
        # 青駒のゴール行動は例外的に合法手リストに追加
        if self.pieces[0] == 1:
            actions.extend([2])  # 0*4 + 2
        if self.pieces[5] == 1:
            actions.extend([22])  # 5*4 + 2
        return actions

    # 駒ごと(駒1つに着目した)の合法手のリストの取得
    def legal_actions_pos(self, position):
        actions = []
        x = position % 6
        y = int(position / 6)
        # 下左上右の順に行動できるか検証し、できるならactionに追加
        # ちなみにand演算子は左の値を評価して右の値を返すか決める(左の値がTrue系でなければ右の値は無視する)ので、はみ出し参照してIndexErrorにはならない(&だとなる)
        if y != 5 and self.pieces[position + 6] == 0:  # 下端でない and 下に自分の駒がいない
            actions.append(self.position_to_action(position, 0))
        if x != 0 and self.pieces[position - 1] == 0:  # 左端でない and 左に自分の駒がいない
            actions.append(self.position_to_action(position, 1))
        if y != 0 and self.pieces[position - 6] == 0:  # 上端でない and 上に自分の駒がいない
            actions.append(self.position_to_action(position, 2))
        if x != 5 and self.pieces[position + 1] == 0:  # 右端でない and 右に自分の駒がいない
            actions.append(self.position_to_action(position, 3))
        # 青駒のゴール行動の可否は1ターンに1度だけ判定すれば良いので、例外的にlegal_actionsで処理する(ここでは処理しない)
        return actions

    # 次の状態の取得
    def next(self, action):
        # 次の状態の作成
        state = State(self.pieces.copy(), self.enemy_pieces.copy(), self.depth + 1)

        # position_bef->移動前の駒の位置、position_aft->移動後の駒の位置
        # 行動を(移動元, 移動方向)に変換
        position_bef, direction = self.action_to_position(action)

        # 合法手がくると仮定
        # 駒の移動(後ろに動くことは少ないかな？ + if文そんなに踏ませたくないな と思ったので判定を左右下上の順番にしてるけど意味あるのかは不明)
        if direction == 0:  # 下
            position_aft = position_bef + 6
        elif direction == 1:  # 左
            position_aft = position_bef - 1
        elif direction == 2:  # 上
            if position_bef == 0 or position_bef == 5:  # 0と5の上行動はゴール処理なので先に弾く
                state.is_goal = True
                position_aft = position_bef  # position_befを入れて駒の場所を動かさない(勝敗は決しているので下手に動かさない方が良いと考えた)
            else:
                position_aft = position_bef - 6
        elif direction == 3:  # 右
            position_aft = position_bef + 1
        else:
            print("error関数名:next")

        # 実際に駒移動
        state.pieces[position_aft] = state.pieces[position_bef]
        state.pieces[position_bef] = 0

        # 移動先に敵駒が存在した場合は取る(比較と値入れどっちが早いかあとで調べて最適化したい)
        # piecesとenemy_piecesを対応させるには値をひっくり返す必要がある(要素のインデックスは0~35だから、 n->35-n でひっくり返せる)
        if state.enemy_pieces[35 - position_aft] != 0:
            state.enemy_pieces[35 - position_aft] = 0

        # 駒の交代(ターンプレイヤが切り替わるため)(pieces <-> enemy_pieces)
        tmp = state.pieces
        state.pieces = state.enemy_pieces
        state.enemy_pieces = tmp
        return state

    # 先手かどうか
    def is_first_player(self):
        return self.depth % 2 == 0

    # どちらのプレイヤーが勝利したか判定
    # 勝敗決定時にやるだけだからそんな高速化しないで大丈夫
    def winner_checker(self, win_player):
        if self.is_goal:
            win_player[self.goal_player] += 1
            return

        board = [0] * 36
        if self.depth % 2 == 0:
            my_p = self.pieces.copy()
            rev_ep = list(reversed(self.enemy_pieces))
            for i in range(36):
                board[i] = my_p[i] - rev_ep[i]
        else:
            my_p = list(reversed(self.pieces))
            rev_ep = self.enemy_pieces.copy()
            for i in range(36):
                board[i] = rev_ep[i] - my_p[i]

        if not any(elem == 1 for elem in board):  # 先手の青駒が存在しない
            win_player[1] += 1  # 後手の勝ち
        elif not any(elem == 2 for elem in board):  # 先手の赤駒が存在しない
            win_player[0] += 1  # 先手の勝ち
        elif not any(elem == -1 for elem in board):  # 後手の青駒が存在しない
            win_player[0] += 1  # 先手の勝ち
        elif not any(elem == -2 for elem in board):  # 後手の赤駒が存在しない
            win_player[1] += 1  # 後手の勝ち
        else:
            pass

    # 文字列表示
    def __str__(self):
        row = "|{}|{}|{}|{}|{}|{}|"
        hr = "\n-------------------------------\n"

        # 1つのボードに味方の駒と敵の駒を集める
        board = [0] * 36
        if self.depth % 2 == 0:
            my_p = self.pieces.copy()
            rev_ep = list(reversed(self.enemy_pieces))
            for i in range(36):
                board[i] = my_p[i] - rev_ep[i]
        else:
            my_p = list(reversed(self.pieces))
            rev_ep = self.enemy_pieces.copy()
            for i in range(36):
                board[i] = rev_ep[i] - my_p[i]

        board_essence = []
        for i in board:
            if i == 1:
                board_essence.append("自青")
            elif i == 2:
                board_essence.append("自赤")
            elif i == -1:
                board_essence.append("敵青")
            elif i == -2:
                board_essence.append("敵赤")
            else:
                board_essence.append("　　")

        str = (
            hr + row + hr + row + hr + row + hr + row + hr + row + hr + row + hr
        ).format(*board_essence)
        return str


# ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions) - 1)]


# 人間に行動を選択させる
def human_player_action(state):
    # 盤面を表示
    print(state)

    # 入力を待つ(受ける)
    before_move_place = int(input("Please enter to move piece (左上~右下にかけて0~35) : "))
    direction = int(input("direction (下0 左1 上2 右3) : "))
    move = state.position_to_action(before_move_place, direction)

    # 合法手か確認
    legal_actions = state.legal_actions()
    if any(elem == move for elem in legal_actions):
        return move

    # エラー処理(デバッグでしか使わんから適当)
    print("非合法手が選択された為、ランダムに行動しました")
    return legal_actions[random.randint(0, len(legal_actions) - 1)]


from GuessEnemyPiece import (
    II_State,
    create_ii_state_from_state,
    guess_enemy_piece_player,
)
import numpy as np
import itertools
import time
from pv_mcts import predict, pv_mcts_scores, pv_mcts_action
from pathlib import Path
from tensorflow.keras.models import load_model
from test import IIHandyAction


# モンテカルロ木探索の行動選択
def mcts_action(state):
    # モンテカルロ木探索のノード
    class node:
        # 初期化
        def __init__(self, state):
            self.state = state  # 状態
            self.w = 0  # 累計価値
            self.n = 0  # 試行回数
            self.child_nodes = None  # 子ノード群

        # 評価
        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                # 勝敗結果で価値を取得
                value = -1 if self.state.is_lose() else 0  # 負けは-1、引き分けは0

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            # 子ノードが存在しない時
            if not self.child_nodes:
                # プレイアウトで価値を取得
                value = playout(self.state)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                # 子ノードの展開
                if self.n == 10:
                    self.expand()
                return value

            # 子ノードが存在する時
            else:
                # UCB1が最大の子ノードの評価で価値を取得
                value = -self.next_child_node().evaluate()

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

        # 子ノードの展開
        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(node(self.state.next(action)))

        # UCB1が最大の子ノードを取得
        def next_child_node(self):
            # 試行回数nが0の子ノードを返す
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            # UCB1の計算
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(
                    -child_node.w / child_node.n
                    + 2 * (2 * math.log(t) / child_node.n) ** 0.5
                )

            # UCB1が最大の子ノードを返す
            return self.child_nodes[argmax(ucb1_values)]

    # ルートノードの生成
    root_node = node(state)
    root_node.expand()

    # ルートノードを評価 (rangeを変化させると評価回数を変化させられる)
    for _ in range(100):
        root_node.evaluate()

    # 試行回数の最大値を持つ行動を返す
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]


# ゲームの終端までシミュレート
def playout(state):
    if state.is_lose():
        return -1
    if state.is_draw():
        return 0
    return -playout(state.next(random_action(state)))


# 最大値のインデックスを返す
def argmax(v_list):
    return v_list.index(max(v_list))


# モンテカルロ木探索用：敵駒のあり得るランダムな状態を返す
def return_random_shuffle_state(state):
    shuffle_state = state

    # shuffle_stateの生存している敵駒をリストに格納
    piece_list = []
    for i in range(20):
        if shuffle_state.enemy_pieces[i] != 0:
            piece_list.append(shuffle_state.enemy_pieces[i])

    # リストをシャッフル
    random.shuffle(piece_list)

    # shuffle_stateの生存している敵駒に上書き
    pl_index = 0
    for i in range(20):
        if shuffle_state.enemy_pieces[i] != 0:
            shuffle_state.enemy_pieces[i] = piece_list[pl_index]
            pl_index += 1

    return shuffle_state


# 不完全情報ゲームにおいて、透視をせずに(相手の状況をランダムとして)モンテカルロ木探索の行動選択を行う
def no_cheat_mcts_action(state):
    return mcts_action(return_random_shuffle_state(state))


# モンテカルロ木探索の行動選択
def predict_mcts_action(state, policy_action):
    # モンテカルロ木探索のノード
    class node:
        # 初期化
        def __init__(self, state):
            self.state = state  # 状態
            self.w = 0  # 累計価値
            self.n = 0  # 試行回数
            self.child_nodes = None  # 子ノード群
            self.policy_action = policy_action

        # 評価
        def evaluate(self):
            if self.state.is_done():
                value = -1 if self.state.is_lose() else 0  # 負けは-1、引き分けは0
                self.w += value
                self.n += 1
                return value

            # 子ノードが存在しない時
            if not self.child_nodes:
                # プレイアウトで価値を取得
                value = policy_playout(self.state, self.policy_action)
                self.w += value
                self.n += 1
                # 子ノードがない場合は初回探索で木を展開
                self.expand()
                return value

            # 子ノードが存在する時
            else:
                # UCB1が最大の子ノードの評価で価値を取得
                value = -self.next_child_node().evaluate()

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

        # 子ノードの展開
        def expand(self):
            legal_actions = self.state.legal_actions()
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(node(self.state.next(action)))

        # UCB1が最大の子ノードを取得
        def next_child_node(self):
            # 試行回数nが0の子ノードを返す
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            # UCB1の計算
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(
                    -child_node.w / child_node.n
                    + 2 * (2 * math.log(t) / child_node.n) ** 0.5
                )

            # UCB1が最大の子ノードを返す
            return self.child_nodes[argmax(ucb1_values)]

    # ルートノードの生成
    root_node = node(state)
    root_node.expand()

    # ルートノードを評価 (rangeを変化させると評価回数を変化させられる)
    for _ in range(50):
        root_node.evaluate()

    # 試行回数の最大値を持つ行動を返す
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]


# 方策を使ってゲームの終端までシミュレート
def policy_playout(state, policy_action):
    if state.is_lose():
        return -1
    if state.is_draw():
        return 0
    return -policy_playout(state.next(policy_action(state)), policy_action)


# どれほど正確に推論できているかどうかを計測する
# 正常に動作しません
def measure_estimate_accuracy(ii_state, state, csvWriter=None):
    # if state.depth % 10 != 0:
    #     return
    estimate_value = ii_state.return_estimate_value()
    real_blue_piece = list(
        ii_state.real_enemy_piece_blue_set
    )  # 0~4の青駒のインデックスを格納 ex)(1, 2, 3, 4)

    # 死んでいる敵駒の数(種類が確定している敵駒の数)
    dead_enemy_piece_num = (
        8 - ii_state.living_piece_color[0] - ii_state.living_piece_color[1]
    )
    dead_enemy_blue_piece_num = 4 - ii_state.living_piece_color[0]

    # estimate_valueの上位二つのインデックスを取得 array([n])でとってくるのでケツに[0][0]つける
    top_four = [
        np.where(estimate_value == np.sort(estimate_value)[-1]),
        np.where(estimate_value == np.sort(estimate_value)[-2]),
        np.where(estimate_value == np.sort(estimate_value)[-3]),
        np.where(estimate_value == np.sort(estimate_value)[-4]),
    ]
    # real_blue_pieceといくつ一致しているかを確認(最大4)
    number_of_matches = 0
    if (
        real_blue_piece[0] == top_four[0][0][0]
        or real_blue_piece[0] == top_four[1][0][0]
        or real_blue_piece[0] == top_four[2][0][0]
        or real_blue_piece[0] == top_four[3][0][0]
    ):
        number_of_matches += 1
    if (
        real_blue_piece[1] == top_four[0][0][0]
        or real_blue_piece[1] == top_four[1][0][0]
        or real_blue_piece[1] == top_four[2][0][0]
        or real_blue_piece[1] == top_four[3][0][0]
    ):
        number_of_matches += 1
    if (
        real_blue_piece[2] == top_four[0][0][0]
        or real_blue_piece[2] == top_four[1][0][0]
        or real_blue_piece[2] == top_four[2][0][0]
        or real_blue_piece[2] == top_four[3][0][0]
    ):
        number_of_matches += 1
    if (
        real_blue_piece[3] == top_four[0][0][0]
        or real_blue_piece[3] == top_four[1][0][0]
        or real_blue_piece[3] == top_four[2][0][0]
        or real_blue_piece[3] == top_four[3][0][0]
    ):
        number_of_matches += 1

    # 一致度合いを計測
    degree_of_match = float(0)
    for index, est_val in enumerate(estimate_value):
        if (
            index == real_blue_piece[0]
            or index == real_blue_piece[1]
            or index == real_blue_piece[2]
            or index == real_blue_piece[3]
        ):
            # 実際の駒の色が青だった場合
            degree_of_match += est_val - 0.5
        else:
            # 実際の駒の色が赤だった場合
            degree_of_match += 0.5 - est_val
    # degree_of_match /= 2

    if True:
        print("敵の青駒のインデックス", real_blue_piece)
        print("ターン数", "上位4駒の一致数", "一致度", "敵の死駒数", "敵の青の死駒数", "推測値", sep=",")
        print(
            state.depth,
            number_of_matches,
            degree_of_match,
            dead_enemy_piece_num,
            dead_enemy_blue_piece_num,
            estimate_value,
            sep=",",
        )


# 推測+完全情報の方策を用いた行動決定
def ci_pridict_action(ii_state, just_before_action_num, model_path, gamma):
    just_before_enemy_action_num = just_before_action_num
    guess_player_action = guess_enemy_piece_player(
        model_path, ii_state, just_before_enemy_action_num, gamma
    )
    return guess_player_action


def evaluate_HandyGeister(path_list=["latest"], gamma=0.9):
    from test import HandyAction

    global drow_count
    for path in path_list:
        print("models:", path)
        # ii_model_path = "ii_models/" + path + ".pth"
        ci_model_path = "models/" + path + ".pth"

        drow_count = 0
        win_player = [0, 0]
        # ii_handy_action = IIHandyAction(ii_model_path)

        # print("start compete : (path) " + path)
        for _ in range(100):
            # 直前の行動を保管
            just_before_action_num = 123  # 30左で初期値に戻った設定(先手検証用)

            # 状態の生成
            state = State()
            ii_state = create_ii_state_from_state(state, True)

            handy_action = HandyAction(ci_model_path)  # 完全情報ガイスター

            # ゲーム終了までループ
            while True:
                # print(state)
                if state.is_done():
                    print("ゲーム終了:ターン数", state.depth)
                    if state.print_is_lose():
                        if state.depth % 2 == 0:
                            print("敗北")
                        else:
                            print("勝利or引き分け")
                    else:
                        if state.depth % 2 == 1:
                            print("勝利or引き分け")
                        else:
                            print("敗北")
                    break

                # 次の状態の取得(ここも可読性下げすぎなので修正すべき)
                if state.depth % 2 == 0:
                    # 不完全情報でそのまま学習したエージェント
                    # just_before_action_num = ii_handy_action(
                    #     state
                    # )

                    # just_before_action_num = random_action(state)  # ランダム
                    just_before_action_num = no_cheat_mcts_action(state)  # 透視なしのMCTS
                    # just_before_action_num = handy_action(state)

                    if just_before_action_num == 2 or just_before_action_num == 22:
                        print("先手ゴール")
                        state.is_goal = True
                        state.goal_player = 0
                        break
                    state = state.next(just_before_action_num)
                else:
                    # 推測+完全情報の方策を用いた行動決定
                    just_before_action_num = ci_pridict_action(
                        ii_state, just_before_action_num, ci_model_path, gamma
                    )
                    # just_before_action_num = random_action(state)  # ランダム
                    # just_before_action_num = mcts_action(state) #透視MCTS
                    # just_before_action_num = no_cheat_mcts_action(state)  # 透視なしのMCTS

                    if just_before_action_num == 2 or just_before_action_num == 22:
                        print("後手ゴール", just_before_action_num)
                        state.is_goal = True
                        state.goal_player = 1
                        break
                    # measure_estimate_accuracy(ii_state, state)
                    state = state.next(just_before_action_num)
            # [先手の勝利数(検証相手), 後手の勝利数(推測するエージェント)]
            state.winner_checker(win_player)
            print(win_player)
        print("結果:", win_player)


# 動作確認
if __name__ == "__main__":
    # evaluate_GeisterLog()
    # path_list = ["1", "3000", "5000", "10000"]
    path_list = ["40000"]

    # path_list = []
    # for num in range(1, 44):
    #     path_list.append(str(num * 4000))

    # print("0.1だぞおおおおおおおおおおおお")
    # evaluate_HandyGeister(path_list, 0.1)
    # print("0.2だぞおおおおおおおおおおおお")
    # evaluate_HandyGeister(path_list, 0.2)
    # print("0.3だぞおおおおおおおおおおおお")
    # evaluate_HandyGeister(path_list, 0.3)
    # print("0.4だぞおおおおおおおおおおおお")
    # evaluate_HandyGeister(path_list, 0.4)
    # print("0.5だぞおおおおおおおおおおおお")
    # evaluate_HandyGeister(path_list, 0.5)
    # print("0.6だぞおおおおおおおおおおおお")
    # evaluate_HandyGeister(path_list, 0.6)
    # print("0.7だぞおおおおおおおおおおおお")
    # evaluate_HandyGeister(path_list, 0.7)
    # print("0.8だぞおおおおおおおおおおおお")
    # evaluate_HandyGeister(path_list, 0.8)
    # print("0.9だぞおおおおおおおおおおおお")
    evaluate_HandyGeister(path_list, 0.9)
    # print("1.0だぞおおおおおおおおおおおお")
    # evaluate_HandyGeister(path_list, 1.0)

