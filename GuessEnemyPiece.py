# 相手の駒配置を予測
# これは不完全情報ゲームにおいて動作するようにする
# 正体が不明な相手の駒をとりあえず-1としておく

# board→14R24R34R44R15B25B35B45B41u31u21u11u40u30u20u10u
# move

import numpy as np
import itertools
import time
from game import State

# from pv_mcts import predict

from pathlib import Path
from tensorflow.keras.models import load_model

from test import convert_func_use_in_guess

# model_path = "models/10000.pth"
default_gamma = 0.9
DN_INPUT_SHAPE = (6, 6, 4)

# おそらく不完全情報ガイスター(のstateのみ？)を定義してそれを更新して管理した方がよさげ
# 不完全情報ガイスターの盤面情報及びそれらの推測値
class II_State:

    # クラス変数で駒順を定義
    piece_name = [
        "h",
        "g",
        "f",
        "e",
        "d",
        "c",
        "b",
        "a",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
    ]

    # 初期化
    def __init__(
        self,
        real_my_piece_blue_set,
        real_enemy_piece_blue_set=None,
        all_piece=None,
        enemy_estimated_num=None,
        my_estimated_num=None,
        enemy_piece_list=None,
        my_piece_list=None,
        living_piece_color=None,
    ):
        #  全ての駒(hgfedcbaABCDEFGHの順になっている)
        # 敵駒0~7,自駒8~15
        if all_piece == None:
            # numpyは基本的に型指定しない方が早い(指定すると裏で余計な処理するっぽい)
            self.all_piece = np.zeros(16, dtype=np.int16)
            # 初期配置を代入(各値は座標を示す)(脱出が88、死亡が99)
            # 0~7は敵駒, 8~15は自駒
            self.all_piece[0] = 1
            self.all_piece[1] = 2
            self.all_piece[2] = 3
            self.all_piece[3] = 4
            self.all_piece[4] = 7
            self.all_piece[5] = 8
            self.all_piece[6] = 9
            self.all_piece[7] = 10

            self.all_piece[8] = 25
            self.all_piece[9] = 26
            self.all_piece[10] = 27
            self.all_piece[11] = 28
            self.all_piece[12] = 31
            self.all_piece[13] = 32
            self.all_piece[14] = 33
            self.all_piece[15] = 34
        else:
            self.all_piece = all_piece

        if enemy_piece_list == None:
            self.enemy_piece_list = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            self.enemy_piece_list = enemy_piece_list

        if my_piece_list == None:
            self.my_piece_list = [8, 9, 10, 11, 12, 13, 14, 15]
        else:
            self.my_piece_list = my_piece_list

        # real_my_piece_blue_setは自分の青駒のIDのセット(引数必須)
        self.real_my_piece_blue_set = real_my_piece_blue_set
        self.real_my_piece_red_set = (
            set(self.my_piece_list) - self.real_my_piece_blue_set
        )

        # 敵の青駒のセット(デバッグ用)
        self.real_enemy_piece_blue_set = real_enemy_piece_blue_set

        # {敵青, 敵赤, 自青,　自赤}
        if living_piece_color == None:
            self.living_piece_color = [4, 4, 4, 4]
        else:
            self.living_piece_color = living_piece_color

        # [[推測値A,(パターンAの青駒のtuple表現)],[推測値B,(パターンBの青駒のtuple表現),...]
        if enemy_estimated_num == None:
            # 盤面の推測値を作成(大きい程青らしく、小さい程赤らしい)
            self.enemy_estimated_num = []
            for enemy_blue in itertools.combinations(
                set(self.enemy_piece_list), self.living_piece_color[0]
            ):
                self.enemy_estimated_num.append([0, enemy_blue])
        else:
            self.enemy_estimated_num = enemy_estimated_num

        if my_estimated_num == None:
            # 盤面の推測値を作成(大きい程青らしく、小さい程赤らしい)
            self.my_estimated_num = []
            for my_blue in itertools.combinations(
                set(self.my_piece_list), self.living_piece_color[0]
            ):
                self.my_estimated_num.append([0, my_blue])
        else:
            self.my_estimated_num = my_estimated_num

    #   ボードの初期配置はこんな感じ(小文字が敵の駒で大文字が自分の駒)
    #     0 1 2 3 4 5
    #   0   h g f e
    #   1   d c b a
    #   2
    #   3
    #   4   A B C D
    #   5   E F G H

    # 合法手のリストの取得
    # NNはactionを与えると事前に学習した方策を返す。
    # 赤のゴール(非合法なので知らない手)を与えると、そこを0にして返してくれるはず(エラーは吐かないはず？？？)
    def legal_actions(self):
        actions = []

        # リストに自分の駒を全て追加
        piece_coordinate_array = np.array([0] * 8)
        index = 0
        for i in range(8, 16):
            piece_coordinate_array[index] = self.all_piece[i]
            index += 1
        np.sort(piece_coordinate_array)

        # print("my:self.all_piece", piece_coordinate_array)

        for piece_coordinate in piece_coordinate_array:
            # 88以上は行動できないので省く(0~35)
            if piece_coordinate < 36:
                actions.extend(
                    self.piece_coordinate_to_actions(
                        piece_coordinate, piece_coordinate_array
                    )
                )
            # 0と5はゴールの選択肢を追加(赤駒でも問答無用)
            if piece_coordinate == 0:
                actions.extend([2])  # 0*4 + 2
            if piece_coordinate == 5:
                actions.extend([22])  # 5*4 + 2
        return actions

    # 相手目線の合法手のリストを返す
    def enemy_legal_actions(self):
        actions = []
        piece_coordinate_array = np.array([0] * 8)
        index = 0
        for i in range(0, 8):
            if self.all_piece[i] < 36:
                piece_coordinate_array[index] = 35 - self.all_piece[i]
            else:
                piece_coordinate_array[index] = 99
            index += 1
        np.sort(piece_coordinate_array)

        # print("enemy:self.all_piece", piece_coordinate_array)

        for piece_coordinate in piece_coordinate_array:
            # 88以上は行動できないので省く(0~35)
            if piece_coordinate < 36:
                actions.extend(
                    self.piece_coordinate_to_actions(
                        piece_coordinate, piece_coordinate_array
                    )
                )
            # 0と5はゴールの選択肢を追加(赤駒でも問答無用)
            if piece_coordinate == 0:
                actions.extend([2])  # 0*4 + 2
            if piece_coordinate == 5:
                actions.extend([22])  # 5*4 + 2
        return actions

    # 駒の移動元と移動方向を行動に変換
    def position_to_action(self, position, direction):
        return position * 4 + direction

    def piece_coordinate_to_actions(self, piece_coordinate, piece_coordinate_array):
        actions = []
        x = piece_coordinate % 6
        y = int(piece_coordinate / 6)

        if y != 5 and not np.any(piece_coordinate_array == (piece_coordinate + 6)):  # 下
            actions.append(self.position_to_action(piece_coordinate, 0))
        if x != 0 and not np.any(piece_coordinate_array == (piece_coordinate - 1)):  # 左
            actions.append(self.position_to_action(piece_coordinate, 1))
        if y != 0 and not np.any(piece_coordinate_array == (piece_coordinate - 6)):  # 上
            actions.append(self.position_to_action(piece_coordinate, 2))
        if x != 5 and not np.any(piece_coordinate_array == (piece_coordinate + 1)):  # 右
            actions.append(self.position_to_action(piece_coordinate, 3))

        return actions

    # 駒ごと(駒1つに着目した)の合法手のリストの取得
    def legal_actions_pos(self, position, piece_index_list):
        piece_list = []
        for piece_index in piece_index_list:
            piece_list.append(self.all_piece[piece_index])

        actions = []
        x = position % 6
        y = int(position / 6)
        # 下左上右の順に行動できるか検証し、できるならactionに追加
        # ちなみにand演算子は左の値を評価して右の値を返すか決める(左の値がTrue系でなければ右の値は無視する)ので、はみ出し参照してIndexErrorにはならない(&だとなる)
        if y != 5 and (position + 6) not in piece_list:  # 下端でない and 下に自分の駒がいない
            actions.append(self.position_to_action(position, 0))
        if x != 0 and (position - 1) not in piece_list:  # 左端でない and 左に自分の駒がいない
            actions.append(self.position_to_action(position, 1))
        if y != 0 and (position - 6) not in piece_list:  # 上端でない and 上に自分の駒がいない
            actions.append(self.position_to_action(position, 2))
        if x != 5 and (position + 1) not in piece_list:  # 右端でない and 右に自分の駒がいない
            actions.append(self.position_to_action(position, 3))
        # 青駒のゴール行動の可否は1ターンに1度だけ判定すれば良いので、例外的にlegal_actionsで処理する(ここでは処理しない)
        return actions

    # 行動を受けて、次の状態に遷移
    def next(self, action_num):
        coordinate_before, coordinate_after = action_to_coordinate(action_num)
        move_piece_index = np.where(self.all_piece == coordinate_before)[0][0]

        # 移動先に駒が存在する場合は殺す(味方の駒も殺してしまうが、そこは行動側で制御)
        if np.any(self.all_piece == coordinate_after):
            dead_piece_ID = np.where(self.all_piece == coordinate_after)[0][0]
            if dead_piece_ID < 8:  # 死んだのが敵駒
                # color_is_blue:死んだのが青駒かどうか
                color_is_blue = any(
                    i == dead_piece_ID for i in self.real_enemy_piece_blue_set
                )
                reduce_pattern(dead_piece_ID, color_is_blue, self)
            else:  # 死んだのが味方の駒
                color_is_blue = any(
                    i == dead_piece_ID for i in self.real_my_piece_blue_set
                )
                reduce_pattern(dead_piece_ID, color_is_blue, self)
        self.all_piece[move_piece_index] = coordinate_after  # 駒の移動

    # 推測値を返す(主にデバッグ用)
    def return_estimate_value(self):
        estimate_value = np.array([0] * 8, dtype="f4")
        for elem in self.enemy_estimated_num:
            id_matrix = [0] * 8
            # 青駒IDのとこだけ1にする
            for blue_id in elem[1]:
                id_matrix[blue_id] = 1
            estimate_value = estimate_value + (
                np.array(id_matrix, dtype="f4") * elem[0]
            )
        if False:
            print(self.enemy_estimated_num)
            print(
                "敵駒の住所",
                self.all_piece[0],
                self.all_piece[1],
                self.all_piece[2],
                self.all_piece[3],
            )
            print(
                "味方駒の住所",
                self.all_piece[4],
                self.all_piece[5],
                self.all_piece[6],
                self.all_piece[7],
            )
        return estimate_value

    # ボードの文字列表示
    def __str__(self):
        row = "|{}|{}|{}|{}|{}|{}|"
        hr = "\n-------------------------------\n"

        # 1つのボードに味方の駒と敵の駒を集める
        board = [0] * 36
        # 0~7が敵、8~15が自分

        # 敵の駒
        for enemy_piece_coo in self.all_piece[0:8]:
            if enemy_piece_coo < 36 and enemy_piece_coo >= 0:
                board[enemy_piece_coo] = -1

        # 自分の駒
        for blue_index in self.real_my_piece_blue_set:
            if self.all_piece[blue_index] < 36 and self.all_piece[blue_index] >= 0:
                board[self.all_piece[blue_index]] = 1
        for red_index in self.real_my_piece_red_set:
            if self.all_piece[red_index] < 36 and self.all_piece[red_index] >= 0:
                board[self.all_piece[red_index]] = 2

        board_essence = []
        for i in board:
            if i == 1:
                board_essence.append("自青")
            elif i == 2:
                board_essence.append("自赤")
            elif i == -1:
                board_essence.append("敵駒")
            else:
                board_essence.append("　　")

        ii_str = (
            hr + row + hr + row + hr + row + hr + row + hr + row + hr + row + hr
        ).format(*board_essence)
        ii_str += "\n" + str(self.living_piece_color)
        return ii_str


# stateの駒の色に応じたii_stateを作成する(初期のstateのみ使用可能)
def create_ii_state_from_state(state, enemy_view=False):
    if enemy_view:
        # 敵視点でii_stateを作成
        pieces = state.enemy_pieces
        enemy_pieces = state.pieces
    else:
        pieces = state.pieces
        enemy_pieces = state.enemy_pieces

    # 駒のIDと座標が紐づいたリストを手動作成(初期配置では座標番号25~28と31~34に駒が存在)
    piece_id_list = [0] * 36
    for i in range(4):
        piece_id_list[25 + i] = 8 + i
    for i in range(4):
        piece_id_list[31 + i] = 12 + i

    blue_piece_set = set({})
    for index, piece_color in enumerate(pieces):
        if piece_color == 1:
            blue_piece_set.add(piece_id_list[index])

    # 敵駒の処理も同様にする
    enemy_piece_id_list = [0] * 36
    for i in range(4):
        enemy_piece_id_list[25 + i] = 8 + i
    for i in range(4):
        enemy_piece_id_list[31 + i] = 12 + i

    enemy_blue_piece_set = set({})
    for index, piece_color in enumerate(enemy_pieces):
        if piece_color == 1:
            enemy_blue_piece_set.add(enemy_piece_id_list[index])

    # enemy_blue_piece_setの値を反転させ、推測の際に扱いやすいように変換する
    # (このままでは8~15の値をとるが、0~7の値に修正し扱う必要がある)
    rev_enemy_blue_piece_set = set({})
    for piece_coo in enemy_blue_piece_set:
        rev_enemy_blue_piece_set.add(15 - piece_coo)

    ii_state = II_State(blue_piece_set, rev_enemy_blue_piece_set)
    return ii_state


### ガイスターAI大会のプロトコル周り

# プロトコルから相手の行動は送られず、更新されたボードが送られてくるそうなので、行動した駒の座標を求める
# これは相手の行動のみ検知可能
def enemy_coordinate_checker(before_board, now_board):
    for i in range(len(before_board) // 2, len(before_board)):
        if before_board[i] != now_board[i]:
            break
    # iではなく(i//3)*3とすることで、座標と駒色(例:14R)の先頭インデックスが取れる(これしないと2文字目からとってくる恐れがある)
    beginningOfTheChanged = (i // 3) * 3

    # 列番号+行番号*6でgame.pyで使ってる表現に直せる
    before_coordinate = (
        int(before_board[beginningOfTheChanged])
        + int(before_board[beginningOfTheChanged + 1]) * 6
    )
    now_coordinate = (
        int(now_board[beginningOfTheChanged])
        + int(now_board[beginningOfTheChanged + 1]) * 6
    )

    # 行動前と行動後の座標を返す
    return before_coordinate, now_coordinate


# 行動番号を駒の移動元と移動方向に変換
def action_to_position(action_num):
    return (int(action_num / 4), action_num % 4)  # position,direction


# 行動番号を移動前の座標と移動後の座標に変換
def action_to_coordinate(action_num):
    coordinate_before, direction = action_to_position(action_num)
    if direction == 0:  # 下
        coordinate_after = coordinate_before + 6
    elif direction == 1:  # 左
        coordinate_after = coordinate_before - 1
    elif direction == 3:  # 右
        coordinate_after = coordinate_before + 1
    elif direction == 2:  # 上
        if coordinate_before == 0 or coordinate_before == 5:  # 0と5の上行動はゴール処理なので弾く
            coordinate_after = coordinate_before  # coordinate_beforeを入れて駒の場所を動かさない(勝敗は決しているので下手に動かさない方が良い(多分))
        else:
            coordinate_after = coordinate_before - 6
    else:
        print("ERROR:action_to_coordinate(illegal action_num)")
    return coordinate_before, coordinate_after


# 移動前の座標と方向番号から行動番号を算出
def position_to_action(position, direction):
    return position * 4 + direction


# 移動前と移動後の座標から相手の行動番号を算出
def calculate_enemy_action_number_from_coordinate(before_coordinate, now_coordinate):
    enemy_looking_now_coordinate = 35 - now_coordinate
    enemy_looking_before_coordinate = 35 - before_coordinate
    difference = enemy_looking_now_coordinate - enemy_looking_before_coordinate
    if difference == 6:  # 下
        return position_to_action(enemy_looking_before_coordinate, 0)
    elif difference == 1:  # 左
        return position_to_action(enemy_looking_before_coordinate, 1)
    elif difference == -6:  # 上
        return position_to_action(enemy_looking_before_coordinate, 2)
    elif difference == -1:  # 右
        return position_to_action(enemy_looking_before_coordinate, 3)
    else:
        print("ERROR:find_enemy_action_number_from_coordinate(illegal move)")
        return -1


###

# 相手の行動を受けて、ガイスターの盤面を更新(駒が死んだ場合の処理もここで行う)
def update_II_state(ii_state, before_coordinate, now_coordinate):
    kill = np.any(ii_state.all_piece == now_coordinate)

    # 敵駒がkillしていたら死んだ駒の処理を行う(99は死んだ駒)
    if kill:
        dead_piece_ID = np.where(ii_state.all_piece == now_coordinate)[0][0]
        color_is_blue = np.any(ii_state.real_my_piece_blue_set == dead_piece_ID)
        # print(dead_piece_ID, color_is_blue)
        reduce_pattern(dead_piece_ID, color_is_blue, ii_state)

    # 行動前の座標を行動後の座標に変更する
    ii_state.all_piece[
        np.where(ii_state.all_piece == before_coordinate)[0][0]
    ] = now_coordinate


# myの視点で状態を作成
def my_looking_create_state(ii_state, my_blue, my_red, enemy_blue, enemy_red):
    # プレイヤー毎のデュアルネットワークの入力の2次元配列の取得
    def pieces_array_of(blue_piece_list, red_piece_list):
        table_list = []
        blue_table = [0] * 36
        table_list.append(blue_table)  # ちなみにappendは参照渡し
        # blue_piece_listは駒のIDの値なので、ii_state.all_pieceでそのIDを参照してあげると座標が取れる
        for blue_piece in blue_piece_list:
            if ii_state.all_piece[blue_piece] < 36:  # 死駒を除外
                blue_table[ii_state.all_piece[blue_piece]] = 1

        red_table = [0] * 36
        table_list.append(red_table)
        for red_piece in red_piece_list:
            if ii_state.all_piece[red_piece] < 36:
                red_table[ii_state.all_piece[red_piece]] = 1

        return table_list

    # デュアルネットワークの入力の2次元配列の取得(自分と敵両方)
    return [pieces_array_of(my_blue, my_red), pieces_array_of(enemy_blue, enemy_red)]


# # 入力の順序はcreate
# # enemyの視点から状態を作成
# def enemy_looking_create_state(ii_state, my_blue, my_red, enemy_blue, enemy_red):
#     # プレイヤー毎のデュアルネットワークの入力の2次元配列の取得
#     def pieces_array_of(blue_piece_list, red_piece_list):
#         table_list = []
#         blue_table = [0] * 36
#         # blue_piece_listは駒のIDの値なので、ii_state.all_pieceでそのIDを参照してあげると座標が取れる
#         for blue_piece in blue_piece_list:
#             if ii_state.all_piece[blue_piece] < 36:  # 死駒を除外
#                 blue_table[ii_state.all_piece[blue_piece]] = 1
#         blue_table.reverse()  # 逆視点にするために要素を反転
#         table_list.append(blue_table)

#         red_table = [0] * 36
#         for red_piece in red_piece_list:
#             if ii_state.all_piece[red_piece] < 36:
#                 red_table[ii_state.all_piece[red_piece]] = 1
#         red_table.reverse()  # 逆視点にするために要素を反転
#         table_list.append(red_table)

#         return table_list

#     # デュアルネットワークの入力の2次元配列の取得(自分と敵両方)
#     return [pieces_array_of(enemy_blue, enemy_red), pieces_array_of(my_blue, my_red)]


# 諸々の情報からstateを作る
def create_state_from_enemy_looking(ii_state, my_blue, my_red, enemy_blue, enemy_red):
    # 自分の駒を格納
    my_table = [0] * 36
    for my_b in my_blue:
        if ii_state.all_piece[my_b] < 36:
            my_table[ii_state.all_piece[my_b]] = 1
    for my_r in my_red:
        if ii_state.all_piece[my_r] < 36:
            my_table[ii_state.all_piece[my_r]] = 2

    # 敵の駒を格納
    enemy_table = [0] * 36
    for en_b in enemy_blue:
        if ii_state.all_piece[en_b] < 36:
            enemy_table[ii_state.all_piece[en_b]] = 1
    for en_r in enemy_red:
        if ii_state.all_piece[en_r] < 36:
            enemy_table[ii_state.all_piece[en_r]] = 2
    enemy_table.reverse()  # このままでは敵の駒の座標が逆なので反転させて戻す

    # 敵視点でのstateを生成
    state = State(enemy_table, my_table)

    return state


# enemy→各駒の推測値を保存。推測のために70パターン想定するが、足し合わせるだけ(各盤面について保存はしない)
# my→推測したい駒配置。
# 行動と推測盤面に対応した行動価値のリストを返す
def my_ii_predict(model_path, ii_state):
    # 推論のための入力データのシェイプの変換
    a, b, c = DN_INPUT_SHAPE  # (6, 6, 4)

    # ii_stateから生きてる駒のリストを取得
    my_piece_set = set(ii_state.my_piece_list)
    enemy_piece_set = set(ii_state.enemy_piece_list)

    # policies_list[パターン(0~最大69)][行動(盤面依存)]
    policies_list = []
    legal_actions = list(ii_state.legal_actions())

    # HandyRLで学習させた方策を取れる関数を定義
    convert_func = convert_func_use_in_guess(model_path)

    for num_and_my_blue in ii_state.my_estimated_num:
        sum_np_policies = np.array([0] * len(legal_actions), dtype="f4")

        # 赤駒のインデックスをセット形式で獲得(青駒以外の駒は赤駒)
        my_red_set = my_piece_set - set(num_and_my_blue[1])

        for num_and_enemy_blue in ii_state.enemy_estimated_num:

            # 同様に赤駒のインデックスを獲得
            enemy_red_set = enemy_piece_set - set(num_and_enemy_blue[1])

            ii_pieces_array = my_looking_create_state(
                ii_state,
                num_and_my_blue[1],
                my_red_set,
                num_and_enemy_blue[1],
                enemy_red_set,
            )

            # HandyRLに適応
            policies = convert_func(ii_pieces_array, legal_actions)

            # 行列演算するためにndarrayに変換
            np_policies = np.array(policies, dtype="f4")

            # 自分のパターンは既存のpoliciesに足すだけ
            sum_np_policies = sum_np_policies + np_policies

            # value = y[1][0][0] # 価値の取得
        policies_list.extend([sum_np_policies])
    return policies_list


# # 相手の行動前に、相手の目線で各パターンにおける各行動の価値を算出
# def enemy_ii_predict(model_path, ii_state):
#     a, b, c = DN_INPUT_SHAPE  # (6, 6, 4)
#     my_piece_set = set(ii_state.my_piece_list)
#     enemy_piece_set = set(ii_state.enemy_piece_list)
#     policies_list = []
#     enemy_legal_actions = sorted(list(ii_state.enemy_legal_actions()), key=lambda x: x)
#     convert_func = convert_func_use_in_guess(model_path)
#     for num_and_enemy_blue in ii_state.enemy_estimated_num:  # enemyのパターンの確からしさを求めたい
#         # 赤駒のインデックスをセット形式で獲得(my_blueはタプル)
#         enemy_red_set = enemy_piece_set - set(num_and_enemy_blue[1])
#         sum_np_policies = np.array([0] * len(enemy_legal_actions), dtype="f4")
#         for num_and_my_blue in ii_state.my_estimated_num:
#             my_red_set = my_piece_set - set(num_and_my_blue[1])
#             # 要修正
#             ii_pieces_array = enemy_looking_create_state(
#                 ii_state,
#                 num_and_my_blue[1],
#                 my_red_set,
#                 num_and_enemy_blue[1],
#                 enemy_red_set,
#             )
#             # HandyRLに適応
#             policies = convert_func(ii_pieces_array, enemy_legal_actions)
#             # 行列演算するためにndarrayに変換
#             np_policies = np.array(policies, dtype="f4")
#             # myのパターンは既存のpoliciesに足すだけ
#             sum_np_policies = sum_np_policies + np_policies
#         policies_list.extend([sum_np_policies])
#     return policies_list


from test import get_policies

# 自分の駒配置を確定させて推測するパターン
# 相手の行動前に、相手の目線で各パターンにおける各行動の価値を算出
def enemy_ii_predict(model_path, ii_state):
    my_piece_set = set(ii_state.my_piece_list)
    enemy_piece_set = set(ii_state.enemy_piece_list)
    policies_list = []
    enemy_legal_actions = sorted(list(ii_state.enemy_legal_actions()), key=lambda x: x)

    # enemy_legal_actions = sorted(enemy_legal_actions, key=lambda x: x)
    # print("ii_legal", enemy_legal_actions)
    # convert_func = convert_func_use_in_guess(model_path)
    # print("盤面", ii_state)

    get_policies_func = get_policies(model_path)

    # enemyのパターンの確からしさ(蓋然性)を求める
    for num_and_enemy_blue in ii_state.enemy_estimated_num:
        # 赤駒のインデックスをセット形式で獲得(my_blueはタプル)
        enemy_red_set = enemy_piece_set - set(num_and_enemy_blue[1])

        # 自分の駒配置は見抜かれているものとして相手の行動の価値を求める
        my_blue_set = ii_state.real_my_piece_blue_set
        my_red_set = my_piece_set - my_blue_set

        # 相手視点のstateを作成した上で方策を獲得
        enemy_looking_state = create_state_from_enemy_looking(
            ii_state, my_blue_set, my_red_set, num_and_enemy_blue[1], enemy_red_set,
        )
        ap_list = sorted(get_policies_func(enemy_looking_state), key=lambda x: x[0])
        policies = []
        for tup in ap_list:
            policies.append(tup[1])

        # HandyRLに適応
        # ii_pieces_array = enemy_looking_create_state(
        #     ii_state, my_blue_set, my_red_set, num_and_enemy_blue[1], enemy_red_set,
        # )
        # policies = convert_func(ii_pieces_array, enemy_legal_actions)

        # ndarrayに変換(自分の駒配置が確定でなかった際に、行列演算するためにnpに変換していた名残)
        np_policies = np.array(policies, dtype="f4")
        policies_list.extend([np_policies])
    return policies_list


# 相手の行動から推測値を更新
# state, enemy_ii_predictで作成した推測値の行列, 敵の行動番号
def update_predict_num_all(
    ii_state, beforehand_estimated_num, enemy_action_num, gamma=default_gamma
):
    # print(enemy_action_num)
    enemy_legal_actions = sorted(list(ii_state.enemy_legal_actions()), key=lambda x: x)
    enemy_action_index = enemy_legal_actions.index(enemy_action_num)

    for index, enemy_estimated_num in enumerate(ii_state.enemy_estimated_num):
        # ii_state.enemy_estimated_num[index][0]
        enemy_estimated_num[0] = (
            enemy_estimated_num[0] * gamma
        ) + beforehand_estimated_num[index][enemy_action_index]


# 相手の行動から推測値を更新
# 相手の取りうる指し手の中で最も価値の高い行動であれば、その盤面である可能性が高いと考える
# 最も価値の高い行動であれば+1、そうでなければ+0
def update_predict_num_max_only(
    ii_state, beforehand_estimated_num, enemy_action_num, gamma=default_gamma
):
    enemy_legal_actions = sorted(list(ii_state.enemy_legal_actions()), key=lambda x: x)
    enemy_action_index = enemy_legal_actions.index(enemy_action_num)
    bf_estimated_num = beforehand_estimated_num

    # 未知の駒が0か5に存在している場合、未知の駒が赤駒である場合でも行動番号2や22を追加しないと配列への参照(beforehand_estimated_num[index][enemy_action_index])がズレる
    goal_actions = []
    if 2 in enemy_legal_actions:
        goal_actions.append(2)
    if 22 in enemy_legal_actions:
        goal_actions.append(22)
    # ゴール行動が含まれていた場合
    if goal_actions != []:
        legal_length = len(enemy_legal_actions)
        for goal in goal_actions:
            new_est_num = []
            for bf_index, bf_est_num in enumerate(bf_estimated_num):
                # このやり方だと0,5の両方に赤駒がいた場合に対処できないが、ごく稀にしか起こらないためその場合の推測は割り切る
                if legal_length > len(bf_est_num):
                    insert_index = enemy_legal_actions.index(goal)
                    new_est_num.append(
                        np.insert(bf_estimated_num[bf_index], insert_index, -999.9)
                    )
                else:
                    new_est_num.append(bf_estimated_num[bf_index])
            bf_estimated_num = new_est_num

    enemy_estimated_num = ii_state.enemy_estimated_num
    for index, en_est_num in enumerate(enemy_estimated_num):
        action_value = bf_estimated_num[index][enemy_action_index]

        # if np.argmax(bf_estimated_num[index]) == enemy_action_index:
        #     # 推測値を更新(1を足す)
        #     en_est_num[0] = (en_est_num[0] * gamma) + 1

        # 相手の選択した行動が、相手の取りうる行動の中で最高の評価であった場合
        if np.partition(bf_estimated_num[index].ravel(), -1)[-1] == action_value:
            en_est_num[0] = (en_est_num[0] * gamma) + 1

        # 相手の選択した行動が、相手の取りうる行動の中で2番目の評価であった場合
        # if (
        #     np.partition(bf_estimated_num[index].ravel(), -2)[-2]
        #     == action_value
        # ):
        #     en_est_num[0] = (en_est_num[0] * gamma) + 0.7
        # # 相手の選択した行動が、相手の取りうる行動の中で3番目の評価であった場合
        # if (
        #     np.partition(bf_estimated_num[index].ravel(), -3)[-3]
        #     == action_value
        # ):
        #     en_est_num[0] = (en_est_num[0] * gamma) + 0.5


# 駒の死亡処理
# 既存のパターンから推測値を抜き出して新しい推測値を作成
def reduce_pattern(dead_piece_ID, color_is_blue: bool, ii_state):
    # print(dead_piece_ID, color_is_blue, ii_state)
    if dead_piece_ID < 8 and color_is_blue:  # 敵駒 and 駒が青色
        # dead_piece_IDが含まれていないものを削除(dead_piece_IDが青色で確定するため、それが青色の駒のリストに含まれていないとおかしい)
        # リストをそのままfor内で削除するとインデックスがバグるのでコピーしたものを参照
        for enemy_estimated_num in ii_state.enemy_estimated_num[:]:
            if dead_piece_ID not in enemy_estimated_num[1]:
                ii_state.enemy_estimated_num.remove(enemy_estimated_num)
    elif dead_piece_ID < 8 and not color_is_blue:  # 敵駒 and 駒が赤色
        # dead_piece_IDが含まれているものを削除
        for enemy_estimated_num in ii_state.enemy_estimated_num[:]:
            if dead_piece_ID in enemy_estimated_num[1]:
                ii_state.enemy_estimated_num.remove(enemy_estimated_num)
    elif dead_piece_ID >= 8 and color_is_blue:  # 自駒 and 駒が青色
        for my_estimated_num in ii_state.my_estimated_num[:]:
            if dead_piece_ID not in my_estimated_num[1]:
                ii_state.my_estimated_num.remove(my_estimated_num)
    elif dead_piece_ID >= 8 and not color_is_blue:  # 自駒 and 駒が赤色
        for my_estimated_num in ii_state.my_estimated_num[:]:
            if dead_piece_ID in my_estimated_num[1]:
                ii_state.my_estimated_num.remove(my_estimated_num)

    # all_pieceから削除
    ii_state.all_piece[dead_piece_ID] = 99
    # **_piece_listから削除
    if dead_piece_ID < 8:
        ii_state.enemy_piece_list.remove(dead_piece_ID)
    elif dead_piece_ID < 16:
        ii_state.my_piece_list.remove(dead_piece_ID)
    else:
        print("ERROR:reduce_pattern(**_piece_listから削除)")

    # living_piece_colorから削除
    if dead_piece_ID < 8 and color_is_blue:  # 敵駒 and 駒が青色
        ii_state.living_piece_color[0] -= 1
    elif dead_piece_ID < 8 and not color_is_blue:  # 敵駒 and 駒が赤色
        ii_state.living_piece_color[1] -= 1
    elif dead_piece_ID >= 8 and color_is_blue:  # 自駒 and 駒が青色
        ii_state.living_piece_color[2] -= 1
    elif dead_piece_ID >= 8 and not color_is_blue:  # 自駒 and 駒が赤色
        ii_state.living_piece_color[3] -= 1
    else:
        print("ERROR:reduce_pattern(living_piece_colorから削除)")


# 相手の推測値を使って無難な手を選択(元祖)
# 価値が最大の行動番号を返す
def action_decision_legacy(model_path, ii_state):
    a, b, c = DN_INPUT_SHAPE  # (6, 6, 4)
    my_piece_set = set(ii_state.my_piece_list)
    enemy_piece_set = set(ii_state.enemy_piece_list)

    # 自分の駒配置を取得(確定)
    real_my_piece_blue_set = ii_state.real_my_piece_blue_set
    real_my_piece_red_set = ii_state.real_my_piece_red_set

    legal_actions = list(ii_state.legal_actions())

    actions_value_sum_list = np.array([0] * len(legal_actions), dtype="f4")

    convert_func = convert_func_use_in_guess(model_path)

    # 相手の70パターンについてforループ(自分のパターンは確定で計算)
    for num_and_enemy_blue in ii_state.enemy_estimated_num:
        enemy_blue_set = set(num_and_enemy_blue[1])
        enemy_red_set = enemy_piece_set - enemy_blue_set

        # 盤面を6*6*4次元の情報に変換
        ii_pieces_array = my_looking_create_state(
            ii_state,
            real_my_piece_blue_set,
            real_my_piece_red_set,
            enemy_blue_set,
            enemy_red_set,
        )

        policies = convert_func(ii_pieces_array, legal_actions)

        # 行列演算するためにndarrayに変換
        np_policies = np.array(policies, dtype="f4")

        # パターンごとに「推測値を重みとして掛けた方策」を足し合わせる
        actions_value_sum_list = actions_value_sum_list + (
            np_policies * num_and_enemy_blue[0]
        )

    best_action_index = np.argmax(actions_value_sum_list)  # 最大値のインデックスを取得
    best_action = legal_actions[best_action_index]  # 価値が最大の行動を取得
    return best_action


# 上位半分のみの世界しか考えない
def hoge_action_decision(model_path, ii_state):
    a, b, c = DN_INPUT_SHAPE  # (6, 6, 4)
    my_piece_set = set(ii_state.my_piece_list)
    enemy_piece_set = set(ii_state.enemy_piece_list)

    # 自分の駒配置を取得(確定)
    real_my_piece_blue_set = ii_state.real_my_piece_blue_set
    real_my_piece_red_set = ii_state.real_my_piece_red_set
    legal_actions = list(ii_state.legal_actions())
    actions_value_sum_list = np.array([0] * len(legal_actions), dtype="f4")
    convert_func = convert_func_use_in_guess(model_path)

    # 価値が上位の世界を抽出
    en_est_num = ii_state.enemy_estimated_num.copy()
    sorted_en_est_num = sorted(en_est_num, reverse=True, key=lambda x: x[0])
    sorted_en_est_num = sorted_en_est_num[0 : 1 + (len(sorted_en_est_num) // 4)]

    # 相手の70パターンについてforループ(自分のパターンは確定で計算)
    for num_and_enemy_blue in sorted_en_est_num:
        enemy_blue_set = set(num_and_enemy_blue[1])
        enemy_red_set = enemy_piece_set - enemy_blue_set

        # 盤面を6*6*4次元の情報に変換
        ii_pieces_array = my_looking_create_state(
            ii_state,
            real_my_piece_blue_set,
            real_my_piece_red_set,
            enemy_blue_set,
            enemy_red_set,
        )

        policies = convert_func(ii_pieces_array, legal_actions)

        # 行列演算するためにndarrayに変換
        np_policies = np.array(policies, dtype="f4")

        # パターンごとに「推測値を重みとして掛けた方策」を足し合わせる
        actions_value_sum_list = actions_value_sum_list + (
            np_policies * num_and_enemy_blue[0]
        )

    best_action_index = np.argmax(actions_value_sum_list)  # 最大値のインデックスを取得
    best_action = legal_actions[best_action_index]  # 価値が最大の行動を取得
    return best_action


# 1位の世界しか考えない
def action_decision(model_path, ii_state):
    a, b, c = DN_INPUT_SHAPE  # (6, 6, 4)
    my_piece_set = set(ii_state.my_piece_list)
    enemy_piece_set = set(ii_state.enemy_piece_list)

    # 自分の駒配置を取得(確定)
    real_my_piece_blue_set = ii_state.real_my_piece_blue_set
    real_my_piece_red_set = ii_state.real_my_piece_red_set
    legal_actions = list(ii_state.legal_actions())
    actions_value_sum_list = np.array([0] * len(legal_actions), dtype="f4")
    convert_func = convert_func_use_in_guess(model_path)

    # 価値が上位の世界を抽出
    en_est_num = ii_state.enemy_estimated_num.copy()
    sorted_en_est_num = sorted(en_est_num, reverse=True, key=lambda x: x[0])
    # 価値が1位の世界を抽出
    sorted_en_est_num = sorted_en_est_num[0:1]

    # 相手の70パターンについてforループ(自分のパターンは確定で計算)
    for num_and_enemy_blue in sorted_en_est_num:
        enemy_blue_set = set(num_and_enemy_blue[1])
        enemy_red_set = enemy_piece_set - enemy_blue_set

        # 盤面を6*6*4次元の情報に変換
        ii_pieces_array = my_looking_create_state(
            ii_state,
            real_my_piece_blue_set,
            real_my_piece_red_set,
            enemy_blue_set,
            enemy_red_set,
        )

        policies = convert_func(ii_pieces_array, legal_actions)

        # 行列演算するためにndarrayに変換
        np_policies = np.array(policies, dtype="f4")

        # パターンごとに「推測値を重みとして掛けた方策」を足し合わせる
        actions_value_sum_list = actions_value_sum_list + (
            np_policies * num_and_enemy_blue[0]
        )

    best_action_index = np.argmax(actions_value_sum_list)  # 最大値のインデックスを取得
    best_action = legal_actions[best_action_index]  # 価値が最大の行動を取得
    return best_action


# 駒をテレポート(デバッグ用で破壊的)(敵駒の存在を想定していない)
def teleport(ii_state, before, now):
    name = np.where(ii_state.all_piece == before)[0][0]
    ii_state.all_piece[name] = now


# 相手の打った手の価値(方策)を盤面ごとに調査し、ランキング形式でプリントする関数
# 実際の盤面は何位なのか、トップの盤面の形などを出力すると良い？
def value_ranking_by_board(beforehand_estimated_num, action_num, ii_state):
    # 行動番号からインデックスを取得
    enemy_legal_actions = sorted(list(ii_state.enemy_legal_actions()), key=lambda x: x)
    enemy_action_index = enemy_legal_actions.index(action_num)
    en_est_num = ii_state.enemy_estimated_num.copy()

    for index, (action_value_list, prenum_tuple) in enumerate(
        zip(beforehand_estimated_num, en_est_num)
    ):
        prenum_tuple[0] = action_value_list[enemy_action_index]

    # 価値の高い順にen_est_numをソートする
    sorted_en_est_num = sorted(en_est_num, reverse=True, key=lambda x: x[0])

    real_rank = -1
    for index, en_est in enumerate(sorted_en_est_num):
        if en_est[1] == tuple(ii_state.real_enemy_piece_blue_set):
            real_rank = index

    # print("real_set:", tuple(ii_state.real_enemy_piece_blue_set))
    # print("action_num:", action_num)
    print("real_rank:", real_rank)
    # print(len(sorted_en_est_num))
    print(sorted_en_est_num)

    # print(real_board_rank)
    # print(top_board)


# 行動の一連の処理でii_stateを更新する
def guess_enemy_piece_player(
    model_path, ii_state, before_tcp_str, now_tcp_str, gamma=default_gamma
):
    # 相手の盤面から全ての行動の推測値を計算しておく
    print("推測値を算出中")
    beforehand_estimated_num = enemy_ii_predict(model_path, ii_state)

    # 実際に取られた行動を取得
    print("相手の行動番号を取得中")
    BeforeAndNow = enemy_coordinate_checker(before_tcp_str, now_tcp_str)
    print(BeforeAndNow)
    enemy_action_num = calculate_enemy_action_number_from_coordinate(
        BeforeAndNow[0], BeforeAndNow[1]
    )
    print("敵の行動番号", enemy_action_num, sep=":")

    # 実際に取られた行動から推測値を更新
    print("推測値を更新中")
    update_predict_num_all(ii_state, beforehand_estimated_num, enemy_action_num, gamma)

    # 相手の行動からボードを更新
    print("ボード更新中")
    kill = update_II_state(ii_state, BeforeAndNow[0], BeforeAndNow[1])  # 相手の行動をボードに反映

    # 行動を決定
    print("行動を決定中")
    action_num = action_decision(model_path, ii_state)
    print("行動番号", action_num, sep=":")

    # 行動を受けて自分の推測値を更新
    # beforehand_my_estimated_num = my_ii_predict(model, ii_state)
    # (未実装)update_my_predict_num_all(ii_state, beforehand_my_estimated_num, action_num)

    # 自分の決定した行動でii_stateを更新
    ii_state.next(action_num)

    # 行動番号を返す
    return action_num


# デバッグ用(tcpを受けずに直接行動番号を受ける)(これを正式にして本家を分家にした方が良いのでは)
def guess_enemy_piece_player_for_debug(
    model_path, ii_state, just_before_enemy_action_num, gamma
):
    # 相手の盤面から全ての行動の推測値を計算しておく
    # print("推測値を算出中")
    beforehand_estimated_num = enemy_ii_predict(model_path, ii_state)

    # print("敵の行動番号", just_before_enemy_action_num, sep=":")
    # value_ranking_by_board(
    #     beforehand_estimated_num, just_before_enemy_action_num, ii_state
    # )
    # print("盤面：", ii_state)

    # print("〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜〜")
    # print("実際の青駒：", ii_state.real_enemy_piece_blue_set)
    # en_est_num = ii_state.enemy_estimated_num.copy()
    # print("未ソート推測値：", en_est_num)
    # sorted_en_est_num = sorted(en_est_num, reverse=True, key=lambda x: x[0])
    # print("推測値：", sorted_en_est_num)
    # print("beforehand_estimated_num:", beforehand_estimated_num)

    # value_ranking_by_board(
    #     beforehand_estimated_num, just_before_enemy_action_num, ii_state
    # )

    # 実際に取られた行動から推測値を更新
    # update_predict_num_all(
    #     ii_state, beforehand_estimated_num, just_before_enemy_action_num, gamma
    # )
    # print("beforehand", beforehand_estimated_num)
    update_predict_num_max_only(
        ii_state, beforehand_estimated_num, just_before_enemy_action_num, gamma
    )

    # 相手の行動からボードを更新
    # print("ボード更新中")
    before, now = action_to_coordinate(just_before_enemy_action_num)
    my_find_before = 35 - before  # このままでは相手視点の座標なので、自分視点の座標に変換
    my_find_now = 35 - now  # 同様に変換
    # print(my_find_before, my_find_now)
    kill = update_II_state(ii_state, my_find_before, my_find_now)  # 相手の行動をボードに反映

    # 行動を決定
    # print("行動を決定中")
    action_num = action_decision(model_path, ii_state)
    # print("行動番号", action_num, sep=":")

    # 行動を受けて自分の推測値を更新
    # beforehand_my_estimated_num = my_ii_predict(model_path, ii_state)
    # (未実装)update_my_predict_num_all(ii_state, beforehand_my_estimated_num, action_num)

    # 自分の決定した行動でii_stateを更新
    ii_state.next(action_num)

    # 行動番号を返す
    return action_num


# 動作確認
if __name__ == "__main__":
    start = time.time()
    # path = sorted(Path("./model").glob("*.h5"))[-1]
    # model = load_model(str(path))
    # ii_state = II_State({8, 9, 10, 11})

    # reduce_pattern(4, True, ii_state)

    # guess_enemy_piece_player(
    #     model,
    #     ii_state,
    #     "14R24R34R44R15B25B35B45B41u31u21u11u40u30u20u10u",
    #     "14R24R34R44R15B25B35B45B41u32u21u11u40u30u20u10u",
    # )

    teleport(ii_state, 10, 22)  # 10の敵駒を22に移動
    guess_enemy_piece_player(
        model,
        ii_state,
        "14R24R34R44R15B25B35B45B43u31u21u11u40u30u20u10u",
        "14R24R34R44R15B25B35B45B44u31u21u11u40u30u20u10u",
    )

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

