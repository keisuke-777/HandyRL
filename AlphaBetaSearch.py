from game import State, human_player_action, mcts_action, random_action
from ii_game import AccessableState
import time

# 評価する深さ
# （絶対に奇数にする。偶数だと相手の駒の色を透視して判断してしまう。）-> おそらく修正できているので偶数にしても大丈夫
max_depth = 7
INFINITY = 10000
n = 1


ev_table = [0] * 36
threshold = [-INFINITY, INFINITY]

# ev_tableを最初に自動で生成する
def create_ev_table(ev_table):
    x = y = r = 0
    for index, _ in enumerate(ev_table):
        x = index % 6
        y = int(index / 6)
        if x < 3:
            r = x + 1 + y
        else:
            r = 6 - x + y
        # FIXME: ここで数式を変更可能（デフォルトは1/r）
        ev_table[index] = 1 / (r ** n)


def evaluate_board_state(ii_state):
    value = 0
    # 自分の青ゴマが敵のゴールにどれだけ近いか
    for index, piece in enumerate(ii_state.pieces):
        if piece == 1:
            value += ev_table[index]
    # 敵のコマがどれだけ自分のゴールに近いか
    for index, piece in enumerate(ii_state.pieces):
        if piece != 0:
            value -= ev_table[35 - index]

    if ii_state.my_turn:
        return value
    else:
        return -value


# アルファベータ法で状態価値計算
def alpha_beta(ii_state, alpha, beta, search_depth):
    # 負けた場合は無条件で評価値最低にする
    if ii_state.is_lose():
        if ii_state.my_turn:
            return -INFINITY
        else:
            return INFINITY

    if ii_state.is_win():
        if ii_state.my_turn:
            return INFINITY
        else:
            return -INFINITY

    # 規定の深さまで来たら、探索を打ち切り状態を評価する
    if search_depth == max_depth:
        return evaluate_board_state(ii_state)

    # 合法手の状態価値の計算
    for action in ii_state.legal_actions():
        score = -alpha_beta(ii_state.next(action), -beta, -alpha, search_depth + 1)
        if score > alpha:
            alpha = score
            # TODO: スレッショルドの導入

        # 現ノードのベストスコアが親ノードを超えたら探索終了
        if alpha >= beta:
            return alpha

    # 合法手の状態価値の最大値を返す
    return alpha


# アルファベータ法で行動選択
def alpha_beta_action(state):
    if 2 in state.legal_actions():
        return 2
    if 22 in state.legal_actions():
        return 22

    ii_state = AccessableState()
    ii_state.create_ii_state_from_state(state)
    # 合法手の状態価値の計算
    best_action = 0
    alpha = -INFINITY
    for action in ii_state.legal_actions():
        score = -alpha_beta(ii_state.next(action), -INFINITY, -alpha, 0)
        if score > alpha:
            best_action = action
            alpha = score

    # 合法手の状態価値の最大値を持つ行動を返す
    return best_action


# アルファベータ法における閾値を計算する
def calculate_threshold(ii_state):
    standard_value = evaluate_board_state(ii_state)
    val_dif = find_pieces_closer_to_the_goal(ii_state, True)
    return


# 評価値の変動量を計算する
def find_pieces_closer_to_the_goal(ii_state, my_view=True):
    # [value, coordinate]
    nearest_piece = [-1, -1]
    sec_nearest_piece = [-1, -1]
    if my_view:
        pieces = ii_state.pieces.copy()
        num_of_act = max_depth // 2 + 1
    else:
        pieces = [0] * 36
        for i, piece in enumerate(ii_state.pieces):
            if piece == -1:
                pieces[35 - i] = 1
        num_of_act = max_depth // 2

    for i, piece in enumerate(ii_state.pieces):
        if piece == 1:
            if ev_table[i] > nearest_piece[0]:
                sec_nearest_piece = nearest_piece.copy()
                nearest_piece = [ev_table[i], i]
            elif ev_table[i] > near_piece_val[1]:
                sec_nearest_piece = [ev_table[i], i]

    x = nearest_piece[1] % 6
    y = int(nearest_piece[1] / 6)
    if x < 3:
        r = x + 1 + y
    else:
        r = 6 - x + y
    # FIXME: 評価値の変動量（閾値に直結）を決める式を変更する
    if r > num_of_act:
        return 1 / ((r - num_of_act) ** n) - 1 / (r ** n)
    return 1 - 1 / (r ** n)  # 仮決め


# 動作確認
if __name__ == "__main__":

    print("起動中...")
    # 状態の生成
    state = State()
    create_ev_table(ev_table)

    # ゲーム終了までのループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 行動の取得
        if state.is_first_player():
            # action = random_action(state)
            action = human_player_action(state)
        else:
            # action = human_player_action(state)
            # action = mcts_action(state)
            start = time.time()
            print("思考中...")
            action = alpha_beta_action(state)
            print("思考時間:{0}".format(time.time() - start) + "[sec]",)
        # 次の状態の取得
        state = state.next(action)
        print(state)
