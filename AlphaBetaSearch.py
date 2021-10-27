from game import State, human_player_action, mcts_action
from ii_game import AccessableState

# 評価する深さ（絶対に奇数にする。偶数だと相手の駒の色を透視して判断してしまう。）
max_depth = 5
INFINITY = 10000

ev_table = [0] * 36

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
        # ここで数式を変更可能（デフォルトは1/r）
        ev_table[index] = 1 / r


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

    return value


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


# 動作確認
if __name__ == "__main__":

    print("起動")
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
            print(state)
            action = mcts_action(state)
        else:
            print(state)
            action = alpha_beta_action(state)

        # print(state)

        # 次の状態の取得
        state = state.next(action)
