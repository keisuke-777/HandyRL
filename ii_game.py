# Stateからアクセスできる情報を抽出したState（毎ターンstateから生成する）
# 用途はアルファベータ探索で使用するのみにとどめ、状態管理などはstateで行う
class AccessableState:
    def __init__(self):
        self.is_goal = False
        self.enemy_is_goal = False
        self.enemy_left_blue_piece = 0
        self.enemy_left_red_piece = 0
        self.my_left_blue_piece = 0
        self.my_left_red_piece = 0
        self.pieces = [0] * 36
        self.my_turn = True
        self.depth = 0

    def create_ii_state_from_state(self, state):
        self.is_goal = False
        self.enemy_is_goal = False

        # 残っている駒の情報（これは公開されている情報）
        self.enemy_left_blue_piece = 0
        self.enemy_left_red_piece = 0
        self.my_left_blue_piece = 0
        self.my_left_red_piece = 0

        # 駒の情報は1つのボードに集める
        self.pieces = [0] * 36
        for i, piece in enumerate(state.pieces):
            if piece != 0:
                if piece == 1:
                    self.my_left_blue_piece += 1
                elif piece == 2:
                    self.my_left_red_piece += 1
                self.pieces[i] = state.pieces[i]
        for i, piece in enumerate(state.enemy_pieces):
            if piece != 0:
                if piece == 1:
                    self.enemy_left_blue_piece += 1
                elif piece == 2:
                    self.enemy_left_red_piece += 1
                # 表示上は全て青駒とする
                self.pieces[35 - i] = -1
        self.my_turn = True
        self.depth = state.depth

    def overwrite_from_ii_state(self, ii_state):
        self.is_goal = ii_state.is_goal
        self.enemy_is_goal = ii_state.enemy_is_goal
        self.enemy_left_blue_piece = ii_state.enemy_left_blue_piece
        self.enemy_left_red_piece = ii_state.enemy_left_red_piece
        self.my_left_blue_piece = ii_state.my_left_blue_piece
        self.my_left_red_piece = ii_state.my_left_red_piece
        self.pieces = ii_state.pieces.copy()
        self.my_turn = ii_state.my_turn
        self.depth = ii_state.depth

    # 引き分けはゲーム側で判定するため実装しない
    def is_lose(self):
        return (
            self.my_left_blue_piece <= 0
            or self.enemy_left_red_piece <= 0
            or self.enemy_is_goal
        )

    def is_win(self):
        # 青駒を全て食べるのはあり得ない（探索中は駒を食べると赤になるため）
        return self.my_left_red_piece <= 0 or self.is_goal

    # 駒の移動元と移動方向を行動に変換
    def position_to_action(self, position, direction):
        return position * 4 + direction

    # 行動を駒の移動元と移動方向に変換
    def action_to_position(self, action):
        return (int(action / 4), action % 4)  # position,direction

    def legal_actions(self):
        actions = []
        if self.my_turn:  # 自分のターン
            for p in range(36):
                if self.pieces[p] in (1, 2):
                    # 自分の駒が存在するなら駒の位置を渡して、その駒の取れる行動をactionsに追加
                    actions.extend(self.legal_actions_pos(p))
            # 青駒のゴール行動は例外的に合法手リストに追加
            if self.pieces[0] == 1:
                actions.extend([2])  # 0*4 + 2
            if self.pieces[5] == 1:
                actions.extend([22])  # 5*4 + 2
        else:
            for p in range(36):
                if self.pieces[p] in (-1, -2):
                    actions.extend(self.enemy_legal_actions_pos(p))
            # ゴール行動は例外的に合法手リストに追加（相手の駒は全てゴール可能とする）
            if self.pieces[30] == -1:
                actions.extend([120])  # 30*4 + 0
            if self.pieces[35] == -1:
                actions.extend([140])  # 35*4 + 0
        return actions

    # 駒ごと(駒1つに着目した)の合法手のリストの取得
    def legal_actions_pos(self, position):
        actions = []
        x = position % 6
        y = int(position / 6)
        # 下左上右の順に行動できるか検証し、できるならactionに追加
        if y != 5 and self.pieces[position + 6] not in (1, 2):  # 下端でない and 下に自分の駒がいない
            actions.append(self.position_to_action(position, 0))
        if x != 0 and self.pieces[position - 1] not in (1, 2):  # 左端でない and 左に自分の駒がいない
            actions.append(self.position_to_action(position, 1))
        if y != 0 and self.pieces[position - 6] not in (1, 2):  # 上端でない and 上に自分の駒がいない
            actions.append(self.position_to_action(position, 2))
        if x != 5 and self.pieces[position + 1] not in (1, 2):  # 右端でない and 右に自分の駒がいない
            actions.append(self.position_to_action(position, 3))
        # 青駒のゴール行動の可否は1ターンに1度だけ判定すれば良いので、例外的にlegal_actionsで処理する(ここでは処理しない)
        return actions

    # 敵視点でのlegal_actions_pos
    def enemy_legal_actions_pos(self, position):
        actions = []
        x = position % 6
        y = int(position / 6)
        # 下左上右の順に行動できるか検証し、できるならactionに追加
        if y != 5 and self.pieces[position + 6] not in (-1, -2):  # 下端でない and 下に自分の駒がいない
            actions.append(self.position_to_action(position, 0))
        if x != 0 and self.pieces[position - 1] not in (-1, -2):  # 左端でない and 左に自分の駒がいない
            actions.append(self.position_to_action(position, 1))
        if y != 0 and self.pieces[position - 6] not in (-1, -2):  # 上端でない and 上に自分の駒がいない
            actions.append(self.position_to_action(position, 2))
        if x != 5 and self.pieces[position + 1] not in (-1, -2):  # 右端でない and 右に自分の駒がいない
            actions.append(self.position_to_action(position, 3))
        # 青駒のゴール行動の可否は1ターンに1度だけ判定すれば良いので、例外的にlegal_actionsで処理する(ここでは処理しない)
        return actions

    # 敵駒を取る行動とゴールに近づく行動以外を排除したlegal_actions_pos
    def reduced_legal_actions_pos(self, position):
        actions = []
        x = position % 6
        y = int(position / 6)
        if x < 3:  # 左のゴールに近い
            if x != 0 and self.pieces[position - 1] not in (1, 2):
                actions.append(self.position_to_action(position, 1))  # 左
            if x != 5 and self.pieces[position + 1] == -1:  # 右に敵の駒が存在
                actions.append(self.position_to_action(position, 3))  # 右
        else:  # 右のゴールに近い
            if x != 5 and self.pieces[position + 1] not in (1, 2):
                actions.append(self.position_to_action(position, 3))  # 右
            if x != 0 and self.pieces[position - 1] == -1:  # 左に敵の駒が存在
                actions.append(self.position_to_action(position, 1))  # 左

        if y != 0 and self.pieces[position - 6] not in (1, 2):
            actions.append(self.position_to_action(position, 2))  # 上
        if y != 5 and self.pieces[position + 6] == -1:  # 下に敵の駒が存在
            actions.append(self.position_to_action(position, 0))  # 下
        return actions

    # 敵駒を取る行動とゴールに近づく行動以外を排除したenemy_legal_actions_pos
    def reduced_enemy_legal_actions_pos(self, position):
        actions = []
        x = position % 6
        y = int(position / 6)
        if x < 3:  # 左のゴールに近い
            if x != 0 and self.pieces[position - 1] != -1:
                actions.append(self.position_to_action(position, 1))  # 左
            if x != 5 and self.pieces[position + 1] in (1, 2):  # 右に相手の駒が存在
                actions.append(self.position_to_action(position, 3))  # 右
        else:  # 右のゴールに近い
            if x != 5 and self.pieces[position + 1] != -1:
                actions.append(self.position_to_action(position, 3))  # 右
            if x != 0 and self.pieces[position - 1] in (1, 2):  # 左に敵の駒が存在
                actions.append(self.position_to_action(position, 1))  # 左

        if y != 5 and self.pieces[position + 6] != -1:
            actions.append(self.position_to_action(position, 0))  # 下
        if y != 0 and self.pieces[position - 6] in (1, 2):  # 上に敵の駒が存在
            actions.append(self.position_to_action(position, 2))  # 上
        return actions

    # 次の状態の取得
    def next(self, action):
        ii_state = AccessableState()
        ii_state.overwrite_from_ii_state(self)
        if ii_state.my_turn:  # 自分のターン
            # position_bef->移動前の駒の位置、position_aft->移動後の駒の位置
            # 行動を(移動元, 移動方向)に変換
            position_bef, direction = ii_state.action_to_position(action)

            # 合法手がくると仮定
            # 駒の移動(後ろに動くことは少ないかな？ + if文そんなに踏ませたくないな と思ったので判定を左右下上の順番にしてるけど意味あるのかは不明)
            if direction == 1:  # 左
                position_aft = position_bef - 1
            elif direction == 3:  # 右
                position_aft = position_bef + 1
            elif direction == 0:  # 下
                position_aft = position_bef + 6
            elif direction == 2:  # 上
                if position_bef == 0 or position_bef == 5:  # 0と5の上行動はゴール処理なので先に弾く
                    ii_state.is_goal = True
                    position_aft = position_bef  # position_befを入れて駒の場所を動かさない(勝敗は決しているので下手に動かさない方が良いと考えた)
                else:
                    position_aft = position_bef - 6
            else:
                print("error関数名:next")

            # if position_aft > 35:
            #     print(ii_state.pieces)
            #     print(ii_state.depth)
            #     print(action)

            # 倒した駒を反映（倒した駒は全て赤駒として扱う）
            if ii_state.pieces[position_aft] != 0:
                ii_state.enemy_left_red_piece -= 1

            # 実際に駒移動
            ii_state.pieces[position_aft] = ii_state.pieces[position_bef]
            ii_state.pieces[position_bef] = 0

            ii_state.my_turn = False
            ii_state.depth += 1
            return ii_state

        else:  # 敵のターン
            position_bef, direction = ii_state.action_to_position(action)
            if direction == 1:  # 左
                position_aft = position_bef - 1
            elif direction == 3:  # 右
                position_aft = position_bef + 1
            elif direction == 0:  # 下
                if position_bef == 30 or position_bef == 35:  # 30と35の下行動はゴール処理なので先に弾く
                    ii_state.enemy_is_goal = True
                    position_aft = position_bef
                else:
                    position_aft = position_bef + 6
            elif direction == 2:  # 上
                position_aft = position_bef - 6
            else:
                print("error関数名:next")

            # 倒した駒を反映
            if ii_state.pieces[position_aft] != 0:
                if ii_state.pieces[position_aft] == 1:
                    ii_state.my_left_blue_piece -= 1
                elif ii_state.pieces[position_aft] == 2:
                    ii_state.my_left_red_piece -= 1

            # 実際に駒移動
            ii_state.pieces[position_aft] = ii_state.pieces[position_bef]
            ii_state.pieces[position_bef] = 0

            ii_state.my_turn = True
            ii_state.depth += 1
            return ii_state

    # 文字列表示
    def __str__(self):
        row = "|{}|{}|{}|{}|{}|{}|"
        hr = "\n-------------------------------\n"
        board_essence = []
        for i in self.pieces:
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
