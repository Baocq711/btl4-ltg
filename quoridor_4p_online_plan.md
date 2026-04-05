# Kế hoạch chi tiết triển khai Quoridor 4 người có chơi qua mạng

## 1. Mục tiêu đồ án

Dựa trên tiêu chí bài tập board game/turn-based, mục tiêu là xây dựng một game **Quoridor 4 người có chơi qua mạng**, đồng thời vẫn đáp ứng được các yêu cầu bắt buộc như:

- AI chơi random
- AI chơi theo Monte-Carlo Tree Search (MCTS)
- AI chơi theo Minimax
- Menu có **Continue** và tính năng này hoạt động được

Ngoài ra, game còn có thể tận dụng phần thưởng vì có nhiều hơn 1 đối thủ non-player nếu triển khai chế độ nhiều bot. 

---

## 2. Chốt hướng công nghệ

### 2.1. Ngôn ngữ khuyến nghị: Python

Khuyến nghị dùng **Python** thay vì C++.

### Lý do:
- Tốc độ phát triển nhanh hơn
- Dễ làm AI hơn
- Dễ làm networking qua WebSocket hơn
- Dễ test logic game hơn
- Dễ kịp deadline hơn so với C++

### Stack đề xuất:
- **UI / Client:** `pygame`
- **Online server:** `asyncio`, `websockets`
- **Save/Load:** `json`
- **Test:** `pytest`
- **Đóng gói:** `pyinstaller` nếu cần build exe

### Nếu bắt buộc muốn dùng C++:
- UI: `SFML`
- Networking: `Asio` hoặc `Boost.Asio`
- JSON: `nlohmann/json`
- Test: `Catch2`

Tuy nhiên độ khó sẽ cao hơn đáng kể.

---

## 3. Định hướng thiết kế tổng thể

### 3.1. Mục tiêu sản phẩm
Game sẽ có 2 nhánh chế độ chính:

#### A. Local / AI Mode
Dùng để đảm bảo hoàn thành yêu cầu môn học:
- 2 người hoặc 4 người local
- Người vs AI
- AI Random
- AI MCTS
- AI Minimax

#### B. Online Mode
Dùng để tạo điểm nhấn:
- 4 người chơi qua mạng
- Có lobby / room
- Có đồng bộ state
- Có save/load hoặc continue ở local mode

---

## 4. Quyết định kiến trúc quan trọng

### 4.1. Tách riêng Core Engine, UI, Network, AI
Không được viết luật game trực tiếp trong UI.

Phải tách thành 4 phần:

#### a. Core Engine
Chứa:
- trạng thái bàn cờ
- danh sách người chơi
- vị trí quân
- số tường còn lại
- lượt chơi hiện tại
- luật di chuyển
- luật đặt tường
- kiểm tra hợp lệ
- kiểm tra thắng/thua
- serialize/deserialize state

#### b. UI / Presentation
Chứa:
- render bàn cờ
- render quân cờ
- render tường
- menu
- button
- HUD
- scene management

#### c. Network
Chứa:
- server authoritative
- room manager
- WebSocket client
- protocol message

#### d. AI
Chứa:
- random AI
- minimax AI
- MCTS AI
- heuristic
- action pruning

### Lợi ích:
- core logic tái sử dụng cho local, AI, online
- test được không cần UI
- server có thể dùng chung engine với local
- dễ sửa bug hơn

---

## 5. Phạm vi chức năng nên làm

### 5.1. Chức năng bắt buộc
- Menu chính
- New Game
- Continue
- Chơi local
- Chơi online
- Save / Load
- Random AI
- MCTS AI
- Minimax AI
- Kiểm tra thắng/thua
- Hiển thị lượt chơi

### 5.2. Chức năng nên có thêm
- Lobby online
- Room code
- Ready / Start match
- Pause menu
- Timer mỗi lượt
- Replay đơn giản từ action log
- Sound effect cơ bản

### 5.3. Không nên ôm quá nhiều
- account/login
- matchmaking phức tạp
- database online
- ranking
- anti-cheat nâng cao
- animation quá nặng

---

## 6. Luật game cần chốt ngay

### 6.1. Bàn cờ
- Kích thước: **9x9**
- Người chơi đứng trên ô
- Tường đặt ở cạnh giữa các ô

### 6.2. Vị trí xuất phát cho 4 người
Đề xuất:
- Player 1: cạnh trên, mục tiêu đi xuống dưới
- Player 2: cạnh dưới, mục tiêu đi lên trên
- Player 3: cạnh trái, mục tiêu đi sang phải
- Player 4: cạnh phải, mục tiêu đi sang trái

### 6.3. Số tường
Bản 4 người nên giảm số tường mỗi người.

Khuyến nghị:
- **5 tường/người**

Có thể cân nhắc 6, nhưng 5 là hợp lý hơn để game không quá dài.

### 6.4. Luật đặt tường
Một tường hợp lệ khi:
- không nằm ngoài biên
- không chồng lên tường đã có
- không giao cắt tường theo kiểu không hợp lệ
- không làm bất kỳ người chơi nào mất hoàn toàn đường đến đích

### 6.5. Luật di chuyển quân
Phải hỗ trợ:
- đi 1 ô hợp lệ
- không đi xuyên tường
- nhảy qua quân khác khi đứng đối diện và phía sau không bị chặn
- đi chéo nếu không thể nhảy thẳng do tường chặn

---

## 7. Thiết kế dữ liệu lõi

### 7.1. Player
```python
from dataclasses import dataclass

@dataclass
class Player:
    id: int
    name: str
    x: int
    y: int
    walls_left: int
    goal: str  # "TOP", "BOTTOM", "LEFT", "RIGHT"
    active: bool = True
```

### 7.2. Wall
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Wall:
    x: int
    y: int
    orientation: str  # "H" hoặc "V"
```

### 7.3. GameState
```python
from dataclasses import dataclass

@dataclass
class GameState:
    board_size: int
    players: list[Player]
    horizontal_walls: set[tuple[int, int]]
    vertical_walls: set[tuple[int, int]]
    current_turn: int
    move_count: int
    winner_id: int | None
    mode: str  # LOCAL, AI, ONLINE
```

---

## 8. Cách biểu diễn bàn cờ

### Khuyến nghị: biểu diễn theo graph movement
Thay vì lưu bàn cờ như ma trận đơn giản, nên coi mỗi ô là một node trong graph.

Mỗi ô mặc định nối với các ô lân cận:
- trên
- dưới
- trái
- phải

Khi đặt tường, ta cắt bớt các cạnh tương ứng trong graph.

### Lợi ích:
- sinh nước đi dễ
- BFS tìm đường dễ
- validate đường đi sau khi đặt tường dễ
- AI dễ đánh giá khoảng cách đến đích

---

## 9. Action model

Mọi thao tác từ local click, AI, online đều phải quy về action.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class MovePawnAction:
    to_x: int
    to_y: int

@dataclass(frozen=True)
class PlaceWallAction:
    x: int
    y: int
    orientation: str
```

Không cho UI hay client tự sửa state trực tiếp.

Mọi thay đổi phải đi qua:

```python
def apply_action(state, action):
    ...
```

---

## 10. Luồng xử lý chuẩn

### 10.1. Local
1. Người chơi click
2. UI sinh action
3. Engine validate
4. Nếu hợp lệ thì apply
5. Update turn
6. Kiểm tra thắng/thua
7. Render lại

### 10.2. AI
1. Đến lượt bot
2. Bot đọc GameState
3. Bot sinh action
4. Engine validate
5. Apply action

### 10.3. Online
1. Client gửi action lên server
2. Server validate bằng core engine
3. Nếu hợp lệ thì apply
4. Server broadcast state mới cho tất cả client
5. Client render theo state từ server

---

## 11. Kiểm tra đường đi khi đặt tường

Đây là luật quan trọng nhất của Quoridor.

### Thuật toán:
Khi người chơi muốn đặt tường:
1. Clone state hiện tại
2. Thử thêm tường vào state clone
3. Với từng player, chạy BFS để xem còn đường tới goal không
4. Nếu có bất kỳ player nào không còn đường đi thì từ chối action

### BFS goal check
Ví dụ:
- goal `BOTTOM`: đạt đích khi `y == board_size - 1`
- goal `TOP`: đạt đích khi `y == 0`
- goal `LEFT`: đạt đích khi `x == 0`
- goal `RIGHT`: đạt đích khi `x == board_size - 1`

---

## 12. Sinh nước đi hợp lệ

### 12.1. Sinh nước đi quân
```python
def generate_pawn_moves(state: GameState, player_id: int) -> list[MovePawnAction]:
    ...
```

Hàm này phải xử lý:
- đi 1 bước thông thường
- chặn bởi tường
- nhảy qua quân đối diện
- đi chéo khi bị chặn nhảy thẳng

### 12.2. Sinh nước đặt tường
```python
def generate_wall_actions(state: GameState, player_id: int) -> list[PlaceWallAction]:
    ...
```

Mỗi action đặt tường phải qua validate.

### 12.3. Tổng hợp action hợp lệ
```python
def generate_legal_actions(state, player_id):
    return pawn_actions + wall_actions
```

---

## 13. Tối ưu không gian hành động cho AI

Quoridor có rất nhiều vị trí đặt tường, nếu duyệt toàn bộ sẽ rất chậm.

### Cách prune khuyến nghị
Chỉ xét các tường:
- gần shortest path của đối thủ
- gần shortest path của chính mình
- quanh khu vực trung tâm
- gần vị trí các player

### Quy trình prune:
1. Chạy BFS tìm shortest path cho từng player
2. Lấy các cạnh trên các đường đi quan trọng
3. Tạo danh sách candidate walls có khả năng chặn các cạnh đó
4. Giới hạn tối đa khoảng 20–40 wall candidates

Điều này cực kỳ quan trọng cho MCTS và Minimax.

---

## 14. Heuristic đánh giá trạng thái

Dùng cho Minimax và có thể dùng làm bias cho rollout của MCTS.

### Công thức gợi ý
```text
score =
+ w1 * (distance_opponents_sum - my_distance)
+ w2 * walls_left
+ w3 * mobility
+ w4 * center_control
- w5 * danger_of_being_blocked
```

### Ý nghĩa:
- `my_distance`: khoảng cách ngắn nhất của mình tới goal
- `distance_opponents_sum`: tổng hoặc trung bình khoảng cách của đối thủ
- `walls_left`: số tường còn lại
- `mobility`: số nước đi hợp lệ
- `center_control`: đứng gần trung tâm để linh hoạt hơn
- `danger_of_being_blocked`: nguy cơ bị kẹt hoặc mất lợi thế

### Trọng số ban đầu gợi ý
- `w1 = 10`
- `w2 = 2`
- `w3 = 1`
- `w4 = 1`
- `w5 = 3`

---

## 15. Thiết kế AI

## 15.1. AI Random
Mục tiêu:
- dễ làm
- test engine
- đáp ứng yêu cầu cơ bản

```python
import random

def choose_random_action(state, player_id):
    legal_actions = generate_legal_actions(state, player_id)
    return random.choice(legal_actions)
```

---

## 15.2. AI Minimax

### Sự thật quan trọng
Minimax chuẩn phù hợp nhất cho **2 người chơi**.

Với 4 người, nếu cố dùng Minimax thuần sẽ rất khó và chậm.

### Chiến lược an toàn nhất
- Hỗ trợ **Minimax + Alpha-Beta pruning cho mode 2 người**
- Với mode 4 người, dùng **MCTS** là hợp lý hơn

### Tại sao vẫn ổn?
Vì bài tập chỉ yêu cầu có AI Minimax, không bắt buộc mọi mode đều phải dùng Minimax.

### Cấu trúc đề xuất
```python
def minimax(state, depth, alpha, beta, maximizing_player_id):
    ...
```

### Tối ưu:
- giới hạn depth = 2 hoặc 3
- action ordering
- chỉ xét một phần wall candidates đã prune
- dùng heuristic ở node lá

### Nếu còn thời gian
Có thể nghiên cứu thêm **MaxN** cho 4 người:
- mỗi node trả về vector điểm `[p1, p2, p3, p4]`
- người chơi hiện tại maximize điểm của mình

Nhưng không nên coi đây là phần bắt buộc nếu muốn chắc deadline.

---

## 15.3. AI MCTS

MCTS sẽ là AI mạnh và phù hợp hơn cho Quoridor 4 người.

### 4 bước chính
1. **Selection**
2. **Expansion**
3. **Simulation**
4. **Backpropagation**

### Node structure
```python
class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = []
```

### UCT score
```python
score = (wins / visits) + c * sqrt(log(parent_visits) / visits)
```

### Reward cho 4 người
Khuyến nghị ban đầu:
- thắng: `1.0`
- không thắng: `0.0`

Nếu muốn đẹp hơn:
- hạng 1: `1.0`
- hạng 2: `0.5`
- hạng 3: `0.2`
- hạng 4: `0.0`

### Tối ưu rollout
Không nên rollout random hoàn toàn.

Có thể bias nhẹ:
- ưu tiên đi gần goal
- đôi khi mới đặt tường
- hạn chế rollout quá sâu

### Cấu hình ban đầu
- 300–1000 iterations mỗi lượt
hoặc
- 0.5–1.5 giây / lượt
- rollout depth giới hạn 20–40 bước

---

## 16. Online architecture

### 16.1. Mô hình server authoritative
Đây là mô hình nên dùng.

### Server chịu trách nhiệm:
- tạo phòng
- cho người chơi tham gia
- quản lý trạng thái phòng
- kiểm tra turn hợp lệ
- validate action
- apply action
- broadcast state
- xử lý disconnect

### Client chịu trách nhiệm:
- hiển thị UI
- gửi input người chơi
- nhận state từ server
- không tự quyết định state cuối cùng

### Lợi ích:
- tránh lệch state giữa client
- tránh client gian lận
- logic tập trung một nơi

---

## 17. Protocol mạng

### 17.1. Client -> Server
#### Join room
```json
{
  "type": "join_room",
  "room_id": "ABCD",
  "player_name": "Bao"
}
```

#### Ready
```json
{
  "type": "ready"
}
```

#### Move pawn
```json
{
  "type": "action",
  "action": {
    "kind": "move_pawn",
    "to_x": 4,
    "to_y": 1
  }
}
```

#### Place wall
```json
{
  "type": "action",
  "action": {
    "kind": "place_wall",
    "x": 3,
    "y": 4,
    "orientation": "H"
  }
}
```

### 17.2. Server -> Client
#### State update
```json
{
  "type": "state_update",
  "state": { }
}
```

#### Error
```json
{
  "type": "error",
  "message": "Invalid move"
}
```

#### Game started
```json
{
  "type": "game_started",
  "player_id": 2
}
```

#### Game over
```json
{
  "type": "game_over",
  "winner_id": 3
}
```

---

## 18. Save / Load / Continue

Theo yêu cầu, menu cần có **Continue** và phải chạy được.

### 18.1. Dữ liệu cần lưu
- board_size
- players
- wall list
- current_turn
- move_count
- mode
- move_history
- timestamp

### 18.2. Format lưu
Dùng JSON:

```json
{
  "board_size": 9,
  "players": [],
  "horizontal_walls": [],
  "vertical_walls": [],
  "current_turn": 2,
  "mode": "LOCAL_AI",
  "move_history": []
}
```

### 18.3. Continue local
Luồng đơn giản nhất:
- vào menu
- bấm Continue
- load file save gần nhất
- restore GameState
- tiếp tục chơi

### 18.4. Continue online
Không bắt buộc phải làm quá sâu.

Khuyến nghị:
- chỉ cần **Continue cho local mode** là đủ để đáp ứng tiêu chí
- online chỉ cần room reconnect ở mức cơ bản nếu còn thời gian

---

## 19. Menu flow

### 19.1. Main Menu
- New Game
- Continue
- Online
- Settings
- Quit

### 19.2. New Game
- 2 Players
- 4 Players
- Player vs AI
- Chọn AI type
- Chọn số bot

### 19.3. Online
- Host Room
- Join Room
- Nhập room code
- Ready

### 19.4. Pause Menu
- Resume
- Save
- Exit to menu

---

## 20. Cấu trúc project đề xuất

```text
quoridor/
├─ client/
│  ├─ main.py
│  ├─ scenes/
│  │  ├─ menu_scene.py
│  │  ├─ lobby_scene.py
│  │  ├─ game_scene.py
│  │  └─ pause_scene.py
│  ├─ ui/
│  │  ├─ button.py
│  │  ├─ panel.py
│  │  └─ hud.py
│  └─ net/
│     └─ ws_client.py
│
├─ server/
│  ├─ server.py
│  ├─ room_manager.py
│  └─ protocol.py
│
├─ core/
│  ├─ models.py
│  ├─ constants.py
│  ├─ game_state.py
│  ├─ rules.py
│  ├─ move_generator.py
│  ├─ pathfinding.py
│  ├─ validator.py
│  ├─ reducer.py
│  └─ serializer.py
│
├─ ai/
│  ├─ random_ai.py
│  ├─ heuristic.py
│  ├─ minimax_ai.py
│  ├─ mcts_ai.py
│  └─ action_pruning.py
│
├─ saves/
│  └─ ...
│
├─ tests/
│  ├─ test_rules.py
│  ├─ test_pathfinding.py
│  ├─ test_minimax.py
│  └─ test_mcts.py
│
└─ README.md
```

---

## 21. Kế hoạch triển khai theo giai đoạn

## Giai đoạn 1: Dựng core engine

### Việc cần làm
- Tạo `Player`, `Wall`, `GameState`
- Viết `generate_pawn_moves`
- Viết `generate_wall_actions`
- Viết `apply_action`
- Viết `check_winner`
- Viết `BFS pathfinding`
- Viết `validate_wall_placement`

### Mục tiêu đầu ra
- Chạy được bằng console
- Có thể mô phỏng 1 trận đơn giản bằng text
- Test được luật đi và đặt tường

### Ưu tiên
Đây là phần quan trọng nhất. Nếu core engine sai thì mọi phần còn lại đều lỗi theo.

---

## Giai đoạn 2: Local playable prototype

### Việc cần làm
- Dựng giao diện với pygame
- Vẽ board 9x9
- Vẽ player token
- Vẽ wall slot
- Click để di chuyển
- Click để đặt tường
- Hiển thị turn hiện tại
- Hiển thị walls_left
- Hiển thị winner

### Mục tiêu đầu ra
- 2 người local chơi trọn vẹn được
- Có thể pause / resume cơ bản

---

## Giai đoạn 3: Random AI

### Việc cần làm
- Tạo bot random
- Kết nối bot vào game loop
- Chế độ Human vs Random
- Chế độ nhiều bot local

### Mục tiêu đầu ra
- Bot luôn trả về action hợp lệ
- Có thể demo yêu cầu AI random

---

## Giai đoạn 4: MCTS AI

### Việc cần làm
- Tạo `MCTSNode`
- Selection / Expansion / Simulation / Backpropagation
- Tối ưu rollout policy
- Action pruning cho wall
- Time budget / iteration limit

### Mục tiêu đầu ra
- 4-player bot match chạy được
- Human vs MCTS chơi được
- Bot ra quyết định trong thời gian chấp nhận được

---

## Giai đoạn 5: Minimax AI

### Việc cần làm
- Minimax cho 2-player mode
- Alpha-beta pruning
- Heuristic evaluation
- Action ordering
- Giới hạn depth

### Mục tiêu đầu ra
- Human vs Minimax chơi ổn ở mode 2 người
- Có thể trình bày rõ trong báo cáo: Minimax áp dụng cho 2-player, MCTS áp dụng tốt hơn cho 4-player

---

## Giai đoạn 6: Save / Load / Continue

### Việc cần làm
- Serialize `GameState` ra JSON
- Load lại state từ JSON
- Tạo menu Continue
- Tự động load save gần nhất

### Mục tiêu đầu ra
- Save giữa trận
- Continue hoạt động đúng

---

## Giai đoạn 7: Online multiplayer

### Việc cần làm
- WebSocket server
- Room manager
- Join room
- Ready / Start
- Server validate action
- Broadcast state
- Client sync state
- Xử lý disconnect cơ bản

### Mục tiêu đầu ra
- 2 client hoặc 4 client chơi qua mạng được
- Không bị desync state

---

## Giai đoạn 8: Polish và chuẩn bị demo

### Việc cần làm
- sửa bug
- cải thiện UI
- thêm sound effect nếu có thời gian
- thêm timer nếu có thời gian
- tối ưu AI
- viết README
- chuẩn bị báo cáo / slide / video demo

### Mục tiêu đầu ra
- game chạy ổn định
- demo mượt
- thuyết trình rõ ràng

---

## 22. Timeline 4 tuần đề xuất

## Tuần 1
- chốt luật game
- dựng core engine
- test di chuyển
- test đặt tường
- test BFS

## Tuần 2
- dựng local UI
- local 2-player
- local 4-player
- random AI

## Tuần 3
- MCTS AI
- Minimax AI cho 2-player
- save/load
- continue menu

## Tuần 4
- online server/client
- lobby / room
- fix bug
- tối ưu demo
- chuẩn bị báo cáo

---

## 23. Rủi ro và cách giảm rủi ro

### Rủi ro 1: Sai luật nhảy / chéo
**Giải pháp:**
- viết unit test riêng cho nhiều case
- test từng luật tách biệt

### Rủi ro 2: Validate tường sai
**Giải pháp:**
- luôn simulate trước khi commit
- mọi wall placement đều phải qua BFS path check

### Rủi ro 3: AI quá chậm
**Giải pháp:**
- prune candidate walls
- giới hạn depth minimax
- giới hạn iterations/time budget MCTS

### Rủi ro 4: Online lệch state
**Giải pháp:**
- authoritative server
- client chỉ render state do server gửi

### Rủi ro 5: Không kịp Minimax 4 người
**Giải pháp:**
- Minimax triển khai chắc cho mode 2 người
- MCTS là AI chính cho 4 người

### Rủi ro 6: UI quá mất thời gian
**Giải pháp:**
- ưu tiên UI đơn giản nhưng rõ ràng
- không làm animation phức tạp giai đoạn đầu

---

## 24. Bộ test nên có

### 24.1. Test luật di chuyển
- đi 1 ô bình thường
- bị tường chặn không đi được
- nhảy qua player khác
- đi chéo khi không thể nhảy thẳng

### 24.2. Test đặt tường
- đặt ngoài biên
- chồng tường
- giao cắt tường không hợp lệ
- tường làm một player mất hết đường đi

### 24.3. Test thắng/thua
- chạm goal thì thắng
- chưa chạm goal thì chưa thắng

### 24.4. Test AI
- AI luôn sinh action hợp lệ
- AI không crash ở state đặc biệt
- AI ra quyết định trong giới hạn thời gian

### 24.5. Test network
- join room
- ready
- start game
- gửi action
- broadcast state
- disconnect

---

## 25. Thiết kế báo cáo và cách trình bày

### 25.1. Điểm nhấn khi trình bày
- Đây là board game có nhiều luật kiểm tra hợp lệ
- Có online multiplayer 4 người
- Có AI random
- Có AI MCTS
- Có AI Minimax
- Có Continue bằng save/load

### 25.2. Cách giải thích AI hợp lý
- Minimax hiệu quả hơn ở mode 2 người
- Quoridor 4 người có branching factor lớn, MCTS phù hợp hơn
- Hệ thống hỗ trợ nhiều chiến lược AI tùy mode

### 25.3. Cách demo đề xuất
#### Demo 1
- Main menu
- New Game
- Human vs Random AI

#### Demo 2
- Human vs Minimax trong 2-player

#### Demo 3
- 4-player với bot hoặc online room
- show MCTS hoạt động

#### Demo 4
- Pause -> Save -> Continue

---

## 26. Kết luận hướng triển khai cuối cùng

### Chốt phương án tốt nhất
- **Ngôn ngữ:** Python
- **UI:** pygame
- **Network:** asyncio + websockets
- **AI:** Random + Minimax (2-player) + MCTS (4-player)
- **Core engine:** tách riêng hoàn toàn
- **Save/Load:** JSON
- **Continue:** hỗ trợ local mode chắc chắn hoạt động
- **Online nổi bật:** 4 người chơi qua mạng

### Lý do chọn phương án này
- Thực tế nhất để hoàn thành đúng hạn
- Vẫn đáp ứng được tiêu chí bài tập
- Có điểm nhấn rõ ràng khi demo
- Dễ chia việc theo nhóm
- Dễ mở rộng nếu còn thời gian

---

## 27. Gợi ý phân chia công việc nếu làm nhóm

### Thành viên 1: Core Engine
- GameState
- rules
- move generator
- wall validator
- BFS pathfinding
- save/load

### Thành viên 2: UI / Client
- menu
- game scene
- rendering
- HUD
- input handling

### Thành viên 3: AI
- random AI
- heuristic
- minimax
- mcts
- benchmark AI

### Thành viên 4: Network
- websocket server
- room manager
- client sync
- protocol
- online testing

Nếu nhóm ít người hơn thì ghép:
- 1 người core + AI
- 1 người UI + network

---

## 28. Đề xuất thứ tự commit thực tế

1. `init project structure`
2. `implement game state models`
3. `implement pawn movement rules`
4. `implement wall placement validation`
5. `implement winner detection`
6. `add console simulation`
7. `add pygame board rendering`
8. `add local input handling`
9. `add random ai`
10. `add save load continue`
11. `add minimax ai`
12. `add mcts ai`
13. `add websocket server`
14. `add room system`
15. `add online sync`
16. `fix bugs and polish`

---

## 29. Phiên bản MVP tối thiểu nên đạt

Nếu thời gian quá gấp, MVP nên là:
- local playable
- 4-player support
- random AI
- MCTS hoặc Minimax ít nhất một loại chạy ổn
- save/load/continue
- online cơ bản hoặc local nhiều bot

Nhưng nếu muốn bám sát tiêu chí tốt nhất thì nên cố đạt đủ:
- Random AI
- MCTS AI
- Minimax AI
- Continue
- 4-player online hoặc 4-player local + online room

---

## 30. Khuyến nghị cuối

Để dự án thành công, thứ tự ưu tiên phải là:

1. **Core rules đúng**
2. **Playable local mode**
3. **Save/Continue**
4. **Random AI**
5. **MCTS / Minimax**
6. **Online**
7. **Polish UI**

Không nên làm online trước khi core engine ổn định.
Không nên làm AI trước khi luật game và action generator đã đúng.
Không nên đầu tư UI quá sớm.

Phương án tốt nhất để cân bằng giữa độ khó, tính ấn tượng và khả năng hoàn thành đúng hạn là:

> **Python + pygame + asyncio/websockets + core engine tách riêng + MCTS cho 4 người + Minimax cho 2 người + Continue bằng JSON save/load.**
