def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)
def check_winner(board, player):
    for row in board:
        if all([cell == player for cell in row]):
            return True
    for col in range(3):
        if all([board[row][col] == player for row in range(3)]):
            return True
    if all([board[i][i] == player for i in range(3)]):
        return True
    if all([board[i][2 - i] == player for i in range(3)]):
        return True
    return False

def is_full(board):
    for row in board:
        if " " in row:
            return False
    return True

def play_game():
    board = [[" " for _ in range(3)] for _ in range(3)]
    current_player = "X"
    
    while True:
        print_board(board)
        print("your turn{current_player}:")
        
        try:
            row = int(input("اختر الصف (0-2): "))
            col = int(input("اختر العمود (0-2): "))
        except ValueError:
            print("الرجاء إدخال أرقام صحيحة.")
            continue
        
        # التحقق من صحة المدخلات
        if row < 0 or row > 2 or col < 0 or col > 2 or board[row][col] != " ":
            print("الخلية محجوزة أو المدخل غير صحيح. حاول مرة أخرى.")
            continue
        
        # تحديث اللوحة
        board[row][col] = current_player
        
        # التحقق من الفائز
        if check_winner(board, current_player):
            print_board(board)
            print(f"اللاعب {current_player} فاز!")
            break
        
        # التحقق من تعادل
        if is_full(board):
            print_board(board)
            print("التعادل!")
            break
        
        # التبديل بين اللاعبين
        current_player = "O" if current_player == "X" else "X"
# تشغيل اللعبة
play_game()
