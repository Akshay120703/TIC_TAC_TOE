import streamlit as st
import joblib
import numpy as np
import streamlit.components.v1 as components

# Load the trained model and LabelEncoder
model = joblib.load('tic_tac_toe_model.pkl')
le = joblib.load('label_encoder.pkl')

st.title('Tic-Tac-Toe with AI')
st.write('Play against the AI!')

# Initialize board state
if 'board' not in st.session_state:
    st.session_state.board = [''] * 9
    st.session_state.current_player = 'X'

# Reset game function
def reset_game():
    st.session_state.board = [''] * 9
    st.session_state.current_player = 'X'

# Handle cell click
def handle_click(index):
    if st.session_state.board[index] == '':
        st.session_state.board[index] = st.session_state.current_player
        if check_winner(st.session_state.board, st.session_state.current_player):
            st.success(f'Player {st.session_state.current_player} wins!')
            st.session_state.current_player = None
        elif '' not in st.session_state.board:
            st.info('It\'s a draw!')
            st.session_state.current_player = None
        else:
            st.session_state.current_player = 'O' if st.session_state.current_player == 'X' else 'X'
            if st.session_state.current_player == 'O':
                ai_move()

# AI Move function
def ai_move():
    board_state = np.array(st.session_state.board)
    board_state[board_state == ''] = 'b'
    board_state[board_state == 'X'] = 'x'
    board_state[board_state == 'O'] = 'o'
    board_encoded = le.transform(board_state)
    board_encoded = board_encoded.reshape(1, -1)
    pred = model.predict(board_encoded)
    for i in range(9):
        if st.session_state.board[i] == '' and pred[0] == True:
            st.session_state.board[i] = 'O'
            break
    if check_winner(st.session_state.board, 'O'):
        st.success('AI wins!')
        st.session_state.current_player = None
    elif '' not in st.session_state.board:
        st.info('It\'s a draw!')
        st.session_state.current_player = None

# Check winner function
def check_winner(board, player):
    winning_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    for condition in winning_conditions:
        if all(board[i] == player for i in condition):
            return True
    return False

# Display the board
st.write('Current Player: ', st.session_state.current_player)
for i in range(3):
    cols = st.columns(3)
    for j in range(3):
        index = i * 3 + j
        cols[j].button(st.session_state.board[index], key=index, on_click=handle_click, args=(index,))

# Reset game button
if st.button('Reset Game'):
    reset_game()

# Load and render the HTML file
with open("static/index.html", 'r', encoding='utf-8') as file:
    html_content = file.read()
components.html(html_content, height=600)
