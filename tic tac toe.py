import socket
import json
import threading
import time
import pygame
import sys
import random
import os
from typing import List, Dict, Tuple, Optional, Union
import numpy as np  # Added for stats tracking

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (0, 0, 128)
PURPLE = (128, 0, 128)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
LIGHT_GREEN = (144, 238, 144)
PINK = (255, 192, 203)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (100, 100, 100)

# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 700
BOARD_SIZE = 450
CELL_SIZE = BOARD_SIZE // 3

# Fonts
FONT = pygame.font.SysFont('Arial', 24)
LARGE_FONT = pygame.font.SysFont('Arial', 40)
SMALL_FONT = pygame.font.SysFont('Arial', 18)
TINY_FONT = pygame.font.SysFont('Arial', 14)

# Game states
STATE_MENU = "menu"
STATE_PLAY_AI = "play_ai"
STATE_PLAY_NETWORK = "play_network"
STATE_WAITING = "waiting"
STATE_TRAINING = "training"
STATE_TRAINING_STATS = "training_stats"
STATE_RULES = "rules"  # New state for game rules
STATE_HELP = "help"    # New state for help/guide

# Game constants
AI_DIFFICULTY_EASY = 0.5    # Higher epsilon = more random moves
AI_DIFFICULTY_MEDIUM = 0.3
AI_DIFFICULTY_HARD = 0.1

class TicTacToe:
    """Core Tic-Tac-Toe game logic"""
    
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 'X'  # X always starts
        self.winner = None
        self.game_over = False
        self.winning_line = None  # Store winning line for highlighting
        self.move_history = []    # Track moves for undo feature
        
    def reset(self):
        """Reset the game board"""
        self.board = [' '] * 9
        self.current_player = 'X'
        self.winner = None
        self.game_over = False
        self.winning_line = None
        self.move_history = []
        return self.board.copy()
        
    def print_board(self):
        """Display the game board (console version)"""
        print("\n")
        for i in range(3):
            print(f" {self.board[i*3]} | {self.board[i*3+1]} | {self.board[i*3+2]} ")
            if i < 2:
                print("---+---+---")
        print("\n")
    
    def get_valid_moves(self):
        """Get current valid moves"""
        return [i for i, cell in enumerate(self.board) if cell == ' ']
    
    def check_winner(self):
        """Check if there's a winner and return the winning line"""
        # Horizontal rows
        for i in range(0, 9, 3):
            if self.board[i] != ' ' and self.board[i] == self.board[i+1] == self.board[i+2]:
                self.winning_line = [(i, i+1, i+2)]
                return self.board[i]
        
        # Vertical columns
        for i in range(3):
            if self.board[i] != ' ' and self.board[i] == self.board[i+3] == self.board[i+6]:
                self.winning_line = [(i, i+3, i+6)]
                return self.board[i]
        
        # Diagonals
        if self.board[0] != ' ' and self.board[0] == self.board[4] == self.board[8]:
            self.winning_line = [(0, 4, 8)]
            return self.board[0]
        if self.board[2] != ' ' and self.board[2] == self.board[4] == self.board[6]:
            self.winning_line = [(2, 4, 6)]
            return self.board[2]
        
        return None
    
    def is_full(self):
        """Check if the board is full"""
        return ' ' not in self.board
    
    def undo_move(self):
        """Undo the last move if possible"""
        if not self.move_history:
            return False
        
        # Get the last move
        last_move = self.move_history.pop()
        
        # Restore the board state
        self.board[last_move] = ' '
        
        # Switch player back
        self.current_player = 'O' if self.current_player == 'X' else 'X'
        
        # Reset game over state
        self.game_over = False
        self.winner = None
        self.winning_line = None
        
        return True
    
    def step(self, action):
        """
        Make a move and return the new state, reward, and whether the game is over
        """
        # If the game is already over or invalid action
        if self.game_over or action < 0 or action >= 9 or self.board[action] != ' ':
            return self.board.copy(), -10, True
        
        # Record the move for undo feature
        self.move_history.append(action)
        
        # Make the move
        self.board[action] = self.current_player
        
        # Check if there's a winner
        self.winner = self.check_winner()
        
        # Calculate reward
        reward = 0
        if self.winner:
            self.game_over = True
            # Positive reward if AI (O) wins, negative if opponent (X) wins
            reward = 1 if self.winner == 'O' else -1
        elif self.is_full():
            self.game_over = True
            # Small reward for a tie
            reward = 0.1
        
        # Change player if the game is not over
        if not self.game_over:
            self.current_player = 'O' if self.current_player == 'X' else 'X'
        
        return self.board.copy(), reward, self.game_over


class SimpleQLearningAI:
    """Q-learning AI for Tic-Tac-Toe using a dictionary-based approach"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.difficulty = AI_DIFFICULTY_MEDIUM  # Default difficulty
        
        # Q-table is a dictionary: {state_key: [q_values for all actions]}
        self.q_table = {}
        
        # Load model if exists
        self.model_file = "tictactoe_qtable.txt"
        self.load_model()
    
    def set_difficulty(self, difficulty_level):
        """Set AI difficulty level by adjusting epsilon"""
        self.difficulty = difficulty_level
    
    def get_state_key(self, state):
        """Convert board state to a unique string key"""
        return ''.join([cell if cell != ' ' else '-' for cell in state])
    
    def get_q_values(self, state):
        """Get Q-values for a state, or initialize them if not seen before"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0] * 9  # Initialize with zeros
        return self.q_table[state_key]
    
    def get_action(self, state, training=False):
        """Choose an action using epsilon-greedy policy"""
        valid_moves = [i for i, cell in enumerate(state) if cell == ' ']
        
        if not valid_moves:
            return -1  # No valid moves
        
        # Use difficulty as epsilon for non-training mode
        actual_epsilon = self.epsilon if training else self.difficulty
        
        # Exploration with probability epsilon
        if random.random() < actual_epsilon:
            return random.choice(valid_moves)
        
        # Exploitation: choose best action among valid moves
        q_values = self.get_q_values(state)
        
        # Filter valid moves
        valid_q_values = [(i, q_values[i]) for i in valid_moves]
        valid_q_values.sort(key=lambda x: x[1], reverse=True)
        
        return valid_q_values[0][0]  # Return action with highest Q-value
    
    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-value for a state-action pair"""
        q_values = self.get_q_values(state)
        
        if done:
            # Terminal state
            q_target = reward
        else:
            # Non-terminal state
            next_q_values = self.get_q_values(next_state)
            max_next_q = max([next_q_values[i] for i in range(9) if next_state[i] == ' '])
            q_target = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        q_values[action] = q_values[action] + self.learning_rate * (q_target - q_values[action])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self):
        """Save Q-table to a file"""
        with open(self.model_file, 'w') as f:
            for state_key, q_values in self.q_table.items():
                q_values_str = ','.join([str(val) for val in q_values])
                f.write(f"{state_key}:{q_values_str}\n")
        print(f"Model saved to {self.model_file}")
    
    def load_model(self):
        """Load Q-table from a file if it exists"""
        try:
            with open(self.model_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        state_key, q_values_str = line.strip().split(':')
                        q_values = [float(val) for val in q_values_str.split(',')]
                        self.q_table[state_key] = q_values
            print("Q-table loaded from file.")
            # Reduce epsilon for a pre-trained model
            self.epsilon = max(0.1, self.epsilon * 0.1)
        except FileNotFoundError:
            print("No existing Q-table found. Starting with an empty table.")


class GameServer:
    """Server for hosting Tic-Tac-Toe games"""
    
    def __init__(self, host='localhost', port=5556):
        self.host = host
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.clients = []
        self.games = {}  # Dictionary to track games
        self.is_running = False
    
    def start(self):
        """Start the server"""
        try:
            self.server.bind((self.host, self.port))
            self.server.listen(5)
            self.is_running = True
            print(f"Server started on {self.host}:{self.port}")
            
            # Accept connections in a separate thread
            accept_thread = threading.Thread(target=self.accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
            return True
        except Exception as e:
            print(f"Error starting server: {e}")
            return False
    
    def stop(self):
        """Stop the server"""
        self.is_running = False
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        try:
            self.server.close()
        except:
            pass
        print("Server stopped")
    
    def accept_connections(self):
        """Accept incoming connections"""
        self.server.settimeout(1)  # 1 second timeout to check is_running flag
        
        while self.is_running:
            try:
                client, address = self.server.accept()
                print(f"New connection from {address}")
                
                # Start a new thread to handle this client
                client_thread = threading.Thread(target=self.handle_client, args=(client,))
                client_thread.daemon = True
                client_thread.start()
                
                self.clients.append(client)
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error accepting connection: {e}")
                if not self.is_running:
                    break
        
        print("Stopped accepting connections")
    
    def handle_client(self, client):
        """Handle communication with a client"""
        # Create a new game or join an existing one
        game_id = None
        player_symbol = None
        
        # Find a game with only one player or create a new one
        for gid, game_info in self.games.items():
            if len(game_info['players']) < 2:
                game_id = gid
                player_symbol = 'O'  # Second player is O
                game_info['players'].append(client)
                break
        
        if game_id is None:
            # Create a new game
            game_id = len(self.games) + 1
            player_symbol = 'X'  # First player is X
            self.games[game_id] = {
                'players': [client],
                'board': [' '] * 9,
                'current_player': 'X',
                'game_over': False
            }
        
        # Send initial game state to client
        self.send_game_state(client, game_id, player_symbol)
        
        # If game has two players, notify both that the game has started
        if len(self.games[game_id]['players']) == 2:
            self.broadcast_game_state(game_id, "start")
        
        # Handle client messages
        try:
            while self.is_running:
                data = client.recv(1024).decode('utf-8')
                if not data:
                    break
                
                try:
                    message = json.loads(data)
                    
                    if 'move' in message and game_id in self.games:
                        game_info = self.games[game_id]
                        
                        # Check if it's this player's turn
                        if (player_symbol == game_info['current_player'] and 
                            not game_info['game_over']):
                            
                            move = message['move'] - 1  # Convert 1-9 to 0-8
                            
                            # Validate move
                            if 0 <= move < 9 and game_info['board'][move] == ' ':
                                # Make the move
                                game_info['board'][move] = player_symbol
                                
                                # Check for winner
                                winner = self.check_winner(game_info['board'])
                                if winner:
                                    game_info['game_over'] = True
                                    self.broadcast_game_state(game_id, "win", winner)
                                elif ' ' not in game_info['board']:
                                    game_info['game_over'] = True
                                    self.broadcast_game_state(game_id, "tie")
                                else:
                                    # Switch player
                                    game_info['current_player'] = 'O' if game_info['current_player'] == 'X' else 'X'
                                    self.broadcast_game_state(game_id)
                
                except json.JSONDecodeError:
                    print(f"Invalid JSON received: {data}")
                except Exception as e:
                    print(f"Error processing message: {e}")
        
        except ConnectionResetError:
            print("Client disconnected abruptly")
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            # Clean up when client disconnects
            if client in self.clients:
                self.clients.remove(client)
            
            # Handle game cleanup
            if game_id in self.games:
                game_info = self.games[game_id]
                if client in game_info['players']:
                    game_info['players'].remove(client)
                    
                    # If the other player is still connected, notify them
                    if game_info['players']:
                        try:
                            self.send_message(game_info['players'][0], 
                                             {'status': 'player_disconnected'})
                        except:
                            pass
                    
                    # If no players left, remove the game
                    if not game_info['players']:
                        del self.games[game_id]
            
            try:
                client.close()
            except:
                pass
    
    def send_game_state(self, client, game_id, player_symbol=None):
        """Send current game state to a client"""
        if game_id not in self.games:
            return
        
        game_info = self.games[game_id]
        message = {
            'board': game_info['board'],
            'current_player': game_info['current_player']
        }
        
        if player_symbol:
            message['player_symbol'] = player_symbol
        
        self.send_message(client, message)
    
    def broadcast_game_state(self, game_id, status=None, player=None):
        """Send game state to all players in a game"""
        if game_id not in self.games:
            return
        
        game_info = self.games[game_id]
        
        for client in game_info['players']:
            message = {
                'board': game_info['board'],
                'current_player': game_info['current_player']
            }
            
            if status:
                message['status'] = status
            
            if player:
                message['player'] = player
                
            self.send_message(client, message)
    
    def send_message(self, client, message):
        """Send a message to a client"""
        try:
            client.send(json.dumps(message).encode('utf-8'))
        except Exception as e:
            print(f"Error sending message: {e}")
    
    def check_winner(self, board):
        """Check if there's a winner on the board"""
        # Horizontal rows
        for i in range(0, 9, 3):
            if board[i] != ' ' and board[i] == board[i+1] == board[i+2]:
                return board[i]
        
        # Vertical columns
        for i in range(3):
            if board[i] != ' ' and board[i] == board[i+3] == board[i+6]:
                return board[i]
        
        # Diagonals
        if board[0] != ' ' and board[0] == board[4] == board[8]:
            return board[0]
        if board[2] != ' ' and board[2] == board[4] == board[6]:
            return board[2]
        
        return None


class NetworkClient:
    """Client for connecting to a Tic-Tac-Toe game server"""
    
    def __init__(self, host='localhost', port=5556):
        self.host = host
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False
        self.player_symbol = None
        self.board = [' '] * 9
        self.current_player = None
        self.game_over = False
        self.game_status = "Connecting..."
        self.receive_thread = None
        self.message_buffer = ""
        self.winning_line = None  # Store winning line for highlighting
    
    def connect(self):
        """Connect to the server"""
        try:
            self.client.connect((self.host, self.port))
            self.connected = True
            self.game_status = "Connected to server! Waiting for second player..."
            
            # Start receiving thread
            self.receive_thread = threading.Thread(target=self.receive_updates)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            return True
        except Exception as e:
            self.game_status = f"Cannot connect to server: {e}"
            return False
    
    def disconnect(self):
        """Disconnect from the server"""
        self.connected = False
        try:
            self.client.close()
        except:
            pass
    
    def make_move(self, position):
        """Send a move to the server"""
        if not self.connected or self.game_over:
            return False
        
        try:
            self.client.send(json.dumps({'move': position}).encode('utf-8'))
            return True
        except Exception as e:
            self.game_status = f"Error sending move: {e}"
            self.connected = False
            return False
    
    def process_message(self, data):
        """Process a message from the server"""
        try:
            message = json.loads(data)
            
            if 'player_symbol' in message:
                self.player_symbol = message['player_symbol']
                self.game_status = f"You are playing as: {self.player_symbol}"
            
            if 'board' in message:
                self.board = message['board']
                # Check for winning line
                self.winning_line = self.find_winning_line(self.board)
            
            if 'current_player' in message:
                self.current_player = message['current_player']
                if self.current_player == self.player_symbol:
                    self.game_status = "It's your turn!"
                else:
                    self.game_status = f"Player {self.current_player}'s turn"
            
            if 'status' in message:
                if message['status'] == 'win':
                    winner = message.get('player', '')
                    if winner == self.player_symbol:
                        self.game_status = "Congratulations! You won!"
                    else:
                        self.game_status = f"Player {winner} won!"
                    self.game_over = True
                elif message['status'] == 'tie':
                    self.game_status = "It's a tie!"
                    self.game_over = True
                elif message['status'] == 'start':
                    self.game_status = "Game started!"
                elif message['status'] == 'player_disconnected':
                    self.game_status = "The other player disconnected. Game over."
                    self.game_over = True
        except json.JSONDecodeError:
            print(f"Invalid data received: {data}")
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def find_winning_line(self, board):
        """Find the winning line if there is one"""
        # Horizontal rows
        for i in range(0, 9, 3):
            if board[i] != ' ' and board[i] == board[i+1] == board[i+2]:
                return [(i, i+1, i+2)]
        
        # Vertical columns
        for i in range(3):
            if board[i] != ' ' and board[i] == board[i+3] == board[i+6]:
                return [(i, i+3, i+6)]
        
        # Diagonals
        if board[0] != ' ' and board[0] == board[4] == board[8]:
            return [(0, 4, 8)]
        if board[2] != ' ' and board[2] == board[4] == board[6]:
            return [(2, 4, 6)]
        
        return None
    
    def receive_updates(self):
        """Receive updates from the server"""
        buffer = ""
        while self.connected:
            try:
                data = self.client.recv(1024).decode('utf-8')
                if not data:
                    self.game_status = "Disconnected from server"
                    self.connected = False
                    break
                
                # Add to buffer
                buffer += data
                
                # Process complete messages
                while True:
                    try:
                        # Try to parse as JSON
                        message = json.loads(buffer)
                        # If successful, process it
                        self.process_message(buffer)
                        # Clear buffer
                        buffer = ""
                        break
                    except json.JSONDecodeError:
                        # Check if we have multiple messages
                        pos = buffer.find('}{')
                        if pos != -1:
                            # Process first message
                            first_message = buffer[:pos+1]
                            try:
                                self.process_message(first_message)
                            except:
                                pass
                            # Keep remainder in buffer
                            buffer = buffer[pos+1:]
                        else:
                            # Incomplete message
                            break
                    except Exception as e:
                        print(f"Error processing buffer: {e}")
                        buffer = ""
                        break
                
            except ConnectionResetError:
                self.game_status = "Connection reset by server"
                self.connected = False
                break
            except Exception as e:
                self.game_status = f"Error receiving data: {e}"
                self.connected = False
                break


class TrainingStats:
    """Class to track and display training statistics"""
    
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.win_rates = []
        self.avg_rewards = []
        self.episodes = []
        self.epsilon_values = []
        self.last_check_episode = 0
        self.eval_interval = 100  # Evaluate every 100 episodes
        self.eval_games = 50  # Number of games to play for evaluation
        
        # History file
        self.history_file = "training_history.txt"
        self.load_history()
    
    def load_history(self):
        """Load training history if it exists"""
        try:
            with open(self.history_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        try:
                            episode = int(parts[0])
                            win_rate = float(parts[1])
                            avg_reward = float(parts[2])
                            epsilon = float(parts[3])
                            
                            self.episodes.append(episode)
                            self.win_rates.append(win_rate)
                            self.avg_rewards.append(avg_reward)
                            self.epsilon_values.append(epsilon)
                        except:
                            pass
            print(f"Loaded {len(self.episodes)} data points from history file")
        except FileNotFoundError:
            print("No training history file found")
    
    def save_history(self):
        """Save training history to file"""
        with open(self.history_file, 'w') as f:
            for i in range(len(self.episodes)):
                f.write(f"{self.episodes[i]},{self.win_rates[i]},{self.avg_rewards[i]},{self.epsilon_values[i]}\n")
        print(f"Saved {len(self.episodes)} data points to history file")
    
    def add_evaluation(self, episode, win_rate, avg_reward, epsilon):
        """Add evaluation results"""
        self.episodes.append(episode)
        self.win_rates.append(win_rate)
        self.avg_rewards.append(avg_reward)
        self.epsilon_values.append(epsilon)
        
        # Keep only the most recent max_points
        if len(self.episodes) > self.max_points:
            self.episodes = self.episodes[-self.max_points:]
            self.win_rates = self.win_rates[-self.max_points:]
            self.avg_rewards = self.avg_rewards[-self.max_points:]
            self.epsilon_values = self.epsilon_values[-self.max_points:]
    
    def evaluate_ai(self, ai):
        """Evaluate AI performance by playing games against random player"""
        wins = 0
        ties = 0
        total_reward = 0
        
        for _ in range(self.eval_games):
            game = TicTacToe()
            state = game.reset()
            done = False
            game_reward = 0
            
            while not done:
                if game.current_player == 'O':  # AI plays as O
                    action = ai.get_action(state, training=False)
                else:  # Random player plays as X
                    valid_moves = [i for i, cell in enumerate(state) if cell == ' ']
                    action = random.choice(valid_moves) if valid_moves else -1
                
                state, reward, done = game.step(action)
                if game.current_player == 'X':  # Reward is for the AI's move
                    game_reward += reward
            
            total_reward += game_reward
            
            if game.winner == 'O':  # AI wins
                wins += 1
            elif game.winner is None:  # Tie
                ties += 1
        
        win_rate = wins / self.eval_games
        tie_rate = ties / self.eval_games
        avg_reward = total_reward / self.eval_games
        
        return win_rate, tie_rate, avg_reward
    
    def draw_stats(self, screen, rect):
        """Draw training statistics graphs"""
        if not self.episodes:
            # No data to display
            msg = FONT.render("No training data available", True, BLACK)
            screen.blit(msg, (rect[0] + 10, rect[1] + rect[3]//2))
            return
        
        # Draw background
        pygame.draw.rect(screen, WHITE, rect)
        pygame.draw.rect(screen, BLACK, rect, 2)
        
        # Draw title
        title = FONT.render("Training Progress", True, BLACK)
        screen.blit(title, (rect[0] + rect[2]//2 - title.get_width()//2, rect[1] + 10))
        
        # Draw axes
        margin = 50
        graph_rect = (rect[0] + margin, rect[1] + margin, 
                     rect[2] - 2*margin, rect[3] - 2*margin)
        
        # X-axis
        pygame.draw.line(screen, BLACK, 
                       (graph_rect[0], graph_rect[1] + graph_rect[3]),
                       (graph_rect[0] + graph_rect[2], graph_rect[1] + graph_rect[3]), 2)
        
        # Y-axis
        # Y-axis
        pygame.draw.line(screen, BLACK, 
                       (graph_rect[0], graph_rect[1] + graph_rect[3]),
                       (graph_rect[0], graph_rect[1]), 2)
        
        # Draw win rate graph (green)
        if len(self.win_rates) > 1:
            points = []
            for i, rate in enumerate(self.win_rates):
                x = graph_rect[0] + (i / (len(self.win_rates) - 1)) * graph_rect[2]
                y = graph_rect[1] + graph_rect[3] - rate * graph_rect[3]
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(screen, GREEN, False, points, 2)
        
        # Draw avg reward graph (blue)
        if len(self.avg_rewards) > 1:
            max_reward = max(max(self.avg_rewards), 0.1)  # Normalize
            min_reward = min(min(self.avg_rewards), -0.1)
            reward_range = max_reward - min_reward
            
            points = []
            for i, reward in enumerate(self.avg_rewards):
                x = graph_rect[0] + (i / (len(self.avg_rewards) - 1)) * graph_rect[2]
                normalized_reward = (reward - min_reward) / reward_range
                y = graph_rect[1] + graph_rect[3] - normalized_reward * graph_rect[3] * 0.8
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(screen, BLUE, False, points, 2)
        
        # Draw epsilon values (red)
        if len(self.epsilon_values) > 1:
            points = []
            for i, epsilon in enumerate(self.epsilon_values):
                x = graph_rect[0] + (i / (len(self.epsilon_values) - 1)) * graph_rect[2]
                y = graph_rect[1] + graph_rect[3] - epsilon * graph_rect[3]
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(screen, RED, False, points, 2)
        
        # Draw legend
        legend_y = rect[1] + 40
        legend_x = rect[0] + rect[2] - 120
        
        # Win rate
        pygame.draw.line(screen, GREEN, (legend_x, legend_y), (legend_x + 20, legend_y), 2)
        text = SMALL_FONT.render("Win Rate", True, BLACK)
        screen.blit(text, (legend_x + 25, legend_y - 7))
        
        # Avg reward
        pygame.draw.line(screen, BLUE, (legend_x, legend_y + 20), (legend_x + 20, legend_y + 20), 2)
        text = SMALL_FONT.render("Avg Reward", True, BLACK)
        screen.blit(text, (legend_x + 25, legend_y + 13))
        
        # Epsilon
        pygame.draw.line(screen, RED, (legend_x, legend_y + 40), (legend_x + 20, legend_y + 40), 2)
        text = SMALL_FONT.render("Epsilon", True, BLACK)
        screen.blit(text, (legend_x + 25, legend_y + 33))
        
        # Latest values
        if self.episodes and self.win_rates and self.avg_rewards:
            text = TINY_FONT.render(f"Episodes: {self.episodes[-1]}", True, BLACK)
            screen.blit(text, (rect[0] + 10, rect[1] + rect[3] - 60))
            
            text = TINY_FONT.render(f"Win Rate: {self.win_rates[-1]:.2f}", True, BLACK)
            screen.blit(text, (rect[0] + 10, rect[1] + rect[3] - 40))
            
            text = TINY_FONT.render(f"Avg Reward: {self.avg_rewards[-1]:.2f}", True, BLACK)
            screen.blit(text, (rect[0] + 10, rect[1] + rect[3] - 20))


class GameGUI:
    """Graphical user interface for the Tic-Tac-Toe game"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe with Q-Learning AI")
        
        self.game = TicTacToe()
        self.ai = SimpleQLearningAI()
        self.network_client = None
        self.server = None
        self.training_stats = TrainingStats()
        
        self.game_state = STATE_MENU
        self.ai_thinking = False
        self.training_running = False
        self.training_episodes = 0
        self.training_max_episodes = 10000
        self.training_start_time = None
        
        # UI elements
        self.buttons = {}
        self.create_buttons()
        
        # Training thread
        self.training_thread = None
    
    def create_buttons(self):
        """Create UI buttons"""
        # Menu buttons
        button_width = 200
        button_height = 50
        button_margin = 20
        
        x = SCREEN_WIDTH // 2 - button_width // 2
        y = 150
        
        self.buttons["play_ai"] = pygame.Rect(x, y, button_width, button_height)
        self.buttons["play_network"] = pygame.Rect(x, y + button_height + button_margin, 
                                               button_width, button_height)
        self.buttons["train_ai"] = pygame.Rect(x, y + 2 * (button_height + button_margin), 
                                           button_width, button_height)
        self.buttons["rules"] = pygame.Rect(x, y + 3 * (button_height + button_margin), 
                                          button_width, button_height)
        self.buttons["help"] = pygame.Rect(x, y + 4 * (button_height + button_margin), 
                                         button_width, button_height)
        
        # In-game buttons
        self.buttons["menu"] = pygame.Rect(10, SCREEN_HEIGHT - 70, 100, 40)
        self.buttons["reset"] = pygame.Rect(120, SCREEN_HEIGHT - 70, 100, 40)
        self.buttons["undo"] = pygame.Rect(230, SCREEN_HEIGHT - 70, 100, 40)
        self.buttons["save"] = pygame.Rect(340, SCREEN_HEIGHT - 70, 100, 40)
        
        # AI difficulty buttons
        self.buttons["easy"] = pygame.Rect(450, SCREEN_HEIGHT - 70, 40, 40)
        self.buttons["medium"] = pygame.Rect(500, SCREEN_HEIGHT - 70, 40, 40)
        self.buttons["hard"] = pygame.Rect(550, SCREEN_HEIGHT - 70, 40, 40)
        
        # Network connection button
        self.buttons["connect"] = pygame.Rect(SCREEN_WIDTH // 2 - 100, 300, 200, 50)
        
        # Training control buttons
        self.buttons["start_training"] = pygame.Rect(SCREEN_WIDTH // 2 - 150, 200, 140, 40)
        self.buttons["stop_training"] = pygame.Rect(SCREEN_WIDTH // 2 + 10, 200, 140, 40)
        self.buttons["view_stats"] = pygame.Rect(SCREEN_WIDTH // 2 - 100, 250, 200, 40)
        self.buttons["back_to_menu"] = pygame.Rect(SCREEN_WIDTH // 2 - 100, 600, 200, 40)
    
    def handle_menu_click(self, pos):
        """Handle clicks in the menu state"""
        if self.buttons["play_ai"].collidepoint(pos):
            self.reset_game()
            self.game_state = STATE_PLAY_AI
        elif self.buttons["play_network"].collidepoint(pos):
            self.game_state = STATE_WAITING
            # Initialize network client
            if self.network_client is None:
                self.network_client = NetworkClient()
            if not self.network_client.connected:
                # Connect in a separate thread to not block the UI
                threading.Thread(target=self.connect_to_server).start()
        elif self.buttons["train_ai"].collidepoint(pos):
            self.game_state = STATE_TRAINING
        elif self.buttons["rules"].collidepoint(pos):
            self.game_state = STATE_RULES
        elif self.buttons["help"].collidepoint(pos):
            self.game_state = STATE_HELP
    
    def connect_to_server(self):
        """Connect to game server"""
        if self.network_client:
            # Try to connect to the server
            if not self.network_client.connect():
                # If server is not running, try to start one
                if self.server is None:
                    self.server = GameServer()
                    if self.server.start():
                        time.sleep(0.5)  # Wait for server to start
                        self.network_client.connect()
    
    def handle_play_ai_click(self, pos):
        """Handle clicks in the play AI state"""
        # Check if a cell was clicked
        if not self.game.game_over and self.game.current_player == 'X':
            for i in range(9):
                row, col = i // 3, i % 3
                cell_rect = pygame.Rect(
                    (SCREEN_WIDTH - BOARD_SIZE) // 2 + col * CELL_SIZE,
                    100 + row * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                )
                
                if cell_rect.collidepoint(pos) and self.game.board[i] == ' ':
                    # Make player move
                    self.game.step(i)
                    
                    # Schedule AI move
                    if not self.game.game_over:
                        self.ai_thinking = True
                        threading.Thread(target=self.make_ai_move).start()
                    break
        
        # Check if control buttons were clicked
        if self.buttons["menu"].collidepoint(pos):
            self.game_state = STATE_MENU
        elif self.buttons["reset"].collidepoint(pos):
            self.reset_game()
        elif self.buttons["undo"].collidepoint(pos):
            if self.game.undo_move():  # Undo player move
                if not self.game.game_over and not self.game.current_player == 'X':
                    self.game.undo_move()  # Undo AI move too
        elif self.buttons["save"].collidepoint(pos):
            self.ai.save_model()
        
        # Check difficulty buttons
        if self.buttons["easy"].collidepoint(pos):
            self.ai.set_difficulty(AI_DIFFICULTY_EASY)
        elif self.buttons["medium"].collidepoint(pos):
            self.ai.set_difficulty(AI_DIFFICULTY_MEDIUM)
        elif self.buttons["hard"].collidepoint(pos):
            self.ai.set_difficulty(AI_DIFFICULTY_HARD)
    
    def handle_network_play_click(self, pos):
        """Handle clicks in network play state"""
        # Check if a cell was clicked
        if (self.network_client and self.network_client.connected and 
            not self.network_client.game_over and 
            self.network_client.current_player == self.network_client.player_symbol):
            
            for i in range(9):
                row, col = i // 3, i % 3
                cell_rect = pygame.Rect(
                    (SCREEN_WIDTH - BOARD_SIZE) // 2 + col * CELL_SIZE,
                    100 + row * CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                )
                
                if cell_rect.collidepoint(pos) and self.network_client.board[i] == ' ':
                    # Make player move (1-based indexing for network protocol)
                    self.network_client.make_move(i + 1)
                    break
        
        # Check if control buttons were clicked
        if self.buttons["menu"].collidepoint(pos):
            if self.network_client:
                self.network_client.disconnect()
            self.game_state = STATE_MENU
    
    def handle_waiting_click(self, pos):
        """Handle clicks in waiting state"""
        if self.buttons["menu"].collidepoint(pos):
            if self.network_client:
                self.network_client.disconnect()
            self.game_state = STATE_MENU
        elif self.buttons["connect"].collidepoint(pos):
            # Try to connect again
            if self.network_client:
                self.network_client.disconnect()
                threading.Thread(target=self.connect_to_server).start()
    
    def handle_training_click(self, pos):
        """Handle clicks in training state"""
        if self.buttons["start_training"].collidepoint(pos) and not self.training_running:
            self.start_training()
        elif self.buttons["stop_training"].collidepoint(pos) and self.training_running:
            self.stop_training()
        elif self.buttons["view_stats"].collidepoint(pos):
            self.game_state = STATE_TRAINING_STATS
        elif self.buttons["back_to_menu"].collidepoint(pos):
            if self.training_running:
                self.stop_training()
            self.game_state = STATE_MENU
    
    def handle_stats_click(self, pos):
        """Handle clicks in training stats state"""
        if self.buttons["back_to_menu"].collidepoint(pos):
            self.game_state = STATE_TRAINING
    
    def handle_rules_help_click(self, pos):
        """Handle clicks in rules or help state"""
        if self.buttons["back_to_menu"].collidepoint(pos):
            self.game_state = STATE_MENU
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop_training()
                if self.network_client:
                    self.network_client.disconnect()
                if self.server:
                    self.server.stop()
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                
                if self.game_state == STATE_MENU:
                    self.handle_menu_click(pos)
                elif self.game_state == STATE_PLAY_AI:
                    self.handle_play_ai_click(pos)
                elif self.game_state == STATE_PLAY_NETWORK:
                    self.handle_network_play_click(pos)
                elif self.game_state == STATE_WAITING:
                    self.handle_waiting_click(pos)
                elif self.game_state == STATE_TRAINING:
                    self.handle_training_click(pos)
                elif self.game_state == STATE_TRAINING_STATS:
                    self.handle_stats_click(pos)
                elif self.game_state in [STATE_RULES, STATE_HELP]:
                    self.handle_rules_help_click(pos)
        
        # Check if we need to transition from waiting to play state
        if (self.game_state == STATE_WAITING and self.network_client and 
            self.network_client.connected and self.network_client.player_symbol):
            self.game_state = STATE_PLAY_NETWORK
        
        return True
    
    def make_ai_move(self):
        """Make AI move in a separate thread"""
        time.sleep(0.5)  # Add a small delay for better UX
        
        if not self.game.game_over and self.game.current_player == 'O':
            ai_action = self.ai.get_action(self.game.board, training=False)
            if ai_action >= 0:
                self.game.step(ai_action)
        
        self.ai_thinking = False
    
    def reset_game(self):
        """Reset the game"""
        self.game.reset()
    
    def start_training(self):
        """Start AI training in a separate thread"""
        if not self.training_running:
            self.training_running = True
            self.training_start_time = time.time()
            self.training_episodes = 0
            
            # Start training in a separate thread
            self.training_thread = threading.Thread(target=self.train_ai)
            self.training_thread.daemon = True
            self.training_thread.start()
    
    def stop_training(self):
        """Stop AI training"""
        self.training_running = False
        if self.training_thread:
            # Wait for training thread to stop (max 1 sec)
            self.training_thread.join(1.0)
            self.training_thread = None
        
        # Save the model and training stats
        self.ai.save_model()
        self.training_stats.save_history()
    
    def train_ai(self):
        """Train the AI model"""
        training_game = TicTacToe()
        episode = 0
        
        # Continue from last saved episode count
        if self.training_stats.episodes:
            episode = self.training_stats.episodes[-1]
        
        while self.training_running and episode < self.training_max_episodes:
            episode += 1
            self.training_episodes = episode
            
            # Reset the game
            state = training_game.reset()
            done = False
            
            # Play one game
            while not done and self.training_running:
                # Choose action
                action = self.ai.get_action(state, training=True)
                
                # Take action
                next_state, reward, done = training_game.step(action)
                
                # Update Q-value
                self.ai.update_q_value(state, action, reward, next_state, done)
                
                # Update state
                state = next_state.copy()
            
            # Evaluate and log progress periodically
            if episode % self.training_stats.eval_interval == 0 and self.training_running:
                win_rate, tie_rate, avg_reward = self.training_stats.evaluate_ai(self.ai)
                self.training_stats.add_evaluation(episode, win_rate, avg_reward, self.ai.epsilon)
                print(f"Episode {episode}: Win Rate = {win_rate:.2f}, Tie Rate = {tie_rate:.2f}, "
                     f"Avg Reward = {avg_reward:.2f}, Epsilon = {self.ai.epsilon:.4f}")
        
        # Save the model when training completes or is stopped
        if self.training_running:
            self.ai.save_model()
            self.training_stats.save_history()
        
        self.training_running = False
    
    def draw_board(self, board, current_player, game_over=False, winner=None, winning_line=None):
        """Draw the game board"""
        # Draw board background
        board_rect = pygame.Rect(
            (SCREEN_WIDTH - BOARD_SIZE) // 2, 
            100, 
            BOARD_SIZE, 
            BOARD_SIZE
        )
        pygame.draw.rect(self.screen, WHITE, board_rect)
        pygame.draw.rect(self.screen, BLACK, board_rect, 3)
        
        # Draw grid lines
        for i in range(1, 3):
            # Vertical lines
            pygame.draw.line(
                self.screen, 
                BLACK,
                (board_rect.left + i * CELL_SIZE, board_rect.top),
                (board_rect.left + i * CELL_SIZE, board_rect.bottom),
                3
            )
            
            # Horizontal lines
            pygame.draw.line(
                self.screen, 
                BLACK,
                (board_rect.left, board_rect.top + i * CELL_SIZE),
                (board_rect.right, board_rect.top + i * CELL_SIZE),
                3
            )
        
        # Draw X's and O's
        for i in range(9):
            row, col = i // 3, i % 3
            x = board_rect.left + col * CELL_SIZE + CELL_SIZE // 2
            y = board_rect.top + row * CELL_SIZE + CELL_SIZE // 2
            
            if board[i] == 'X':
                # Draw X
                offset = CELL_SIZE // 3
                pygame.draw.line(
                    self.screen, 
                    RED,
                    (x - offset, y - offset),
                    (x + offset, y + offset),
                    6
                )
                pygame.draw.line(
                    self.screen, 
                    RED,
                    (x - offset, y + offset),
                    (x + offset, y - offset),
                    6
                )
            elif board[i] == 'O':
                # Draw O
                pygame.draw.circle(
                    self.screen,
                    BLUE,
                    (x, y),
                    CELL_SIZE // 3,
                    6
                )
        
        # Highlight winning line
        if winning_line:
            for line in winning_line:
                # Get center points of cells in the winning line
                points = []
                for cell in line:
                    row, col = cell // 3, cell % 3
                    x = board_rect.left + col * CELL_SIZE + CELL_SIZE // 2
                    y = board_rect.top + row * CELL_SIZE + CELL_SIZE // 2
                    points.append((x, y))
                
                # Draw line through winning cells
                pygame.draw.line(
                    self.screen,
                    GREEN,
                    points[0],
                    points[-1],
                    10
                )
        
        # Draw current player indicator
        player_text = FONT.render(f"Current Player: {'X' if current_player == 'X' else 'O'}", True, BLACK)
        self.screen.blit(player_text, (board_rect.left, board_rect.top - 40))
        
        # Draw game status
        if game_over:
            if winner:
                status_text = LARGE_FONT.render(f"Player {winner} wins!", True, GREEN)
            else:
                status_text = LARGE_FONT.render("It's a tie!", True, BLUE)
            
            self.screen.blit(status_text, 
                           (SCREEN_WIDTH // 2 - status_text.get_width() // 2, 50))
    
    def draw_control_buttons(self):
        """Draw game control buttons"""
        # Draw menu button
        pygame.draw.rect(self.screen, LIGHT_BLUE, self.buttons["menu"])
        pygame.draw.rect(self.screen, BLACK, self.buttons["menu"], 2)
        text = SMALL_FONT.render("Menu", True, BLACK)
        self.screen.blit(text, (self.buttons["menu"].x + self.buttons["menu"].width // 2 - text.get_width() // 2,
                              self.buttons["menu"].y + self.buttons["menu"].height // 2 - text.get_height() // 2))
        
        # Draw reset button
        pygame.draw.rect(self.screen, LIGHT_GREEN, self.buttons["reset"])
        pygame.draw.rect(self.screen, BLACK, self.buttons["reset"], 2)
        text = SMALL_FONT.render("Reset", True, BLACK)
        self.screen.blit(text, (self.buttons["reset"].x + self.buttons["reset"].width // 2 - text.get_width() // 2,
                              self.buttons["reset"].y + self.buttons["reset"].height // 2 - text.get_height() // 2))
        
        # Draw undo button
        pygame.draw.rect(self.screen, LIGHT_GRAY, self.buttons["undo"])
        pygame.draw.rect(self.screen, BLACK, self.buttons["undo"], 2)
        text = SMALL_FONT.render("Undo", True, BLACK)
        self.screen.blit(text, (self.buttons["undo"].x + self.buttons["undo"].width // 2 - text.get_width() // 2,
                              self.buttons["undo"].y + self.buttons["undo"].height // 2 - text.get_height() // 2))
        
        # Draw save button
        pygame.draw.rect(self.screen, PINK, self.buttons["save"])
        pygame.draw.rect(self.screen, BLACK, self.buttons["save"], 2)
        text = SMALL_FONT.render("Save AI", True, BLACK)
        self.screen.blit(text, (self.buttons["save"].x + self.buttons["save"].width // 2 - text.get_width() // 2,
                              self.buttons["save"].y + self.buttons["save"].height // 2 - text.get_height() // 2))
    
    def draw_difficulty_buttons(self):
        """Draw AI difficulty buttons"""
        difficulties = [
            ("E", self.buttons["easy"], AI_DIFFICULTY_EASY),
            ("M", self.buttons["medium"], AI_DIFFICULTY_MEDIUM),
            ("H", self.buttons["hard"], AI_DIFFICULTY_HARD)
        ]
        
        # Draw difficulty label
        text = SMALL_FONT.render("AI Level:", True, BLACK)
        self.screen.blit(text, (self.buttons["easy"].x - 70, self.buttons["easy"].y + 10))
        
        # Draw difficulty buttons
        for label, button, level in difficulties:
            color = YELLOW if abs(self.ai.difficulty - level) < 0.01 else LIGHT_GRAY
            pygame.draw.rect(self.screen, color, button)
            pygame.draw.rect(self.screen, BLACK, button, 2)
            text = SMALL_FONT.render(label, True, BLACK)
            self.screen.blit(text, (button.x + button.width // 2 - text.get_width() // 2,
                                  button.y + button.height // 2 - text.get_height() // 2))
    
    def draw_connect_button(self):
        """Draw connect button for network play"""
        pygame.draw.rect(self.screen, LIGHT_GREEN, self.buttons["connect"])
        pygame.draw.rect(self.screen, BLACK, self.buttons["connect"], 2)
        text = FONT.render("Connect to Server", True, BLACK)
        self.screen.blit(text, (self.buttons["connect"].x + self.buttons["connect"].width // 2 - text.get_width() // 2,
                              self.buttons["connect"].y + self.buttons["connect"].height // 2 - text.get_height() // 2))
    
    def draw_menu(self):
        """Draw main menu"""
        # Draw title
        title = LARGE_FONT.render("Tic-Tac-Toe with Q-Learning", True, BLACK)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 50))
        
        # Draw menu buttons
        button_labels = [
            ("Play vs. AI", self.buttons["play_ai"], LIGHT_BLUE),
            ("Play Online", self.buttons["play_network"], LIGHT_GREEN),
            ("Train AI", self.buttons["train_ai"], ORANGE),
            ("Game Rules", self.buttons["rules"], YELLOW),
            ("Help / Guide", self.buttons["help"], PINK)
        ]
        
        for label, button, color in button_labels:
            pygame.draw.rect(self.screen, color, button)
            pygame.draw.rect(self.screen, BLACK, button, 2)
            text = FONT.render(label, True, BLACK)
            self.screen.blit(text, (button.x + button.width // 2 - text.get_width() // 2,
                                  button.y + button.height // 2 - text.get_height() // 2))
    
    def draw_training_screen(self):
        """Draw AI training screen"""
        # Draw title
        title = LARGE_FONT.render("Train Q-Learning AI", True, BLACK)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 50))
        
        # Draw explanation
        lines = [
            "Training lets the AI learn by playing against itself.",
            "The AI uses Q-learning to improve its strategy over time.",
            "The longer you train, the better the AI becomes.",
            f"Current epsilon (randomness): {self.ai.epsilon:.4f}"
        ]
        
        for i, line in enumerate(lines):
            text = SMALL_FONT.render(line, True, BLACK)
            self.screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, 100 + i * 25))
        
        # Draw training status
        if self.training_running:
            status_text = f"Training in progress... Episodes: {self.training_episodes}/{self.training_max_episodes}"
            elapsed = time.time() - self.training_start_time
            time_text = f"Time elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s"
            
            text = FONT.render(status_text, True, GREEN)
            self.screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, 300))
            
            text = FONT.render(time_text, True, BLACK)
            self.screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, 330))
            
            # Draw progress bar
            progress_rect = pygame.Rect(100, 370, SCREEN_WIDTH - 200, 30)
            pygame.draw.rect(self.screen, WHITE, progress_rect)
            pygame.draw.rect(self.screen, BLACK, progress_rect, 2)
            
            if self.training_max_episodes > 0:
                progress_width = (self.training_episodes / self.training_max_episodes) * (progress_rect.width - 4)
                progress_fill = pygame.Rect(progress_rect.x + 2, progress_rect.y + 2, 
                                         progress_width, progress_rect.height - 4)
                pygame.draw.rect(self.screen, GREEN, progress_fill)
        else:
            text = FONT.render("Click 'Start Training' to begin", True, BLACK)
            self.screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, 300))
        
        # Draw buttons
        buttons = [
            ("Start Training", self.buttons["start_training"], LIGHT_GREEN if not self.training_running else GRAY),
            ("Stop Training", self.buttons["stop_training"], RED if self.training_running else GRAY),
            ("View Statistics", self.buttons["view_stats"], LIGHT_BLUE),
            ("Return to Menu", self.buttons["back_to_menu"], YELLOW ),
            ]        
        
        for label, button, color in buttons:
            pygame.draw.rect(self.screen, color, button)
            pygame.draw.rect(self.screen, BLACK, button, 2)
            text = FONT.render(label, True, BLACK)
            self.screen.blit(text, (button.x + button.width // 2 - text.get_width() // 2,
                                  button.y + button.height // 2 - text.get_height() // 2))
    
    def draw_waiting_screen(self):
        """Draw waiting screen for network play"""
        # Draw title
        title = LARGE_FONT.render("Online Play", True, BLACK)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 50))
        
        # Draw status
        if self.network_client:
            if self.network_client.connected:
                if self.network_client.player_symbol:
                    status = f"Connected as Player {self.network_client.player_symbol}. Waiting for opponent..."
                else:
                    status = "Connected to server. Waiting for game assignment..."
            else:
                status = self.network_client.game_status
        else:
            status = "Initializing network client..."
        
        text = FONT.render(status, True, BLACK)
        self.screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, 200))
        
        # Draw connect button if not connected
        if not (self.network_client and self.network_client.connected):
            self.draw_connect_button()
        
        # Draw menu button
        pygame.draw.rect(self.screen, LIGHT_BLUE, self.buttons["menu"])
        pygame.draw.rect(self.screen, BLACK, self.buttons["menu"], 2)
        text = FONT.render("Back to Menu", True, BLACK)
        self.screen.blit(text, (self.buttons["menu"].x + self.buttons["menu"].width // 2 - text.get_width() // 2,
                              self.buttons["menu"].y + self.buttons["menu"].height // 2 - text.get_height() // 2))
    
    def draw_rules_screen(self):
        """Draw game rules screen"""
        # Draw title
        title = LARGE_FONT.render("Game Rules", True, BLACK)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 50))
        
        # Draw rules text
        rules = [
            "Tic-Tac-Toe is a classic two-player game played on a 3x3 grid.",
            "",
            "Rules:",
            "1. Players alternate placing X and O marks on empty squares.",
            "2. The first player to get 3 of their marks in a row wins.",
            "3. The row can be horizontal, vertical, or diagonal.",
            "4. If all squares are filled without a winner, the game is a tie.",
            "",
            "Controls:",
            "- Click on an empty square to place your mark.",
            "- Use the Reset button to start a new game.",
            "- Use the Undo button to take back the last move.",
            "- Use the Menu button to return to the main menu."
        ]
        
        for i, line in enumerate(rules):
            text = SMALL_FONT.render(line, True, BLACK)
            self.screen.blit(text, (50, 120 + i * 25))
        
        # Draw back button
        pygame.draw.rect(self.screen, LIGHT_BLUE, self.buttons["back_to_menu"])
        pygame.draw.rect(self.screen, BLACK, self.buttons["back_to_menu"], 2)
        text = FONT.render("Back to Menu", True, BLACK)
        self.screen.blit(text, (self.buttons["back_to_menu"].x + self.buttons["back_to_menu"].width // 2 - text.get_width() // 2,
                              self.buttons["back_to_menu"].y + self.buttons["back_to_menu"].height // 2 - text.get_height() // 2))
    
    def draw_help_screen(self):
        """Draw help/guide screen"""
        # Draw title
        title = LARGE_FONT.render("Help & Guide", True, BLACK)
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 50))
        
        # Draw help text
        help_text = [
            "Game Modes:",
            "1. Play vs AI: Play against the computer AI.",
            "2. Play Online: Play against another player over the network.",
            "3. Train AI: Let the AI learn by playing against itself.",
            "",
            "AI Difficulty:",
            "- Easy: AI makes more random moves.",
            "- Medium: Balanced AI behavior.",
            "- Hard: AI makes optimal moves most of the time.",
            "",
            "Training:",
            "- The AI uses Q-learning to improve its strategy.",
            "- Training saves automatically when completed.",
            "- View Statistics shows the AI's learning progress.",
            "",
            "Network Play:",
            "- If no server is running, one will be created automatically.",
            "- First player to connect becomes X, second becomes O."
        ]
        
        for i, line in enumerate(help_text):
            text = SMALL_FONT.render(line, True, BLACK)
            self.screen.blit(text, (50, 120 + i * 20))
        
        # Draw back button
        pygame.draw.rect(self.screen, LIGHT_BLUE, self.buttons["back_to_menu"])
        pygame.draw.rect(self.screen, BLACK, self.buttons["back_to_menu"], 2)
        text = FONT.render("Back to Menu", True, BLACK)
        self.screen.blit(text, (self.buttons["back_to_menu"].x + self.buttons["back_to_menu"].width // 2 - text.get_width() // 2,
                              self.buttons["back_to_menu"].y + self.buttons["back_to_menu"].height // 2 - text.get_height() // 2))
    
    def draw_network_status(self):
        """Draw network status information"""
        if self.network_client:
            status_text = FONT.render(self.network_client.game_status, True, BLACK)
            self.screen.blit(status_text, (10, SCREEN_HEIGHT - 30))
    
    def draw(self):
        """Draw the current game state"""
        # Fill background
        self.screen.fill(WHITE)
        
        # Draw based on current game state
        if self.game_state == STATE_MENU:
            self.draw_menu()
        elif self.game_state == STATE_PLAY_AI:
            self.draw_board(self.game.board, self.game.current_player, 
                           self.game.game_over, self.game.winner, self.game.winning_line)
            self.draw_control_buttons()
            self.draw_difficulty_buttons()
            
            if self.ai_thinking:
                text = FONT.render("AI is thinking...", True, BLACK)
                self.screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, 70))
        elif self.game_state == STATE_PLAY_NETWORK:
            if self.network_client:
                self.draw_board(self.network_client.board, self.network_client.current_player,
                               self.network_client.game_over, None, self.network_client.winning_line)
                self.draw_network_status()
            
            # Only show menu button in network play
            pygame.draw.rect(self.screen, LIGHT_BLUE, self.buttons["menu"])
            pygame.draw.rect(self.screen, BLACK, self.buttons["menu"], 2)
            text = FONT.render("Menu", True, BLACK)
            self.screen.blit(text, (self.buttons["menu"].x + self.buttons["menu"].width // 2 - text.get_width() // 2,
                                  self.buttons["menu"].y + self.buttons["menu"].height // 2 - text.get_height() // 2))
        elif self.game_state == STATE_WAITING:
            self.draw_waiting_screen()
        elif self.game_state == STATE_TRAINING:
            self.draw_training_screen()
        elif self.game_state == STATE_TRAINING_STATS:
            # Draw stats screen
            title = LARGE_FONT.render("Training Statistics", True, BLACK)
            self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 50))
            
            # Draw stats graph
            stats_rect = (50, 100, SCREEN_WIDTH - 100, 450)
            self.training_stats.draw_stats(self.screen, stats_rect)
            
            # Draw back button
            pygame.draw.rect(self.screen, LIGHT_BLUE, self.buttons["back_to_menu"])
            pygame.draw.rect(self.screen, BLACK, self.buttons["back_to_menu"], 2)
            text = FONT.render("Back to Training", True, BLACK)
            self.screen.blit(text, (self.buttons["back_to_menu"].x + self.buttons["back_to_menu"].width // 2 - text.get_width() // 2,
                                  self.buttons["back_to_menu"].y + self.buttons["back_to_menu"].height // 2 - text.get_height() // 2))
        elif self.game_state == STATE_RULES:
            self.draw_rules_screen()
        elif self.game_state == STATE_HELP:
            self.draw_help_screen()
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            running = self.handle_events()
            self.draw()
            clock.tick(60)  # Cap at 60 FPS
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = GameGUI()
    game.run()
         