from hex import *

class HexMCTS(HexEnv):
    def display(self, state, action):
        pass

    def pack_state(self, state):
        pass

    def pack_action(self, action):
        pass

    def unpack_action(self, action):
        pass

    def legal_actions(self, history):
        pass

    def next_state(self, state, action):
        pass
    
    def current_player(self, state):
        pass
        
    def is_ended(self, history):
        pass

    def win_values(self, history):
        pass

    points_values = win_values
    
    def winner_message(self, winners):
        pass


