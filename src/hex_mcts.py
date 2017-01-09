from hex import HexState

class HexMCTS(HexState):
    def display(self, state, action):
        board = str(state)
        board += "\nPlayed: " + action
        board += "\nPlayer " + state.nextPlayer + " to move."
        return board

    def pack_state(self, state):
        return state

    def pack_action(self, action):
        from ast import literal_eval as make_tuple
        return make_tuple(action)

    def unpack_action(self, action):
        return str(action)

    def legal_actions(self, history):
        return history[-1].getLegalActions()

    def next_state(self, state, action):
        return state.getNextState(action, state.nextPlayer)
    
    def current_player(self, state):
        return state.nextPlayer
        
    def is_ended(self, history):
        return history[-1].isGoalState()

    def win_values(self, history):
        winner = history[-1].getWinner()
        if winner == 3:
            return {1: 0.5, 2: 0.5}
        if not winner:
            return

        return {winner: 1, 3 - winner: 0}

    points_values = win_values
    
    def winner_message(self, winners):
        winners = sorted((v, k) for k, v in winners.iteritems())
        value, winner = winners[-1]
        if value == 0.5:
            return "Stalemate."
        return "Winner: Player {0}.".format(winner)


