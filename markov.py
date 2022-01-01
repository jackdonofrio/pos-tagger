# tag latin parts of speech using hidden markov model
# adapted from java code written for dartmouth cs10 problem set 5

import math
START_STATE = "#"
UNOBSERVED_VALUE = -10

class ModelState:
    """
    Auxiliary class used to help store textual data in 
    hidden markov model dictionary representation
    """
    def __init__(self):
        self.observations = {}   # observations belonging to this map value
        self.out = {}            # states this state can transition to
        self.observation_sum = 0 # total number of observations associated with this state
        self.transition_sum = 0  # sum of outward transition frequencies


    def add_out_transition(self, destination: str):
        """
        Adds destination to map of outward transitions,
        increments frequency of transitions from this state to destination,
        increments total number of transitions
        
        params:
            destination: str representation of transition state
        """

        if destination not in self.out:
            self.out[destination] = 0
        self.out[destination] += 1
        self.transition_sum += 1
    

    def add_observation(self, observation: str):
        """
        adds observation to map of observations for this state,
        increments frequency of said observation within this state,
        increments total count of observations for this state

        params:
            observation: textual observation
        """

        if observation not in self.observations:
            self.observations[observation] = 0
        self.observations[observation] += 1
        self.observation_sum += 1
    

    def normalize_observations(self):
        """
        Convert text observation frequency counts to relative frequencies,
        then take natural log of relative frequencies to be able to multiply
        values without dealing with floating point precision issues, noting
        that log(ab) = log(a) + log(b)
        """

        if self.observation_sum == 0:
            return
        
        for observation, frequency in self.observations.items():
            self.observations[observation] = math.log(frequency / self.observation_sum)

    
    def normalize_transitions(self):
        """
        Perform same operation as in normalize_observations on transitions
        """

        if self.transition_sum == 0:
            return
        
        for destination, frequency in self.out.items():
            self.out[destination] = math.log(frequency / self.transition_sum)
    
    def __str__(self):
        """ String representation of state for debugging use """
        
        return f"observations: {self.observations} | transitions: {self.out}"


def read_file_into_list(filepath, set_lowercase=False):
    lines = []
    with open(filepath, 'r') as file:
        for line in file:
            if set_lowercase:
                line = line.lower()
            lines.append(line.strip().split(' '))
    return lines

def generate_model_from_files(corpus_path, tagfile_path):
    text = read_file_into_list(corpus_path)
    tags = read_file_into_list(tagfile_path)
    return train_model(text, tags)

def normalize_model(model):
    for state in model:
        model[state].normalize_observations()
        model[state].normalize_transitions()

def train_model(text, tags):
    model = {} # Hidden Markov model represented as a dictionary
    model[START_STATE] = ModelState()
    
    if len(text) != len(tags):
        raise Exception("Training error: text/tag files must have an equal number of lines")
    for index, text_line in enumerate(text):
        tag_line = tags[index]
        if len(text_line) != len(tag_line):
            raise Exception(f"Training error, from line {index + 1} in data: text line length does not equal number of tags.")
        for subindex, word in enumerate(text_line):
            tag = tag_line[subindex]
            if tag not in model:
                model[tag] = ModelState()
            model[tag].add_observation(word)
            if subindex == 0:
                model[START_STATE].add_out_transition(tag)
            if subindex < len(text_line) - 1:
                model[tag].add_out_transition(tag_line[subindex + 1])
    normalize_model(model)
    return model

def viterbi_decode(model, observations):
    """
    propagates through hidden Markov model based on observations,
    stores data in backtrack map, then uses best end state to backtrace
    route of tags to match most likely tags to textual observations

    params:
        model: dictionary representing trained Hidden Markov model
        observations: list of string observations to be tagged by traversing HMM
    """

    backtrack = [{}] * len(observations)
    current_states = set()
    current_states.add(START_STATE)
    current_scores = {START_STATE : 0}
    
    for index, observation in enumerate(observations):
        next_states = set()
        next_scores = {}
        for current_state in current_states:
            for next_state in model[current_state].out:
                next_states.add(next_state)
                next_observations = model[next_state].observations
                observation_score = next_observations[observation] if observation in next_observations else UNOBSERVED_VALUE
                next_score = current_scores[current_state] + model[current_state].out[next_state] + observation_score
                if next_state not in next_scores or next_score > next_scores[next_state]:
                    next_scores[next_state] = next_score
                    if not backtrack[index]:
                        backtrack[index] = {}
                    backtrack[index][next_state] = current_state
        current_states = next_states
        current_scores = next_scores
    max_score = -10000
    max_state = ''
    for state, score in current_scores.items():
        if score > max_score:
            max_score = score
            max_state = state
    tags = [''] * len(observations)
    current = max_state
    tags[len(tags) - 1] = max_state
    for index in range(len(observations) - 1, 0, -1):
        current = backtrack[index][current]
        tags[index - 1] = current
    return tags

def prettify(model, observations):
    """
    displays observations with predicted tags applied 
    ex: dog/NOUN eats/VERB food/NOUN
    """
    tags = viterbi_decode(model, observations)
    return ' '.join([observation + '/' + tags[i] for i, observation in enumerate(observations)])


text = read_file_into_list('train_corpus.txt', set_lowercase=True)
tags = read_file_into_list('train_tags.txt')
model = train_model(text, tags)
obs = 'the dog saw trains in the night .'.split()
print(prettify(model, obs))