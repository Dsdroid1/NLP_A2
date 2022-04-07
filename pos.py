"""
Assignment 2 - NLP, Viterbi Algo for POS tagging by BT18CSE046
"""

# Import required libraries
import nltk
import random
import re
import sys

# # Download the required corpus of words and tags
# nltk.download('treebank')
# nltk.download('universal_tagset')



# Define the transition and emission probabilities
def emission_probabilty(tagged_words, tags, vocab):
    # Conditional probabilty that given a tag, the word is emitted
    emit_matrix = {}
    tag_count = {}
    # Precompute all the emission probabilites

    for word in vocab:
        emit_matrix[word] = {}
        for tag in tags:
            emit_matrix[word][tag] = 0
    
    for tag in tags:
        tag_count[tag] = 0

    for word, tag in tagged_words:
        emit_matrix[word][tag] += 1
        tag_count[tag] += 1

    for word in vocab:
        for tag in tags:
            emit_matrix[word][tag] /= tag_count[tag] # Get the emission probabilites
    
    return emit_matrix

def transition_probability(nltk_data, tags):
    
    transit_matrix = {}
    for row in tags:
        transit_matrix[row] = {}
        for col in tags:
            transit_matrix[row][col] = 0
    
    # We need to extra tags, start and end of sentence
    transit_matrix["<S>"] = {}
    for tag in tags:
        transit_matrix["<S>"][tag] = 0
        transit_matrix[tag]["</S>"] = 0

    

    for sentence in nltk_data:
        # All sentences start with <S>
        prev_tag = "<S>"
        for word, tag in sentence:
            transit_matrix[prev_tag][tag] += 1
            prev_tag = tag
        transit_matrix[prev_tag]["</S>"] += 1

    # Get the probabilities
    for row in tags:
        for col in tags:
            if sum(transit_matrix[row].values()) == 0:
                transit_matrix[row][col] = 0
            else:
                transit_matrix[row][col] /= sum(transit_matrix[row].values())

    return transit_matrix

def viterbi(emit_matrix, transit_matrix, sentence, tags):
    # Assume the sentence is sanitsed and split into token/words
    # words = re.findall(r"[\w']+|[.,!?;]", sentence)
    words = sentence.split(" ")
           
    # Initialize the viterbi matrix
    dp_mat = []
    trace_mat = []
    for i,word in enumerate(words):
        dp_mat.append({})
        trace_mat.append({})

    # For the first word, initial state is fixed at <S>
    for tag in tags:
        transition_probability = transit_matrix["<S>"][tag]
        emission_probability = emit_matrix[words[0]][tag]
        dp_mat[0][tag] = 1 * transition_probability * emission_probability # All sentence start from <S>
        trace_mat[0][tag] = "<S>"

    # For all other word positions and a tag at that pos, calculate the edge which has the highest probability of reaching there from the start state
    for i in range(1, len(words)):
        for tag in tags:
            max_state_probability = 0
            max_tag = ""
            for prev_possible_tag in tags:
                transition_probability = transit_matrix[prev_possible_tag][tag]
                emission_probability = emit_matrix[words[i]][tag]
                state_prob = dp_mat[i-1][prev_possible_tag] * transition_probability * emission_probability
                if state_prob > max_state_probability:
                    max_state_probability = state_prob
                    max_tag = prev_possible_tag
            dp_mat[i][tag] = max_state_probability
            trace_mat[i][tag] = max_tag

    # Adjust for </S> tag at end of sentence, last tag should always be </S>
    max_state_probability = 0
    max_tag = ""
    for tag in tags:
        state_prob = dp_mat[len(words)-1][tag] * transit_matrix[tag]["</S>"]
        if state_prob > max_state_probability:
            max_state_probability = state_prob
            max_tag = tag
    
    # Backtrack to get the tags
    # Each state in the dp matrix has only 1 incoming edge due to maximality property
    # trace back the solution
    solution = []
    solution.append(max_tag)

    for i in range(len(words)-1, 0, -1):
        solution.append(trace_mat[i][solution[-1]])

    solution = solution[::-1] # Reverse to get correct order
    # print(solution)

    return solution


def main():
    # Get the tagged sentences from treebank dataset
    nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset="universal"))
    # # Get a sample sentence
    # sentence  = nltk_data[random.randint(0,len(nltk_data)-1)]
    # words = [word for (word,tag) in sentence]
    # tags = [tag for (word,tag) in sentence]
    # print("The sentence is:"," ".join(words))
    # print("The tags are:"," ".join(tags))

    # Get the tagged words from the corpus
    tagged_words = [(word, tag) for sentence in nltk_data for (word,tag) in sentence]
    # Get all the unique tags from the corpus
    tags = {tag for (word,tag) in tagged_words} # Using a set
    # Get the corpus vocabulary
    vocab = {word for (word,tag) in tagged_words} # Using a set
    print("Total words tagged in the corpus:",len(tagged_words))
    print("Total unique tags in the corpus:",len(tags))
    print("Total unique words in the corpus:",len(vocab))
    # Get the emission and transition probabilities
    emit_matrix = emission_probabilty(tagged_words, tags, vocab)
    transit_matrix = transition_probability(nltk_data, tags)

    # sentence = "Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 ."
    while True:
        sentence = input("Enter a sentence: ")
        words = sentence.split(" ")
        process = True
        for word in words:
            if word not in vocab:
                print(f"'{word}' is not in the corpus")
                print("Avoiding this sentence...")
                process = False
                break

        if process:
            solution = viterbi(emit_matrix, transit_matrix, sentence, tags)
            for i,word in enumerate(words):
                print(f"({word} , {solution[i]})")

if __name__ =="__main__":
    main()

