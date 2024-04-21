"""
Nov 7,9,13 ---> witing and makeing story writer better 


- dictionary keys must be immutable
Tupple
    - x = ("the", "hobbit")
    - immutable list
    - can cancatinate
    - (3,) * 4  ===> (3, 3, 3, 3)
not a tupple:
    x = (the")
singleton tupple
    x = (3,) ---> lenght 1


"""


from typing import List, Dict, TextIO

import random



"""
Lec_16 just for showing what can be done with dictonaries, make a "new" story using dirtionary of words form another story
key = word, values = all the words that follow that have followed that word in the story 
eg. key= the, value= [tree, car, tea]

"""


def associate_pair(d: Dict[str, List[str]], key: str, value: str):
    '''Add the key-value pair to d. keys may need to be associated with
    multiple values, so values are placed in a list.
    Assumption: key is immutable
    '''

    if key in d:
        d[key].append(value) # if the key is already in the dictionay, add to list of values
                            # there are multipul of the same value to the same key --> writing style
                            # my: [the, the]
    else:
        d[key] = [value] # is key is not in dictionary, new entry



def make_dictionary(file_name: str) -> Dict[str, List[str]]:
    '''Return a dictionary where the keys are words in f and the value
    for a key is the list of words that were found to follow the key in f.
    '''

    d = {}
    context = '' # context is previous word /key, word is value
    file = open(file_name, "r") 

    for line in file:
        word_list = line.split() #--> defult is spliting on space btw words in file

        for word in word_list:
            associate_pair(d, context, word) #-> update dictionary
            context = word #-> update context

    associate_pair(d, context, '') #---> last context in the story has no following word, assign a word
    file.close()
    return d

def append_dictionary(file_name: str, d, k) -> Dict[str, List[str]]: #paramaters: dictionary and k--> int for context
    '''Return a  new dictionary where the keys have k contex in f and the value
    for a key is the list of words that were found to follow the key in f.
    '''

    context = ('',)* k # context is previous k words /key, word is value
    file = open(file_name, "r") 

    for line in file:
        word_list = line.split() #--> defult is spliting on space btw words in file

        for word in word_list:
            associate_pair(d, context, word) #-> update dictionary
            context = context[1:] + (word,) #-> update context

    file.close()
    return d


def mimic_text(word_dict: Dict[str, List[str]], num_words: int) -> str:
    '''Based on the word patterns in word_dict, return a string that mimics
    that text, and has num_words words.
    '''

    story = ''
    context = ''

    for i in range(num_words):
        # Choose the next word, based on context
        next_words = word_dict[context]
        word = next_words[random.randint(0, len(next_words) - 1)]

        story = story + ' ' + word # add to story ---> how to output , space!!
        context = word

    return story


def mimic_text2(word_dict: Dict[str, List[str]], num_words: int, k) -> str: #--> context is  k word long
    '''Based on the word patterns in word_dict, return a string that mimics
    that text, and has num_words words.
    '''

    story = ''
    context = ('',)* k

    for i in range(num_words):
        # Choose the next word, based on context
        next_words = word_dict[context]
        word = next_words[random.randint(0, len(next_words) - 1)]

        story = story + ' ' + word # add to story ---> how to output , space!!
        context = context[1:] + (word,) #-> update context

    return story


print(mimic_text(make_dictionary("shakespeare.txt"), 50))
