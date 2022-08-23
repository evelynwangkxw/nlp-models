import numpy as np
import argparse

"""
Implementation of English POS tagger using Hidden Markov Model
"""

def train_and_test(entrain,endev,entest):
    
    """read training data to store the word-tag and tag-tag counts""" 

    readline = lambda line: line.strip().split('/')
    
    # read train file
    with open(entrain) as train:
    
        train_words = []
        train_tags = []
        
        for line in train:
    
            gw, gt = readline(line)
    
            train_words.append(gw)
            train_tags.append(gt)  
    
    # create dict storing observed tag-word counts in training set
    
    word_tag_dict, tag_ref = get_word_tag_dict_and_tag_ref(train_words, train_tags)
    
    # create dict storing observed tag-tag counts in training set 
    tag_tag_dict = get_tag_tag_dict(train_tags)
    
    # store indices for observed words
    word_lookup = get_word_lookup(train_words)
    
    # store indices for observed tags
    tag_lookup = get_tag_lookup(train_tags)
    
    # get emission probabilities (smoothing factor was selected based on tuning on validation set)
    B = get_emission(word_lookup, tag_lookup, train_tags, word_tag_dict, 0.1)
    
    # get transition probabilities 
    A = get_transmission(train_tags, tag_lookup, tag_tag_dict, 0.02)
    
    # get initial probabilities
    init = get_init(train_tags, tag_lookup,0.02)
    
    # predict using probabilities computed from training
    
    pred_output_tags, golden_test_tags, test_words = test_predict(entest, A, B, init, tag_lookup, word_lookup)
    
    # generate lines 
    
    output_lines = []
    
    for i in range(len(test_words)):
        
        output_lines.append(test_words[i] + '/' + pred_output_tags[i])

    # write to output file
    
    with open('output.txt', 'w') as f:
        
        for line in output_lines[:-1]:
            
            f.write(line)
            f.write('\n')
            
        f.write(output_lines[-1])


def get_word_tag_dict_and_tag_ref(words,tags):
    
    # calculate word tag count 

    word_tag_dict = dict()
    
    # tags allowed 
    
    tag_ref = dict()

    for i in range(len(words)):

        if words[i] != '###':

            word_tag = words[i] + tags[i]

            if word_tag in word_tag_dict.keys():

                word_tag_dict[word_tag] += 1 

            else:
                word_tag_dict[word_tag] = 1
                
            word = words[i]
            
            if word in tag_ref:
                
                if tags[i] not in tag_ref[word]:
                    tag_ref[word] += [tags[i]]
            else:
                tag_ref[word] = [tags[i]]
                
    # all unique tags
    
    unique_tags = set(tags)
    
    unique_tags.remove('###')
    
    tag_ref['UNK'] = list(unique_tags)
    
    return word_tag_dict, tag_ref

def get_tag_tag_dict(tags):

    # calculate tag tag count

    tag_tag_dict = dict()

    for i in range(len(tags)-1):

        if (tags[i] != '###'):

            tag_tag = tags[i] + tags[i+1]

            if tag_tag in tag_tag_dict.keys():

                tag_tag_dict[tag_tag] += 1

            else:
                tag_tag_dict[tag_tag] = 1

    return tag_tag_dict

def get_word_lookup(words):
    
    word_lookup = dict()
    # add 
    set_words = list(set(words)) + ['UNK']
    word_index = 0
    for word in set_words:
    
        if word != '###':
            word_lookup[word_index] = word
            word_index += 1
            
    return word_lookup

def get_tag_lookup(tags):
    
    tag_lookup = dict()
    set_tag = set(tags)
    set_index = 0
    for tag in set_tag:

        if tag != '###':
            tag_lookup[set_index] = tag
            set_index += 1
    
    return tag_lookup


# k = smoothing factor

def get_emission(word_lookup, tag_lookup, tags, word_tag_dict, k):   

    """compute the emission probabilities in HMM with add-k smoothing""" 
    
    B = np.zeros((len(word_lookup), len(tag_lookup)))

    for j in range(len(tag_lookup)):

        tag = tag_lookup[j]
        tag_count = tags.count(tag)

        for i in range(len(word_lookup)):

            word = word_lookup[i]

            word_tag = word + tag

            if word_tag in list(word_tag_dict.keys()):

                word_tag_count = word_tag_dict[word_tag]

            else:
                word_tag_count = 0

            # calculate in log space
            B[i,j] = np.log((word_tag_count + k)/(tag_count + k*len(word_lookup)))
            
    return B

def get_transmission(tags,tag_lookup, tag_tag_dict, alpha):

    """ compute the transmission probabilities in HMM with linear interpolation (weight defined by alpha) """ 
    
    A = np.zeros((len(tag_lookup), len(tag_lookup)))

    for i in range(A.shape[0]):
        tag_i = tag_lookup[i]
        ptag_count = 0

        for k in list(tag_tag_dict.keys()):

            if tag_i == k[0]:
                ptag_count += tag_tag_dict[k]

        for j in range(A.shape[1]):

            tag_j = tag_lookup[j]

            ptag_ctag = tag_i + tag_j 
            
            if ptag_ctag in (tag_tag_dict.keys()):

                ptag_ctag_count = tag_tag_dict[ptag_ctag]
            
            else:
                ptag_ctag_count = 0
                
            # unigram counts 
            
            unigram_p = tags.count(tag_j) / len(tags)
            
            # interpolation smoothing

            A[i,j] = np.log(alpha*(ptag_ctag_count/ptag_count) + ((1 - alpha)*unigram_p))
    
    return A

def get_init(tags, tag_lookup, alpha):
    
    first_tags = []
    first_tags.append(tags[0])
    
    for tag_index in range(1,len(tags)):

        if tags[tag_index-1] == '###':
            first_tags.append(tags[tag_index])
    
    init = np.zeros(len(tag_lookup))
        
    for index, tag in tag_lookup.items():
        
        unigram_p = tags.count(tag)/len(tags)
        
        init[index] = np.log(alpha*(first_tags.count(tag)/len(first_tags)) + ((1-alpha)*unigram_p))
        
    return init

def observation_to_index(observation, word_lookup):
    
    words = list(word_lookup.values())
    indices = list(word_lookup.keys())
    
    output_observation_index = []
    
    for word in observation:
        
        word_index = 0
        
        if word in word_lookup.values():
            word_index = words.index(word)
        else:
            word_index = words.index("UNK")
        
        output_observation_index.append(indices[word_index])
        
            
    return output_observation_index

def viterbi_decode(o, A, B, init, tag_lookup):
    
    """Discover the best tag sequence using decoder""" 
    
    tag_count = A.shape[0]
    obs_len = len(o)
    
    pmatrix = np.zeros((tag_count, obs_len))
    backpointer = np.zeros((tag_count, obs_len))
    
    for tag_index in range(tag_count):
        
        pmatrix[tag_index, 0] = init[tag_index] + B[o[0],tag_index]
    
    
    for i in range(1,obs_len):
        
        for tag_index in range(tag_count):
            
            new_scores = pmatrix[:,i-1] + A[:,tag_index] + B[o[i],tag_index]
            
            pmatrix[tag_index, i] = np.max(new_scores)
            
            backpointer[tag_index,i] = np.argmax(new_scores)
            
        
    bestpathprob = np.argmax(pmatrix[:, obs_len - 1])
    output_tag = []
    
    output_tag.append(tag_lookup[bestpathprob])
    
    for timestamp in range(obs_len-1,0,-1):
        bestpathprob = backpointer[int(bestpathprob),timestamp]
        output_tag.append(tag_lookup[int(bestpathprob)])
        
    output_tag.reverse()
   
    return output_tag

def test_predict(entest, A, B, init, tag_lookup, word_lookup):
    
    """Compute the tag sequence for test set using the HMM""" 

    readline = lambda line: line.strip().split('/')
    # parse test file
    test_words = []
    test_tags = []
    
    with open(entest) as test:
    
        for line in test:

            tw, tt = readline(line)
            test_words.append(tw)
            test_tags.append(tt)
            
    
    #split into single sequences
    test_obs = []
    temp_tag = []
    for x in test_words:

        if x == '###':
            test_obs.append(temp_tag)
            temp_tag = []
        else:
            temp_tag.append(x)
    
    all_output_tags = []
    
    for obs in test_obs:
        
        if len(obs) != 0:
            
                
            to_decode_index = observation_to_index(obs, word_lookup)
            
            output_tag = viterbi_decode(to_decode_index, A, B, init, tag_lookup)
            output_tag.append('###')
            
            all_output_tags.append(output_tag)
    
    pred_output_tags = [obs for sublist in all_output_tags for obs in sublist]
    
    return pred_output_tags, test_tags, test_words

def eval(gold,pred):
    readline = lambda line: line.strip().split('/')
    words = []
    groundtruth_tags = []
    predicted_tags = []
    with open (gold) as fgold, open (pred) as fpred:
        for g, p in zip(fgold, fpred):
            gw, gt = readline(g)
            pw, pt = readline(p)
            if gw == '###':
                continue
            words.append(gw)
            predicted_tags.append(pt)
            groundtruth_tags.append(gt)
    acc = sum([pt == gt for gt, pt in zip(groundtruth_tags, predicted_tags)]) / len(predicted_tags)
    print('accuracy={}'.format(acc))
    return acc

def main(train, test):

    train_and_test(train, 'placeholder',test)

    accuracy = eval(test,'output.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", help="file path to training set (txt)")
    parser.add_argument("--test", help="file path to test set (txt)")

    args = parser.parse_args() 

    main(args.train, args.test)

