

def read_dataset(inputfile):
    X,Y=[],[]
    with open(inputfile,'r') as fin:
        temp_X=[]
        temp_Y=[]
        for line in fin:
            if line.strip()=='':
                continue
            if line.startswith('Sentence__'):
                if temp_X.__len__()>0:
                    X.append(temp_X)
                    Y.append(temp_Y)
                    temp_X=[]
                    temp_Y=[]
            else:
                #print(line)
                token,label = line.split('\t')
                if token.strip()!='':
                    temp_X.append(token.strip())
                    temp_Y.append(label.strip())
    return X,Y

def get_vocab(X,Y,min_freq=3):
    x_vocab,y_vocab={},set()
    for idx in range(len(X)):
        for token_idx in range(len(X[idx])):
            if X[idx][token_idx] in x_vocab:
                x_vocab[X[idx][token_idx]]+=1
            else:
                x_vocab[X[idx][token_idx]]=1
            y_vocab.add(Y[idx][token_idx])
    x_vocab_list=[x for x in x_vocab if x_vocab[x]>min_freq]
    X_vocab={x:i+2 for i,x in enumerate(x_vocab_list)}
    X_vocab['__MASK__']=0
    X_vocab['__OOV__']=1
    Y_vocab={y:i for i,y in enumerate(y_vocab)}

    return X_vocab,Y_vocab


def encode_dataset(X,Y,X_vocab,Y_vocab):
    newX,newY=[],[]
    for idx in range(len(X)):
       tokens=[X_vocab[x] if x in X_vocab else 1  for x in X[idx]]
       labels=[Y_vocab[y] for y in Y[idx]]
       newX.append(tokens)
       newY.append(labels)
    return newX,newY

def idx_to_word(x,vocab):
    return [vocab[word] for word in x if word !=0]

def idx_to_label(y,vocab):
    return [vocab[label] for label in y]