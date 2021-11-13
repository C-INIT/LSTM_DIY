# https://github.com/graykode/nlp-tutorial
# code by Tae Hwan Jung @graykode
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim

def make_batch(sentences, word_dict):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split() # space tokenizer
        input = [word_dict[n] for n in word[:-1]] # create (1~n-1) as input
        target = word_dict[word[-1]] # create (n) as target, We usually call this 'casual language model'

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch

# Build Model, 继承pytorch的Module类
class NNLM(nn.Module):
    def __init__(self, seq_len, hidden_size, embedding_size, n_class):
        super(NNLM, self).__init__()
        self.embedding = nn.Embedding(n_class, embedding_size)

        self.W1 = nn.Linear(seq_len * embedding_size, hidden_size, bias=False)
        self.B1 = nn.Parameter(torch.ones(hidden_size))

        self.W2 = nn.Linear(hidden_size, n_class, bias=False)
        self.W3 = nn.Linear(seq_len * embedding_size, n_class, bias=False)
        self.B2 = nn.Parameter(torch.ones(n_class))

    def forward(self, X):   # X : [batch_size, len]
        # 词嵌入，将词的索引转换为一个向量
        X = self.embedding(X) # X : [batch_size, len, embedding_size]
        
        batch_size, seq_len, embedding_size = X.shape

        # 维度转换
        X = X.view(-1, seq_len * embedding_size) # [batch_size, len * embedding_size]

        # 送入网络进行计算
        # 激活函数tanh
        Y1 = torch.tanh(self.B1 + self.W1(X)) # [batch_size, hidden_size]
        output = torch.softmax(self.W2(Y1) + self.B2 + self.W3(X), dim=1) # [batch_size, n_class]

        return output

def print_model(model):
	for name, parameters in model.named_parameters():
		print(name, parameters)

def print_grad(model):
	for name, parameters in model.named_parameters():
		print(name, parameters.grad)

def compute_loss(output, target_batch):
    loss = 0
    for idx, _ in enumerate(target_batch):
        log_q = torch.log(output[idx])
        p = torch.zeros(output[idx].shape)
        p[target_batch[idx].data] = 1
        
        # 循环求和
        for idx, _ in enumerate(p):
            loss -= p[idx] * log_q[idx]
    return loss/3

if __name__ == '__main__':
	# 参数配置
    seq_len = 2 # number of steps, n-1 in paper			        输入序列的长度
    hidden_size = 2 # number of hidden size, h in paper	    网络隐藏层的维度
    embedding_size = 2 # embedding size, m in paper			词嵌入的维度

    # 样例输入
    sentences = ["i like dog", "i love coffee", "i hate milk"]

    # 创建词表
    tmp_word_list = " ".join(sentences).split()
    
    word_list = []
    for i in tmp_word_list:
        if i not in word_list:
            word_list.append(i)

    word_dict = {w: i for i, w in enumerate(word_list)}     # 单词对索引
    number_dict = {i: w for i, w in enumerate(word_list)}   # 索引对单词
    n_class = len(word_dict)  # number of Vocabulary

    for k, v in word_dict.items():
        print(k, '\t', v)
    for k, v in number_dict.items():
        print(k, '\t', v)

    # 构建输入与输出的数学表示
    input_batch, target_batch = make_batch(sentences, word_dict)
    # i like, i love, i hate
    print(input_batch)      # 二维数组 
    # dog, coffee, milk
    print(target_batch)     # 一维数组
    # 转化为深度学习框架(pytorch)可以训练用的tensor
    input_batch = torch.LongTensor(input_batch)     #.cuda()     
    target_batch = torch.LongTensor(target_batch)   #.cuda()
    print(input_batch, input_batch.shape)
    print(target_batch, target_batch.shape)

    # 创建模型
    model = NNLM(seq_len, hidden_size, embedding_size, n_class) #.cuda()
    # 引出分类任务
    output = model(input_batch)
    print('output', output)

    # 模型输出
    print(model)
    print_model(model)

    # 创建目标函数 和 优化器
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 训练
    for epoch in range(500):
        # 梯度清零
        optimizer.zero_grad()
        output = model(input_batch)
        print('output', output)

        # output : [batch_size, n_class], target_batch : [batch_size]
        loss = criterion(torch.log(output), target_batch)
        print('loss', loss)
        print('my_loss', compute_loss(output, target_batch))
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        # 反向传播，计算梯度
        loss.backward()
        #print_grad(model)
        # 优化模型参数
        optimizer.step()
        #print_model(model)
        break

    # 预测
    predict = model(input_batch).data.max(1, keepdim=True)[1]   #.cpu()
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])