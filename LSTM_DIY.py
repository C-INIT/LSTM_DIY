import torch as tc
import torch.nn as nn
import numpy as np


#未优化版本，单向
#class LSTM_v0(nn.Module):
class LSTM_Layer_v0(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 bias=True,
                 batch_first=False) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        #i_t
        self.U_i = nn.Parameter(tc.Tensor(input_size, hidden_size))
        self.V_i = nn.Parameter(tc.Tensor(hidden_size, hidden_size))
        #有无偏置
        if bias:
            self.b_i = nn.Parameter(tc.Tensor(hidden_size))
        else:
            self.b_i = tc.zeros(hidden_size, requires_grad=False)
        #f_t
        self.U_f = nn.Parameter(tc.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(tc.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(tc.Tensor(hidden_size))
        #c_t
        self.U_c = nn.Parameter(tc.Tensor(input_size, hidden_size))
        self.V_c = nn.Parameter(tc.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(tc.Tensor(hidden_size))
        #o_t
        self.U_o = nn.Parameter(tc.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(tc.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(tc.Tensor(hidden_size))
        self.init_weights()

    def init_weights(self):
        stdv = 1 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x: tc.Tensor, init_states=None):
        #如果是(batch_size,seq_len,embeding size)，则变为(seq_len,batch_size,embeding size)
        if self.batch_first:
            x = x.transpose(0, 1)
        #统一处理格式
        seq_len, batch_size, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = tc.zeros(batch_size,
                                self.hidden_size).to(x.device), tc.zeros(
                                    batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_len):
            x_t = x[t, :, :]
            i_t = tc.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = tc.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = tc.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = tc.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * tc.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = tc.cat(hidden_seq, dim=0)
        #变回去，.contiguous()可以让tensor在内存中连续
        if self.batch_first:
            hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)


#优化版本，单向
#class LSTM_v1(nn.Module):
class LSTM_Layer_v1(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 bias=True,
                 batch_first=False) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        hidden_size_4 = hidden_size * 4
        self.W = nn.Parameter(tc.Tensor(input_size, hidden_size_4))
        self.U = nn.Parameter(tc.Tensor(hidden_size, hidden_size_4))
        if bias:
            self.bias = nn.Parameter(tc.Tensor(hidden_size_4))
        else:
            self.bias = tc.zeros(hidden_size_4, requires_grad=False)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_size, batch_size, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (tc.zeros(batch_size, self.hidden_size).to(x.device),
                        tc.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(seq_size):
            x_t = x[t, :, :]
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = tc.sigmoid(gates[:, :HS]), tc.sigmoid(
                gates[:, HS:HS * 2]), tc.tanh(
                    gates[:, HS * 2:HS * 3]), tc.sigmoid(gates[:, HS * 3:])
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * tc.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = tc.cat(hidden_seq, dim=0)
        if self.batch_first:
            hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


#可反向遍历时间步
class LSTM_SingleLayer_bi(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            bias=True,
            batch_first=False,
            plus=True,  #时间步是否正向
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        hidden_size_4 = hidden_size * 4
        self.plus = plus
        self.W = nn.Parameter(tc.Tensor(input_size, hidden_size_4))
        self.U = nn.Parameter(tc.Tensor(hidden_size, hidden_size_4))
        if bias:
            self.bias = nn.Parameter(tc.Tensor(hidden_size_4))
        else:
            self.bias = tc.zeros(hidden_size_4, requires_grad=False)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):
        #为4，说明返回的是没有合并的双向LSTM输出
        if len(x.shape) == 4:
            if self.plus:
                x = x[0]
            else:
                x = x[1]
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_size, batch_size, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (tc.zeros(batch_size, self.hidden_size).to(x.device),
                        tc.zeros(batch_size, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        if self.plus:
            for t in range(seq_size):
                x_t = x[t, :, :]
                gates = x_t @ self.W + h_t @ self.U + self.bias
                i_t, f_t, g_t, o_t = tc.sigmoid(gates[:, :HS]), tc.sigmoid(
                    gates[:, HS:HS * 2]), tc.tanh(
                        gates[:, HS * 2:HS * 3]), tc.sigmoid(gates[:, HS * 3:])
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * tc.tanh(c_t)
                hidden_seq.append(h_t.unsqueeze(0))
        else:
            for t in range(seq_size - 1, -1, -1):
                x_t = x[t, :, :]
                gates = x_t @ self.W + h_t @ self.U + self.bias
                i_t, f_t, g_t, o_t = tc.sigmoid(gates[:, :HS]), tc.sigmoid(
                    gates[:, HS:HS * 2]), tc.tanh(
                        gates[:, HS * 2:HS * 3]), tc.sigmoid(gates[:, HS * 3:])
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * tc.tanh(c_t)
                hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = tc.cat(hidden_seq, dim=0)
        if self.batch_first:
            hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

#单层双向LSTM
class LSTM_Layer_bi(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 bias=True,
                 batch_first=False,
                 bidirectional=False,
                 tail=True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.tail = tail
        self.LSTM_forward = LSTM_SingleLayer_bi(input_size, hidden_size, bias,
                                                batch_first, True)
        if bidirectional:
            self.LSTM_backward = LSTM_SingleLayer_bi(input_size, hidden_size,
                                                     bias, batch_first, False)

    def forward(self, x, init_states=None):
        if self.bidirectional:
            if init_states is None:
                init_states_f = init_states_b = None
            else:
                #如果是双向，得把初始状态切开一半一半来分给两个方向的网络
                h_half, c_half = init_states[0].shape[0] / 2, init_states[
                    1].shape[0] / 2
                init_states_f = (init_states[0][:h_half],
                                 init_states[1][:c_half])
                init_states_b = (init_states[0][h_half:],
                                 init_states[1][c_half:])
        else:
            init_states_f = init_states
        hidden_seq_f, (h_tf, c_tf) = self.LSTM_forward(x, init_states_f)
        #如果双向，则返回合并后的结果
        if self.bidirectional:
            if self.tail:
                hidden_seq_b, (h_tb,
                                c_tb) = self.LSTM_backward(x, init_states_b)
                return tc.cat([hidden_seq_f, hidden_seq_b],
                                -1), (tc.cat([h_tf, h_tb],
                                            0), tc.cat([c_tf, c_tb], 0))
            else:
                hidden_seq_b, (h_tb,
                                c_tb) = self.LSTM_backward(x, init_states_b)
                return tc.stack([hidden_seq_f, hidden_seq_b],0)
                #子层无需返回隐状态和记忆信息
                # return tc.stack([hidden_seq_f, hidden_seq_b],
                #                 0), (tc.stack([h_tf, h_tb],
                #                               0), tc.stack([c_tf, c_tb], 0))
        return hidden_seq_f, (h_tf, c_tf)

#多层双向LSTM
class LSTM_bi(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 bidirectional=False) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        if num_layers == 1:
            self.LSTM = LSTM_Layer_bi(input_size, hidden_size, bias,
                                        batch_first, bidirectional)
        else:
            #False表示该LSTM不是尾部，tail参数为False
            self.LSTM = nn.Sequential(*([
                LSTM_Layer_bi(input_size, hidden_size, bias, batch_first,
                                bidirectional,False)
            ] + [
                LSTM_Layer_bi(hidden_size, hidden_size, bias, batch_first,
                                bidirectional, False)
            ] * (num_layers - 2) + [
                LSTM_Layer_bi(hidden_size, hidden_size, bias, batch_first,
                                bidirectional)
            ]))

    def forward(self, x):
        return self.LSTM(x)