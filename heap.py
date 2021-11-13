# l = [0.01,0.15,0.12,0.07,0.08,0.13,0.15,0.03,0.17,0.09]
# l.sort()
# def step():
#     global l
#     if len(l) != 1:
#         a,b = l[0],l[1]
#         l = l[2:]
#         print(a,b,round(a+b,2))
#         l.append(round(a+b,2))
#         l.sort()
#         return True
#     else:
#         print('done')
#         return False

# while(step()):
#     pass
a = 5*0.01+5*0.03+4*0.07+4*0.08+4*0.09+3*0.12+3*0.13+3*0.15+3*0.15+2*0.17
b = 4*0.01+4*0.03+4*0.07+4*0.08+3*0.09+3*0.12+3*0.13+3*0.15+3*0.15+3*0.17
print(a,b)