import pygame
import numpy as np
import math
import colorsys
import time
import sys
pygame.init()

#init
#
# icon = pygame.Surface((32, 32), pygame.SRCALPHA)
#
# # 绘制白色背景
# icon.fill((255, 255, 255, 255))
#
# # 绘制黑色边框
# pygame.draw.rect(icon, (0, 0, 0), (0, 0, 32, 32), 1)
#
# # 绘制白色纸张
# pygame.draw.rect(icon, (255, 255, 255), (1, 1, 30, 30), 0)
#
# # 绘制黑色线条
# pygame.draw.line(icon, (0, 0, 0), (1, 1), (30, 30), 1)
# pygame.draw.line(icon, (0, 0, 0), (30, 1), (1, 30), 1)
#
# # 保存图标
# pygame.image.save(icon, "icon.png")

clean_img = pygame.image.load("./img/clean_img.jpg")
play_img = pygame.image.load("./img/auto_play.png")
paw_img = pygame.image.load("./img/aw.png")

def rainbow(x, i):
    # 引入math模块，用于计算三角函数
    # 定义一个常数c，用于调整亮度
    c = x / 255
    # 定义一个常数f，用于调整频率
    f = i * math.pi / 12
    # 计算r, g, b的值，使用三角函数和相位差来模拟彩虹的颜色变化
    r = int(127.5 * (1 + c * math.sin(f)))
    g = int(127.5 * (1 + c * math.sin(f + 2 * math.pi / 3)))
    b = int(127.5 * (1 + c * math.sin(f + 4 * math.pi / 3)))
    # 返回一个tuple(r, g, b)
    return (r, g, b)

def color_rgb(x, i, cg):
    t = int(x)
    if not cg:
        return (255-t , 255-t , 255-t )
    return rainbow(x, i)

def center_pos(x,y,w,h) :
    pass

def same_rgb(x) :
    return x,x,x

def change_to_rgb(x) :
    x = x - x.min()
    return x / x.max() *255

def pgtcenter (screen,txt,c,pos,large,kjcx) :
    try :
        FONTdef = pygame.font.Font('./meiryo.ttc',large)
        TXTdef = FONTdef.render(txt,kjcx,c)
        w = TXTdef.get_rect()[2]
        h = TXTdef.get_rect()[3]
        x,y = pos
        screen.blit(TXTdef,(x-(w/2),y-(h/2)))
    except :
        pass

def draw_cnn(screen,tensor,w,h,pix_large,stx,sty) :
    for y in range(h) :
        for x in range(w) :
            pygame.draw.rect(screen, same_rgb(tensor[y, x]),(stx + pix_large * x, sty + pix_large * y, pix_large, pix_large), 0)




icon = pygame.image.load("icon.png")

enlarge = 30
large = 28
brush_size = 1
brush_size2 = 30
brush_k = 2.8

pix_width = enlarge
pix_height = enlarge

white = (255, 255, 255)
black = (0, 0, 0)

# white = 0,0,0
# black = 255,255,255

blue = (80,80,200)

cnn_pos = 900,20
cnn_step = 123



#會這計算神經網絡
dtop1 = 30
dleft1 = 1330
neuron_num1 = 20
dtop2 = 40
dleft2 = 1540
neuron_num2 = 16
dtop3 = 100
dleft3 = 1750
neuron_num3 = 10

neuron_size = 20

dbuttom = 60

neuron_top_list1 = []
neuron_top_list2 = []
neuron_top_list3 = []

dis = (1040 - dtop1 * 2 - dbuttom) / (neuron_num1 - 1)
for i in range(neuron_num1):
    neuron_top_list1.append(int(i * dis) + dtop1)
dis = (1040 - dtop2 * 2 - dbuttom) / (neuron_num2 - 1)
for i in range(neuron_num2):
    neuron_top_list2.append(int(i * dis) + dtop2)
dis = (1040 - dtop3 * 2 - dbuttom) / (neuron_num3 - 1)
for i in range(neuron_num3):
    neuron_top_list3.append(int(i * dis) + dtop3)

dcnntop1 = 300
dcnnleft1 = 950
cnn_num1 = 2
cnnshape1 = 28
cnnsize1 = 7


dcnntop2 = 60
dcnnleft2 = 1140
cnn_num2 = 4
cnnshape2 = 10
cnnsize2 = 11

cnn_top_list1 = []
cnn_top_list2 = []

dcnnimgleft1 = (dcnnleft1)-(cnnshape1/2*cnnsize1)
dcnnimgleft2 = (dcnnleft2)-(cnnshape2/2*cnnsize2)
cnn_img_list1 = []
cnn_img_list2 = []

dis = (1040 - dcnntop1 * 2 - dbuttom) / (cnn_num1 - 1)
for i in range(cnn_num1):
    cnn_top_list1.append(int(i * dis) + dcnntop1)
    cnn_img_list1.append((int(i * dis) + dcnntop1)-(cnnshape1/2*cnnsize1))

dis = (1040 - dcnntop2 * 2 - dbuttom) / (cnn_num2 - 1)
for i in range(cnn_num2):
    cnn_top_list2.append(int(i * dis) + dcnntop2)
    cnn_img_list2.append((int(i * dis) + dcnntop2) - (cnnshape2 / 2 * cnnsize2))




def create_board() :
    # enlarge = 30·
    # large = 28
    # brush_size = 1
    #
    #
    # pix_width = enlarge
    # pix_height = enlarge

    screen = pygame.display.set_mode((large*enlarge+980,large*enlarge+178))

    screen.fill(white)

    screen.blit(clean_img, (840, 0))
    screen.blit(play_img, (840, 81))

    pygame.draw.line(screen, blue, (840,0),(840,840), 3)
    pygame.draw.line(screen, blue, (0,840),(841,840), 3)

    for right in range(cnn_num1):
        pygame.draw.line(screen, blue, (0, cnn_top_list1[right]), (dcnnimgleft1, cnn_top_list1[right]), 3)


    pygame.display.set_caption('Hand_write_board')
    pygame.display.set_icon(icon)

    board = np.zeros((large, large))


    #绘制神经网络
    # dtop1 = 40
    # dleft1 = 1350
    # neuron_num1 = 10
    # dtop2 = 40
    # dleft2 = 1550
    # neuron_num2 = 16
    # dtop3 = 100
    # dleft3 = 1750
    # neuron_num3 = 10
    #
    #
    # neuron_top_list1 = []
    # neuron_top_list2 = []
    # neuron_top_list3 = []
    #
    # dis = (1040 - dtop1*2)/(neuron_num1-1)
    # for i in range(neuron_num1) :
    #     neuron_top_list1.append(int(i*dis)+dtop1)
    # dis = (1040 - dtop2*2)/(neuron_num2-1)
    # for i in range(neuron_num2):
    #     neuron_top_list2.append(int(i*dis) + dtop2)
    # dis = (1040 - dtop3*2)/(neuron_num3-1)
    # for i in range(neuron_num3) :
    #     neuron_top_list3.append(int(i*dis)+dtop3)


    for right in range(neuron_num1) :
        for left in range(neuron_num2) :
            pygame.draw.aaline(screen, color_rgb(255,right*2,1), (dleft1,neuron_top_list1[right]),(dleft2,neuron_top_list2[left]), 2)

    for right in range(neuron_num2) :
        for left in range(neuron_num3) :
            pygame.draw.aaline(screen, color_rgb(255,right*-2,1), (dleft2,neuron_top_list2[right]),(dleft3,neuron_top_list3[left]), 2)

    for right in range(cnn_num1) :
        for left in range(cnn_num2) :
            pygame.draw.aaline(screen, color_rgb(255,left*10,1), (dcnnleft1,cnn_top_list1[right]),(dcnnleft2,cnn_top_list2[left]), 2)

    for right in range(neuron_num1) :
        for left in range(cnn_num2) :
            pygame.draw.aaline(screen, color_rgb(255,right*-10,1), (dleft1,neuron_top_list1[right]),(dcnnleft2,cnn_top_list2[left]), 2)

    for right in range(neuron_num3) :
        pygame.draw.circle(screen, black, (dleft3, neuron_top_list3[right]), radius=neuron_size)
        pgtcenter(screen, str(right), black, (dleft3+neuron_size+13, neuron_top_list3[right]), 20, True) # 數字顯示
    for right in range(neuron_num1):
        pygame.draw.circle(screen, black, (dleft1, neuron_top_list1[right]), radius=neuron_size)
    for right in range(neuron_num2):
        pygame.draw.circle(screen, black, (dleft2, neuron_top_list2[right]), radius=neuron_size)

    # 卷积
    # print(rgb_output[0].shape)
    for right in range(cnn_num1):
        pygame.draw.rect(screen, blue, (dcnnimgleft1 - 3, cnn_img_list1[right] - 3,cnnsize1*cnnshape1 + 6, cnnsize1*cnnshape1 + 6), 0)

    for right in range(cnn_num2):
        pygame.draw.rect(screen, blue, (dcnnimgleft2 - 3, cnn_img_list2[right] - 3, cnnsize2*cnnshape2 + 6, cnnsize2*cnnshape2 + 6), 0)

    pgtcenter(screen, "卷積輸出1", black, (dcnnleft1, 1005), 28, True)
    pgtcenter(screen, "卷積輸出2", black, (dcnnleft2, 1005), 28, True)

    pgtcenter(screen, "全連接輸入層", black, (dleft1, 1005), 28, True)
    pgtcenter(screen, "全連接隱藏層", black, (dleft2, 1005), 28, True)
    pgtcenter(screen, "輸出", white, (dleft3, 1015), 28, True)

    pgtcenter(screen, "手寫區域", blue , (420, 890), 55, True)
    pgtcenter(screen, "數字0-9神經網絡識別", (200,50,50), (420, 995), 55, True)


    return screen,board



def update_screen(screen,board,output_list,pred,auto_show,is_change = False) :


    for k in pygame.event.get():
        if k.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    keypress = pygame.key.get_pressed()

    if keypress[pygame.K_q] :
        pygame.quit()
        return board
    if keypress[pygame.K_w] or pygame.mouse.get_pressed()[0] :
        mpx,mpy = pygame.mouse.get_pos()
        #print(mpx,mpy)
        if(840<mpx<=920 and 0<=mpy<=80) :
            #print(123)
            board = np.zeros((large, large))
            is_change = True
            auto_show = False
        elif (840<mpx<=920 and 80<mpy<=160) :
                auto_show = True
                is_change = True
        else :
            for y in range(-brush_size,brush_size+1) :
                for x in range(-brush_size,brush_size+1) :
                    int_x = min(int(mpx / enlarge)+x, 27)
                    int_y = min(int(mpy / enlarge)+y, 27)

                    block_x = (int_x + 0.5) * enlarge
                    block_y = (int_y + 0.5) * enlarge

                    #print(int_x,int_y)
                    if(0<=int_x<=27 and 0<=int_y<=27) :
                        distance = ((block_x-mpx)/brush_k)**2+((block_y-mpy)/brush_k)**2+0.00001
                        board[int_y,int_x] = min(brush_size2*int(255/distance)+board[int_y,int_x],250)
                        is_change = True
                        auto_show = False

    if not is_change :
        return board,is_change,auto_show

    for y in range(len(board)) :
        for x in range(len(board[y])) :
            pygame.draw.rect(screen,color_rgb(board[y,x],x+y,board[y,x]!=0),(enlarge*x,enlarge*y,pix_width,pix_height),0)


    #繪製神經網絡
    if output_list != 0 :

        rgb_output = change_to_rgb(output_list[-4])
        for right in range(neuron_num1) :
            pygame.draw.circle(screen, same_rgb(rgb_output[0,right]), (dleft1, neuron_top_list1[right]), radius=neuron_size-3)

        rgb_output = change_to_rgb(output_list[-2])
        for right in range(neuron_num2) :
            pygame.draw.circle(screen, same_rgb(rgb_output[0,right]), (dleft2, neuron_top_list2[right]), radius=neuron_size-3)

        rgb_output = change_to_rgb(output_list[-1])
        for right in range(neuron_num3) :
            pygame.draw.circle(screen, same_rgb(rgb_output[0,right]), (dleft3, neuron_top_list3[right]), radius=neuron_size-3)

        #卷积
    rgb_output = change_to_rgb(output_list[3][0][:cnn_num1])
    for right in range(cnn_num1):
        draw_cnn(screen, rgb_output[right], cnnshape1, cnnshape1, cnnsize1, dcnnimgleft1, cnn_img_list1[right])

    rgb_output = change_to_rgb(output_list[6][0][:cnn_num2])
    for right in range(cnn_num2):
        draw_cnn(screen, rgb_output[right], cnnshape2, cnnshape2, cnnsize2, dcnnimgleft2, cnn_img_list2[right])




    pygame.draw.rect(screen, white,(dleft3+53, 0, 100, 1040), 0)

    #pygame.draw.circle(screen, (80,250,50), (dleft3+76, neuron_top_list3[pred]),radius=23)

    screen.blit(paw_img,(dleft3+53,neuron_top_list3[pred]-21))
    #pygame.draw.rect(screen, white,(840, 0, 80, 80), 0)
    #screen.blit(clean_img,(840,0))


    if auto_show :
        pgtcenter(screen, "自動播放中", (200,100,100), (105, 865), 40, True)
    else :
        pygame.draw.rect(screen, white, (0, 845, 210, 90), 0)



    pygame.display.update()

    return board,is_change,auto_show


def get_hand_write() :
    # 定义一个彩虹函数，输入x和i，输出一个tuple(r, g, b)
    def rainbow(x, i):
        # 引入math模块，用于计算三角函数
        # 定义一个常数c，用于调整亮度
        c = x / 255
        # 定义一个常数f，用于调整频率
        f = i * math.pi / 12
        # 计算r, g, b的值，使用三角函数和相位差来模拟彩虹的颜色变化
        r = int(127.5 * (1 + c * math.sin(f)))
        g = int(127.5 * (1 + c * math.sin(f + 2 * math.pi / 3)))
        b = int(127.5 * (1 + c * math.sin(f + 4 * math.pi / 3)))
        # 返回一个tuple(r, g, b)
        return (r, g, b)

    def bw_rgb(x) :
        t = 255-int(x)
        return (t,t,t)

    def color_rgb(x,i,cg) :
        t = int(x)
        if not cg :
            return (t+255,t+255,t+255)
        return rainbow(x, i)

    white = (255,255,255)
    black = (0,0,0)

    enlarge = 30
    large = 28
    brush_size = 1


    pix_width = enlarge
    pix_height = enlarge

    screen = pygame.display.set_mode((large*enlarge,large*enlarge))

    pygame.display.set_caption('Hand_write_board')
    pygame.display.set_icon(icon)
    board = np.zeros((large,large))

    while True :
        for k in pygame.event.get():
            if k.type == pygame.QUIT:
                pygame.quit()
                exit()

        keypress = pygame.key.get_pressed()

        if keypress[pygame.K_q] :
            pygame.quit()
            exit()
            return board
        if keypress[pygame.K_w] or pygame.mouse.get_pressed()[0] :
            mpx,mpy = pygame.mouse.get_pos()
            mpx = int(mpx / enlarge)
            mpy = int(mpy / enlarge)
            for y in range(-brush_size,brush_size+1) :
                for x in range(-brush_size,brush_size+1) :
                    try :
                        board[mpy+y,mpx+x] = max(240-(abs(y)+abs(x))*10,board[mpy+y,mpx+x])
                    except :
                        pass

        #screen.fill((255,0,0))

        for y in range(len(board)) :
            for x in range(len(board[y])) :
                pygame.draw.rect(screen,color_rgb(board[y,x],x+y,board[y,x]!=0),(enlarge*x,enlarge*y,pix_width,pix_height),0)

        pygame.display.update()
        #pygame.time.delay(10)

if __name__ == "__main__" :
    #board = get_hand_write()
    screen,board = create_board()
    while True :
        board,is_change = update_screen(screen,board,0,0)

    print(board)
