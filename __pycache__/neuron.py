from tkinter import *
from tkinter import filedialog
from tkinter import colorchooser
import math
from cnn import Node
from NodeCollection import NodeCollection
from  tkinter import ttk
from fc import Network
from connection import Connection
from PIL import Image, ImageTk
from cnn import Node
from cnn import PoolingConnection,ReshapeConnection,\
    ConcatConnection,TensorConnection,Filter
import cnn
# 定义一个画布类

import generator_point
class MyComboxList:

    def __init__(self,root,values):
        self.values=values
        comvalue = StringVar()  # 窗体自带的文本，新建一个值
        self.comboxlist = ttk.Combobox(root, textvariable=comvalue)  # 初始化
        self.comboxlist["values"] = values
        self.comboxlist.current(0)  # 选择第一个
        self.comboxlist.bind("<<ComboboxSelected>>", self.getCurrentValue)  # 绑定事件,(下拉列表框被选中时，绑定go()函数)
        self.comboxlist.pack()

    def getCurrentValue(self,*args):  # 处理事件，*args表示可变参数
        print(self.comboxlist.get())
        return self.comboxlist.get()  # 打印选中的值

class Circle:
    def __init__(self,x,y,num,CircleNum):
        self.x=x
        self.y=y
        self.num=num
        self.start=-1
        self.CircleNum=CircleNum

class CircleLine:
    def __init__(self,x,y,conn,kind):
        self.x=x
        self.y=y
        self.conn=conn
        self.kind=kind
class CNNNode:
    def __init__(self,x,y,node):
        self.x=x
        self.y=y
        self.node=node
class MyCanvas:
    def __init__(self,root,width=1100,height=800):
        #i主要用来区分画的图形
        self.i=0
        self.CircleNum=0
        self.state=''#表示要画的图形(line,oval,arc,rectangle)
        self.color='black' #画笔颜色，默认黑色
        self.root=root
        self.canvas=Canvas(root,width=width,height=height,bg='white')
        self.canvas.pack()
        self.canvas.bind('<Button-1>',self.Press)
        self.Circle=[]
        self.CNNNodeCon=[]
        self.NodeCollection=NodeCollection()
        self.DoubleNodeList=[]
        self.CNNCollectionS=[]
        # self.NodeCollectionS=[]
        self.NodeCollectionS = []
        self.CircleS = [[], [], [], [], [], []]
        self.ConnectionS=[]
        self.layers=[]
        self.CNNNodeCollection=[]
        self.CircleLineS = []
        image = Image.open("conv.png")
        self.im = ImageTk.PhotoImage(image)
        image1 = Image.open("triangle.png")
        self.triangle = ImageTk.PhotoImage(image1)
        image2 = Image.open("contact.png")
        self.contact = ImageTk.PhotoImage(image2)
        image3 = Image.open("dense.png")
        self.dense = ImageTk.PhotoImage(image3)
        self.CNNList=[]
        self.CNNNodeS=[]
        for i in range(6):
            self.NodeCollectionS.append(NodeCollection())

    # def conv(self):
    #     print("23234")
    #press函数表示当鼠标在画布上按下时初始化鼠标按下位置

    def DenseInit(self,layers):
        for i in range(layers.__len__()):
            x=115+i*150
            for j in range(layers[i]):
                self.add_circle(i,x)
    def drawContactNode(self,x,y):
        num = self.canvas.create_oval(x - 30, y - 30, x + 30, y + 30,
                                      outline=self.color, tags='oval', width=2)

        self.canvas.create_oval(x - 13 - 7, y - 13 - 7, x - 13 + 7, y - 13 + 7,
                                outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x + 13 - 7, y + 13 - 7, x + 13 + 7, y + 13 + 7,
                                outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x - 13 - 7, y + 13 - 7, x - 13 + 7, y + 13 + 7,
                                outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x + 13 - 7, y - 13 - 7, x + 13 + 7, y - 13 + 7,
                                outline=self.color, tags='oval', width=2)
        self.canvas.create_image(x, y, image=self.contact, tags="image")
        self.CircleNum += 1
        self.CNNNodeS.append(Circle(x, y, num, self.CircleNum))
        self.Circle.append(Circle(x, y, num, self.CircleNum))
        self.CNNNodeCon.append(CNNNode(x,y,Node()))
        pass
    def drawContactLine(self,startX,startY,endX,endY):
        x = (startX + endX) / 2
        y = (startY + endY) / 2
        self.canvas.create_line(startX, startY, endX, endY,
                                fill=self.color,
                                tags='line',
                                width=2,
                                arrow="last",
                                arrowshape='8 10 3')
        self.canvas.create_image(x, y, image=self.contact, tags="image")
        pass
    # def drawNetWork(self,itemn,startX,startY,endX,endY):
    #     key = list(itemn.keys())[0]
    #     if key=="conv":
    #         self.drawCNNLine(startX,startY,endX,endY)
    #         self.drawCNNNode(endX + 30,endY)
    #         pass
    #     if key=="pool":
    #         self.drawPoolLine(startX,startY,endX,endY)
    #         self.drawPoolNode(endX + 30,endY)
    #         pass
    #     if key=="dense":
    #         self.drawDenseLine(startX,startY,endX,endY)
    #         self.drawDenseNode(endX + 30,endY)
    #     if key=="contact":
    #         self.drawSpreadLine(startX,startY,endX,endY)
    #         self.drawCNNNode(endX+30,endY)
    #         pass
    def drawCNN(self):
        x = 100
        y = 150
        for item in CNNList:

            if type(item).__name__ == 'list' and type(item[0]).__name__ == 'list':
                self.drawCNNNode(x,y)
                le=len(item)

                if le%2==0:
                    res=150/((le/2)+1)
                else:
                    res=300/le

                endy1 = 0
                endy2 = 300
                switch=0
                for itemn in item:
                    x = -100
                    y = 150
                    first=True
                    if switch==0:
                        endy1 = endy1 + res
                        switch = 1
                    else:
                        endy2 = endy2 - res
                        switch = 0
                    for ite in itemn:
                        if first:
                            if switch==1:
                                self.drawNetWork(ite, x + 30, y, x + 75, endy1)
                                x = x + 75
                                y = endy1
                            else:
                                self.drawNetWork(ite, x + 30, y, x + 75, endy2)
                                x = x + 75
                                y = endy2
                            first=False
                        else:
                            self.drawNetWork(ite,x,y,x+60,y)
                            x = x + 210
                        self.drawNetWork({"contact":1},x,y,x+75,150)
                x=x+75+60
                y=150



            if type(item).__name__ == 'list' and type(item[0]).__name__ != 'list':
                first=True
                x = 100
                y = 150
                self.drawCNNNode(x,y)
                for itemn in item:
                    if first:
                        self.drawNetWork(itemn,x+30,y,x+150,y)
                        x= x + 210
                    else:
                        self.drawNetWork(itemn, x, y, x + 150, y)
                        x = x + 210


            if type(item).__name__ != 'list':
                self.drawNetWork(item,x,y,x+150,y)


            # print(item)

            # if list(item.keys())[0]=="conv":
            #     self.drawCNNNode(x, y)
            #     self.drawCNNLine(x+30,y,x+120,y)
            # if list(item.keys())[0]=="pool":
            #
            #     self.drawPool()
            # if list(item.keys())[0]=="dense":
            #     self.drawDense()
        pass
    def drawPoolLine(self,startX,startY,endX,endY):
        self.canvas.create_line(startX, startY, endX, endY,
                                fill=self.color,
                                tags='line',
                                width=2,
                                arrow="last",
                                arrowshape='8 10 3')
        pass
    def drawPoolNode(self,x,y):
        num = self.canvas.create_oval(x - 30, y - 30, x + 30, y + 30,
                                      outline=self.color, tags='oval', width=2)
        self.canvas.create_line(x, y-10, x, y+10,
                                fill=self.color,
                                tags='line',
                                width=2,
                                arrow="last",
                                arrowshape='8 10 3')

        self.canvas.create_oval(x - 11 - 7, y - 11 - 7, x - 11 + 7, y - 11 + 7,
                                outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x + 11 - 7, y + 11 - 7, x + 11 + 7, y + 11 + 7,
                                outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x - 11 - 7, y + 11 - 7, x - 11 + 7, y + 11 + 7,
                                outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x + 11 - 7, y - 11 - 7, x + 11 + 7, y - 11 + 7,
                                outline=self.color, tags='oval', width=2)

        self.CircleNum += 1
        # self.canvas.create_text(x, y, text=self.CircleNum,tags="text")
        self.CNNNodeS.append(Circle(x, y, num, self.CircleNum))
        self.Circle.append(Circle(x, y, num, self.CircleNum))
        self.CNNNodeCon.append(CNNNode(x, y, Node()))
        pass
    def drawSpreadLine(self,startX,startY,endX,endY):
        x = (startX + endX) / 2
        y = startY
        self.canvas.create_line(startX, startY, endX, endY,
                                fill=self.color,
                                tags='line',
                                width=2,
                                arrow="last",
                                arrowshape='8 10 3')
        self.canvas.create_image(x, y, image=self.triangle, tags="image")
        pass
    def drawDenseLine(self,startX, startY, endX, endY):
        self.canvas.create_line(startX, startY, endX, endY,
                                fill=self.color,
                                tags='line',
                                width=2,
                                arrow="last",
                                arrowshape='8 10 3')
        pass
    def drawDenseNode(self,x,y):
        num = self.canvas.create_oval(x - 30, y - 30, x + 30, y + 30,
                                      outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x - 7, y - 7, x  + 7, y + 7,
                                outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x - 7, y + 18 - 7, x  + 7, y + 18 + 7,
                                outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x  - 7, y - 18 - 7, x  + 7, y - 18 + 7,
                                outline=self.color, tags='oval', width=2)

        self.CircleNum += 1
        self.Circle.append(Circle(x, y, num, self.CircleNum))
        # self.canvas.create_text(x, y, text=self.CircleNum,tags="text")
        self.CNNNodeS.append(Circle(x, y, num, self.CircleNum))
        self.CNNNodeCon.append(CNNNode(x, y, Node()))
        pass
    def drawCNNLine(self,startX,startY,endX,endY):
        x=(startX+endX)/2
        y=startY
        self.canvas.create_line(startX, startY, endX, endY,
                                fill=self.color,
                                tags='line',
                                width=2,
                                arrow="last",
                                arrowshape='8 10 3')
        self.canvas.create_image(x, y, image=self.im,tags="image")
        pass
    def clearAll(self):
        self.canvas.delete("line")
        self.canvas.delete("oval")
        self.canvas.delete("image")
        self.canvas.delete("text")
        self.Circle.clear()
        self.DoubleNodeList.clear()
        # self.NodeCollectionS.clear()
        for item in self.CircleS:
            item.clear()
        self.ConnectionS.clear()
        self.layers.clear()
        self.CNNList.clear()
        self.CNNNodeS.clear()
        # self.NodeCollectionS.clear()
        self.CircleNum=0
        for item in self.NodeCollectionS:
            item.node_collection.clear()
        # self.CircleS.clear()
        # self.Circle.clear()

        # for item in self.Circle:
        #     self.canvas1.

        pass
    def drawPoolLine(self,startX,startY,endX,endY):
        x = (startX + endX) / 2
        y = startY
        self.canvas.create_line(startX, startY, endX, endY,
                                fill=self.color,
                                tags='line',
                                width=2,
                                arrow="last",
                                arrowshape='8 10 3')
        self.canvas.create_image(x, y, image=self.im, tags="image")
        pass
    def drawCNNNode(self,x,y):
        num=self.canvas.create_oval(x - 30, y - 30, x + 30, y + 30,
            outline=self.color, tags='oval', width=2)

        self.canvas.create_oval(x - 11 - 7, y - 11 - 7, x - 11 + 7 , y - 11 + 7,
            outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x + 11 - 7, y + 11 - 7, x + 11 + 7, y + 11 + 7,
            outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x - 11 - 7, y + 11 - 7, x - 11 + 7, y + 11 + 7,
            outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x + 11 - 7, y - 11 - 7 , x + 11 + 7, y - 11 + 7,
            outline=self.color, tags='oval', width=2)

        self.CircleNum += 1
        self.Circle.append(Circle(x, y, num, self.CircleNum))
        # self.canvas.create_text(x, y, text=self.CircleNum,tags="text")
        self.CNNNodeS.append(Circle(x, y, num, self.CircleNum))
        self.CNNNodeCon.append(CNNNode(x, y, Node()))
        pass
    def add_circle(self,i,x):
        if self.NodeCollectionS[i].node_list().__len__()>=6:
            return
        j=self.NodeCollectionS[i].node_list().__len__()
        y=(100+j*90)
        num = self.canvas.create_oval(x-30,y-30,x+30,y+30,
        outline=self.color, tags='oval', width=2)
        # print(num)
        self.CircleS[i].append(Circle(x, y, num, self.CircleNum))
        # self.canvas.create_text(x, y, text=self.CircleNum, )
        node = Node(self.CircleNum)
        self.NodeCollectionS[i].add_node(node)
        self.CircleNum += 1
        # self.canvas.delete('oval')
        for n in range(6):
            if self.CircleS[n].__len__()>0:
                for j in range(n+1,6):
                    if self.CircleS[j].__len__()>0:
                        self.drawAllLine(self.CircleS[n],self.CircleS[j])
                        break


    def drawAllLine(self,startCircle,endCircle):

        for c3 in startCircle:
            for c4 in endCircle:
                a = c3.x - c4.x
                b = c3.y - c4.y
                if b==0:
                    b=c3.y
                self.canvas.create_line(c3.x + 30, c3.y, c4.x - 30, c4.y ,
                                                fill=self.color,
                                                tags='line',
                                                width=2,
                                                arrow="last",
                                                arrowshape='8 10 3')
                # if a < 0 and b < 0:
                #     self.canvas.create_line(c3.x + 20, c3.y + 20, c4.x - 20, c4.y - 20,
                #                             fill=self.color,
                #                             tags='line',
                #                             width=2,
                #                             arrow="last",
                #                             arrowshape='8 10 3')
                # elif a < 0 and b > 0:
                #     self.canvas.create_line(c3.x + 20, c3.y - 20, c4.x - 20, c4.y + 20,
                #                             fill=self.color,
                #                             tags='line',
                #                             width=2,
                #                             arrow="last",
                #                             arrowshape='8 10 3')
                #
                # elif a > 0 and b > 0:
                #     self.canvas.create_line(c3.x - 20, c3.y - 20, c4.x + 20, c4.y + 20,
                #                             fill=self.color,
                #                             tags='line',
                #                             width=2,
                #                             arrow="last",
                #                             arrowshape='8 10 3')
                # elif a > 0 and b < 0:
                #     self.canvas.create_line(c3.x - 20, c3.y + 20, c4.x + 20, c4.y - 20,
                #                             fill=self.color,
                #                             tags='line',
                #                             width=2,
                #                             arrow="last",
                #                             arrowshape='8 10 3')
    def setWidth_Height(self,width,height):
        self.canvas.configure(width=width,height=height)
        self.canvas.pack(side="bottom")
    def Press(self,event):
        #每次按下鼠标都会i+1,因此按下鼠标画下的图形都会有为一个标记
        if self.state=="oval":
            self.CircleNum+=1
        self.i += 1
        self.x=event.x
        self.y=event.y
        if self.state=="one_dim":
            self.drawDenseNode(self.x,self.y)
            pass
        if self.state == "two_dim":
            self.drawCNNNode(self.x,self.y)
            pass
        if self.state == "three_dim":
            self.drawContactNode(self.x,self.y)
            pass
        if self.state == "four_dim":
            self.drawPoolNode(self.x,self.y)
            pass
        if self.state == "five_dim":
            self.drawPaddingNode(self.x,self.y)
            pass
        if self.state == "conv":
            self.drawLine("conv")
            pass
        if self.state == "equal":
            self.drawLine("equal")
            pass
        if self.state == "vector":
            self.drawLine("vector")
            pass
        if self.state == "dense":
            self.drawLine("dense")
            pass
        if self.state == "contact":
            self.drawLine("contact")
            pass
        if self.state == 'oval':
            num=self.canvas.create_oval(self.x-30, self.y-30, self.x+30, self.y+30,
                                        outline=self.color,tags='oval',width = 2)
            # print(num)
            self.Circle.append(Circle(self.x, self.y, num,self.CircleNum))
            self.canvas.create_text(self.x, self.y, text=self.CircleNum,tags="text")
            node=Node(self.CircleNum)
            self.NodeCollection.add_node(node)
        if self.state == 'StartCircle':
            for c in self.Circle:
                if math.sqrt(math.fabs(event.x-c.x)*math.fabs(event.x-c.x)+math.fabs(event.y-c.y)*math.fabs(event.y-c.y))<(30):
                    self.canvas.itemconfig(c.num, outline="red")
                    c.start=1
        if self.state == 'EndCircle':
            for c in self.Circle:
                if math.sqrt(math.fabs(event.x - c.x) * math.fabs(event.x - c.x) + math.fabs(event.y - c.y) * math.fabs(
                                event.y - c.y)) < (30):
                    self.canvas.itemconfig(c.num, outline="blue")
                    c.start = 0
        if self.state == 'line':
            self.DoubleNodeList.clear()
            # 如果找不到tag为line的组件，则生成tag为line的组件
            startCircle = []
            endCircle = []
            for c1 in self.Circle:
                if c1.start == 1:
                    startCircle.append(c1)
                elif c1.start==0:
                    endCircle.append(c1)
            for c3 in startCircle:
                for c4 in endCircle:
                    a=c3.x-c4.x
                    b=c3.y-c4.y
                    self.canvas.create_line(c3.x + 30, c3.y, c4.x - 30, c4.y,
                                                                        fill=self.color,
                                                                        tags='line',
                                                                        width=2,
                                                                        arrow="last",
                                                                        arrowshape='8 10 3')
                    # if a<0 and b<0:
                    #     self.canvas.create_line(c3.x+20, c3.y+20, c4.x-20, c4.y-20,
                    #                             fill=self.color,
                    #                             tags='line',
                    #                             width=2,
                    #                             arrow="last",
                    #                             arrowshape='8 10 3')
                    # elif a<0 and b>0:
                    #     self.canvas.create_line(c3.x + 20, c3.y - 20, c4.x - 20, c4.y + 20,
                    #                             fill=self.color,
                    #                             tags='line',
                    #                             width=2,
                    #                             arrow="last",
                    #                             arrowshape='8 10 3')
                    #
                    # elif a>0 and b>0:
                    #     self.canvas.create_line(c3.x - 20, c3.y - 20, c4.x + 20, c4.y + 20,
                    #                             fill=self.color,
                    #                             tags='line',
                    #                             width=2,
                    #                             arrow="last",
                    #                             arrowshape='8 10 3')
                    # elif a>0 and b<0:
                    #     self.canvas.create_line(c3.x - 20, c3.y + 20, c4.x + 20, c4.y - 20,
                    #                                 fill=self.color,
                    #                                 tags='line',
                    #                                 width=2,
                    #                                 arrow="last",
                    #                                 arrowshape='8 10 3')
            #开始节点列表
            startNodeList=[]
            #结束节点列表
            endNodeList=[]
            #获取
            for start in startCircle:
                startNode = self.NodeCollection.get_Node(start.CircleNum)
                startNodeList.append(startNode)
            for end in endCircle:
                endNode=self.NodeCollection.get_Node(end.CircleNum)
                endNodeList.append(endNode)

            for start in startCircle:
                start_node=self.NodeCollection.get_Node(start.CircleNum)
                start_node.add_UpNodeS(endNodeList)

            for end in endCircle:
                end_node=self.NodeCollection.get_Node(end.CircleNum)
                end_node.add_DownNodeS(startNodeList)

            # for node in self.NodeCollection.node_list():
            #     print("节点序号:",node.node_index)
            #
            #     print("下游节点:")
            #     for downstream in node.get_downstream():
            #         print(downstream.node_index)
            #
            #     print("上游节点:")
            #     for upstream in node.get_upstream():
            #         print(upstream.node_index)

            for c in self.Circle:
                c.start=-1
                self.canvas.itemconfig(c.num, outline="black")


            E=self.get_E_V()
            self.DoubleNodeList.extend(E)
            self.ConnectionS.clear()

            print(self.DoubleNodeList.__len__())
            for node1,node2 in self.DoubleNodeList:
                ##创建线并添加进集合
                con=Connection(node2,node1)
                self.ConnectionS.append(con)
                print("(", node1.node_index, node2.node_index, ")")

            for conn in self.ConnectionS:
                conn.downstream_node.append_upstream(conn)
                conn.upstream_node.append_downstream(conn)

            print(self.ConnectionS.__len__())
            # for e1,e2 in E:
            #     print("(",e1.node_index,e2.node_index,")")
        if self.state=="adjust":
            print("adjust")
            self.x = event.x
            self.y = event.y

            for no in self.CNNNodeCon:
                if math.sqrt(math.fabs(event.x - no.x) * math.fabs(event.x - no.x) + math.fabs(event.y - no.y) * math.fabs(
                            event.y - no.y)) < (30):
                    self.AdjustNode(no)
                    break
            for c in self.CircleLineS:
                if math.sqrt(math.fabs(event.x-c.x)*math.fabs(event.x-c.x)+math.fabs(event.y-c.y)*math.fabs(event.y-c.y))<(30):
                    self.AdjustParam(c)
                    break

            pass
    def AdjustNode(self,no):
        root1=Tk()
        def get():
            no.node.activator=comboxlist.get()
            no.node.late_init(no.node.activator)
            print(no.node.activator)
            root1.destroy()
            root1.quit()
            pass
        comvalue = StringVar()
        comboxlist = ttk.Combobox(root1, textvariable=comvalue)
        comboxlist["values"] = ["ReLu", "Tanh","Sigmoid","Linear","Softmax"]
        comboxlist.current(0)
        button=Button(root1,text="确定",command=get)
        comboxlist.pack()
        button.pack()
        root1.mainloop()
    def AdjustParam(self,c):
        print(c.kind)
        def getParam():
            c.conn.filter_width=int(channel1.get())
            c.conn.filter_height=int(channel2.get())
            c.conn.filter_depth=int(channel3.get())
            c.conn.filter_num=int(channel4.get())

            c.conn.stride=int(channel5.get())
            c.conn.late_init(c.conn.filter_width,c.conn.filter_height,c.conn.filter_depth,c.conn.filter_num,c.conn.stride)
            root1.destroy()
            root1.quit()
            pass
        if c.kind=="conv":
            # 5
            print("conv")
            root1 = Tk()
            frame = Frame(root1, width=300, height=400, bg='pink')
            channel1 = Entry(frame)
            channel1.pack()
            channel2= Entry(frame)
            channel2.pack()
            channel3 = Entry(frame)
            channel3.pack()
            channel4 = Entry(frame)
            channel4.pack()
            channel5 = Entry(frame)
            channel5.pack()
            button=Button(frame,text="确定",command=getParam)
            button.pack()
            frame.pack()
            root1.mainloop()
            pass
        def getParam1():
            c.conn.filter_height=int(channel11.get())
            c.conn.filter_width=int(channel21.get())
            c.conn.stride=int(channel31.get())
            root1.destroy()
            root1.quit()
            pass
        def getParam2():
            c.conn.up_node_num=int(channel111.get())
            c.conn.down_node_num=int(channel112.get())
            c.conn.late_init(c.conn.up_node_num, c.conn.down_node_num)
            root1.destroy()
            root1.quit()
            pass
        if c.kind=="equal":
            # 3
            root1 = Tk()
            frame = Frame(root1, width=300, height=400, bg='pink')
            channel11 = Entry(frame)
            channel11.pack()
            channel21 = Entry(frame)
            channel21.pack()
            channel31 = Entry(frame)
            channel31.pack()
            button = Button(frame, text="确定", command=getParam1)
            button.pack()
            frame.pack()
            root1.mainloop()
            pass

        if c.kind == "dense":
            # 2
            root1 = Tk()
            frame = Frame(root1, width=300, height=400, bg='pink')
            channel111 = Entry(frame)
            channel111.pack()
            channel112 = Entry(frame)
            channel112.pack()
            button = Button(frame, text="确定", command=getParam2)
            button.pack()
            frame.pack()
            root1.mainloop()
            pass
        pass
    def setStates(self,state):
        self.state=state
    def drawPaddingNode(self,x,y):
        num1 = self.canvas.create_oval(x - 30, y - 30, x + 30, y + 30,
                                      outline=self.color, tags='oval', width=2)
        num = self.canvas.create_oval(x - 25, y - 25, x + 25, y + 25,
                                      outline=self.color, tags='oval', width=2)
        # self.Circle.append(Circle(x, y, num1, self.CircleNum))
        self.canvas.create_oval(x - 9 - 7, y - 9 - 7, x - 9 + 7, y - 9 + 7,
                                outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x + 9 - 7, y + 9 - 7, x + 9 + 7, y + 9 + 7,
                                outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x - 9 - 7, y + 9 - 7, x - 9 + 7, y + 9 + 7,
                                outline=self.color, tags='oval', width=2)
        self.canvas.create_oval(x + 9 - 7, y - 9 - 7, x + 9 + 7, y - 9 + 7,
                                outline=self.color, tags='oval', width=2)

        self.CircleNum += 1
        # self.canvas.create_text(x, y, text=self.CircleNum,tags="text")
        self.CNNNodeS.append(Circle(x, y, num, self.CircleNum))
        self.Circle.append(Circle(x, y, num, self.CircleNum))
        self.CNNNodeCon.append(CNNNode(x, y, Node()))
    def get_E_V(self):
        V = []
        E = []
        for node in self.NodeCollection.node_list():
            V.append(node)
            for up in node.get_UpNodeS():
                E.append((node, up))
            # for down in node.get_downstream():
            #     E.append((down, node))
        # print("__len__:",V.__len__())
        return E
    def drawLine(self,kind):
        # self.DoubleNodeList.clear()
        # 如果找不到tag为line的组件，则生成tag为line的组件
        startCirc=None
        endCirc=None
        # print(len(self.Circle))
        for start in self.Circle:
            if start.start == 1:
                startCirc = start

        for end in self.Circle:
            if end.start == 0:
                endCirc = end


        for startNo in self.CNNNodeCon:
            if startNo.x == startCirc.x and startNo.y == startCirc.y:
                startN = startNo.node
                pass

        for endNo in self.CNNNodeCon:
            if endNo.x == endCirc.x and endNo.y == endCirc.y:
                endN = endNo.node
                pass
        startCircle=[]
        endCircle=[]
        for c1 in self.Circle:
            if c1.start == 1:
                startCircle.append(c1)
            elif c1.start == 0:
                endCircle.append(c1)
        for c3 in startCircle:
            for c4 in endCircle:
                avx = (c3.x + 30 + c4.x - 30) / 2
                avy = (c3.y + c4.y) / 2
                if kind=="conv":
                    self.canvas.create_line(c3.x + 30, c3.y, c4.x - 30, c4.y,
                                            fill=self.color,
                                            tags='line',
                                            width=2,
                                            arrow="last",
                                            arrowshape='8 10 3')
                    self.canvas.create_image(avx, avy, image=self.im, tags="image")

                    filter=Filter()
                    self.CNNNodeCollection.append(filter)
                    cirline=CircleLine(avx,avy,filter,"conv")
                    self.CircleLineS.append(cirline)
                    startN.upstream=[filter]
                    endN.downstream=[filter]
                    filter.upstream_node=endN
                    filter.downstream_node=startN

                if kind=="equal":
                    self.canvas.create_line(c3.x + 30, c3.y, c4.x - 30, c4.y,
                                            fill=self.color,
                                            tags='line',
                                            width=2,
                                            arrow="last",
                                            arrowshape='8 10 3')
                    # self.CNNNodeCollection.append(PoolingConnection())

                    pool = PoolingConnection()
                    self.CNNNodeCollection.append(pool)
                    cirline = CircleLine(avx, avy, pool, "equal")
                    self.CircleLineS.append(cirline)

                    startN.upstream= [pool]
                    endN.downstream = [pool]
                    pool.upstream_node =  endN
                    pool.downstream_node =startN

                if kind=="vector":
                    self.canvas.create_line(c3.x + 30, c3.y, c4.x - 30, c4.y,
                                            fill=self.color,
                                            tags='line',
                                            width=2,
                                            arrow="last",
                                            arrowshape='8 10 3')
                    self.canvas.create_image(avx, avy, image=self.triangle, tags="image")
                    # self.CNNNodeCollection.append(ReshapeConnection())
                    reshape = ReshapeConnection()
                    self.CNNNodeCollection.append(reshape)
                    cirline = CircleLine(avx, avy, reshape, "vector")
                    self.CircleLineS.append(cirline)

                    startN. upstream= [reshape]
                    endN.downstream = [reshape]
                    reshape.upstream_node =  endN
                    reshape.downstream_node = startN

                if kind == "dense":
                    self.canvas.create_line(c3.x + 30, c3.y, c4.x - 30, c4.y,
                                            fill=self.color,
                                            tags='line',
                                            width=2,
                                            arrow="last",
                                            arrowshape='8 10 3')
                    self.canvas.create_image(avx, avy, image=self.dense, tags="image")
                    # self.CNNNodeCollection.append(TensorConnection())
                    Tensor = TensorConnection()
                    self.CNNNodeCollection.append(Tensor)
                    cirline = CircleLine(avx, avy, Tensor, "dense")
                    self.CircleLineS.append(cirline)

                    startN.upstream = [Tensor]
                    endN.downstream = [Tensor]
                    Tensor.upstream_node =  endN
                    Tensor.downstream_node = startN
                if kind=="contact":
                    self.canvas.create_line(c3.x + 30, c3.y, c4.x - 30, c4.y,
                                            fill=self.color,
                                            tags='line',
                                            width=2,
                                            arrow="last",
                                            arrowshape='8 10 3')
                    self.canvas.create_image(avx, avy, image=self.contact, tags="image")
                    # self.CNNNodeCollection.append(ConcatConnection())
                    Concat = ConcatConnection()
                    self.CNNNodeCollection.append(Concat)
                    cirline = CircleLine(avx, avy, Concat, "contact")
                    self.CircleLineS.append(cirline)

                    startN.upstream = [Concat]
                    endN.downstream = [Concat]
                    Concat.upstream_node = endN
                    Concat.downstream_node =startN

        for c in self.Circle:
            c.start = -1
            self.canvas.itemconfig(c.num, outline="black")

#定义一个按钮类
class MyButton:
    def __init__(self,root,label,canvas=None,canvasFrame=None,num=None):
        self.root=root
        self.label=label
        self.canvas=canvas
        self.canvasFrame=canvasFrame
        self.num=num
        if label == '箭头':
            self.button=Button(root,text=label,command=self.draw_line)
        elif label == '圆形':
            self.button=Button(root,text=label,command=self.draw_oval)
        elif label == '画笔颜色':
            self.button=Button(root,text=label,command=self.chooseColor)
        elif label == '箭头起始圆':
            self.button=Button(root,text=label,command=self.select_StartCircle)
        elif label == '箭头终止圆':
            self.button=Button(root,text=label,command=self.select_EndCircle)
        elif label == '+':
            self.button=Button(root,text=label,command=self.add_circle)
        elif label == '-':
            self.button=Button(root,text=label,command=self.delete_circle)
        elif label=="执行训练":
            self.button=Button(root,text=label,command=self.exe)
        elif label=="一维胶囊":
            self.button=Button(root,text=label,command=self.one_dim)
        elif label=="二维胶囊":
            self.button=Button(root,text=label,command=self.two_dim)
        elif label=="连接胶囊":
            self.button=Button(root,text=label,command=self.three_dim)
        elif label=="池化胶囊":
            self.button=Button(root,text=label,command=self.four_dim)
        elif label=="padding胶囊":
            self.button=Button(root,text=label,command=self.five_dim)
        elif label=="卷积连接":
            self.button=Button(root,text=label,command=self.conv)
        elif label=="恒等传输":
            self.button=Button(root,text=label,command=self.equal)
        elif label=="Reshape连接":
            self.button=Button(root,text=label,command=self.vector)
        elif label=="全连接":
            self.button=Button(root,text=label,command=self.dense)
        elif label==" 拼接 ":
            self.button=Button(root,text=label,command=self.contact)
        elif label=="CNN训练":
            self.button=Button(root,text=label,command=self.train)
        elif label=="调整参数":
            self.button=Button(root,text=label,command=self.correct)
        self.button.pack()

    def correct(self):
        self.canvasFrame.config({"cursor": "pencil"})
        self.canvas.setStates("adjust")
        pass

    def train(self):
        V = []
        E = []
        connection = []
        for c in self.canvas.CircleLineS:
            E.append(c.conn)
            print(c.conn.upstream_node)
            print(c.conn.downstream_node)
            connection.append((c.conn.downstream_node, c.conn.upstream_node))

        for no in self.canvas.CNNNodeCon:
            V.append(no.node)
            print(no.node.upstream)
            print(no.node.downstream)
        nodeConn = NodeCollection()
        nodeConn.node_collection = V

        net = cnn.Network(nodeConn, connection, "Classification", 5)
        net.train()
    def contact(self):
        self.canvasFrame.config({"cursor": "cross"})
        self.canvas.setStates("contact")
        pass

    # def adjust(self):
    #     self.canvasFrame.config({"cursor": "cross"})
    #     self.canvas.setStates("adjust")
    #     pass

    def five_dim(self):
        self.canvasFrame.config({"cursor": "circle"})
        self.canvas.setStates("five_dim")
    def one_dim(self):
        self.canvasFrame.config({"cursor": "circle"})
        self.canvas.setStates("one_dim")
    def two_dim(self):
        self.canvasFrame.config({"cursor": "circle"})
        self.canvas.setStates("two_dim")
    def three_dim(self):
        self.canvasFrame.config({"cursor": "circle"})
        self.canvas.setStates("three_dim")
    def four_dim(self):
        self.canvasFrame.config({"cursor": "circle"})
        self.canvas.setStates("four_dim")

    def conv(self):
        self.canvasFrame.config({"cursor": "cross"})
        self.canvas.setStates("conv")
    def equal(self):
        self.canvasFrame.config({"cursor": "cross"})
        self.canvas.setStates("equal")
    def vector(self):
        self.canvasFrame.config({"cursor": "cross"})
        self.canvas.setStates("vector")
    def dense(self):
        self.canvasFrame.config({"cursor": "cross"})
        self.canvas.setStates("dense")

    def exe(self):
        net = Network(self.canvas.layers, 1e-2, 100)
        data,label = generator_point.Generator_Point().simple_two_class()
        net.train(data, label)

    def add_circle(self):
        self.canvas.setStates("add_circle")
        i=int((self.num - 1) / 2)
        self.canvas.add_circle(i,115+i*150)
        pass

    def delete_circle(self):
        self.canvas.setStates("delete_circle")
        pass

    def draw_line(self):
        self.canvasFrame.config({"cursor": "cross"})
        self.canvas.setStates("line")
    def draw_oval(self):
        # canvas1.CircleNum = 0
        # for item in canvas1.CircleS:
        #     item.clear()
        # for item in canvas1.NodeCollectionS:
        #     item.node_collection.clear()
        # self.canvas.CircleNum = 0
        # self.canvas.CNNList = CNNList
        self.canvas.clearAll()
        self.canvasFrame.config({"cursor": "circle"})
        self.canvas.setStates("oval")

    def select_StartCircle(self):
        self.canvasFrame.config({"cursor": "dot"})
        self.canvas.setStates("StartCircle")

    def select_EndCircle(self):
        self.canvasFrame.config({"cursor": "target"})
        self.canvas.setStates("EndCircle")

    def chooseColor(self):
        r=colorchooser.askcolor()
        self.canvas.color=r[1]



def FileOpen():
    tempfile=filedialog.askopenfilename(title='请选择相应的文件',
    filetypes=[('图片','*.jpg *.png'),('All files','*')],
    initialdir='e:/')
    global canvas1
    canvas1.fileInput(tempfile)
def FileSave():
    filedialog.asksaveasfilename(title='保存' ,initialdir='e:/')


#菜单绑定函数
#打开文件对话框
"""格式：tkFileDialog.askopenfilename(title,filetypes,initialdir,initialfile)
    其中参数为：title:指定对话框标题          filetypes：指定要选择文件的类型  
           initialdir:指定的默认目录    initialfile：指定的默认文件
    函数的返回类型则是该文件的路径名。比如：H:/3.jpg
"""


#保存文件对话框
"""函数名：tkFileDialog.asksaveasfilename(title,filetypes,initialdir,initialfile)
    参数如上函数参数
    函数返回类型为：保存文件路径
"""
layers=[]

CNNList=[]
def CNN():
    root1 = Tk()
    i=1
    root1.geometry('600x800')
    Layers=[]
    def addCNN():
        nonlocal i
        i=i+1
        Layer=[]
        kernelText=Label(root1,text="卷积核大小:")
        kernelText.place(x=30,y=30*i)
        kernelSize=Entry(root1)
        kernelSize.place(x=100,y=30*i,width=30)

        Layer.append(kernelSize)

        paddingText = Label(root1, text="padding:")
        paddingText.place(x=140, y=30 * i)
        paddingSize = Entry(root1)
        paddingSize.place(x=200, y=30 * i, width=30)

        Layer.append(paddingSize)

        strideText = Label(root1, text="步长:")
        strideText.place(x=240, y=30 * i)
        strideSize = Entry(root1)
        strideSize.place(x=280, y=30 * i, width=30)

        Layer.append(strideSize)

        channelText = Label(root1, text="通道数:")
        channelText.place(x=300, y=30 * i)
        channelSize = Entry(root1)
        channelSize.place(x=350, y=30 * i, width=30)

        Layer.append(channelSize)

        Layers.append({"conv":Layer})

    def addPool():
        nonlocal i
        i = i + 1
        Layer = []
        poolText = Label(root1, text="池化大小:")
        poolText.place(x=30, y=30 * i)
        poolSize = Entry(root1)
        poolSize.place(x=100, y=30 * i, width=30)

        Layer.append(poolSize)

        strideText = Label(root1, text="padding:")
        strideText.place(x=140, y=30 * i)
        strideSize = Entry(root1)
        strideSize.place(x=200, y=30 * i, width=30)

        Layer.append(strideSize)

        poolTypeText=Label(root1, text="池化类型:")
        poolTypeText.place(x=240, y=30 * i)
        comvalue = StringVar()
        comboxlist = ttk.Combobox(root1, textvariable=comvalue)
        comboxlist["values"] = ["max","avg"]
        comboxlist.current(0)
        comboxlist.place(x=300, y=30 * i, width=50)
        Layer.append(comboxlist)

        Layers.append({"pool":Layer})

    def finishNetwork():
        nonlocal i
        i = i + 1
        print(layers)
        text = Label(root1, text="-----------------------"
        "-------------------------"
        "----------------------")
        text.place(x = 30,y = 30 * i )
        items=[]
        longitems=[]
        for item in Layers:
            if type(item).__name__ == 'dict':
                # Layers.remove(item)
                items.append(item)
        for item in items:
            Layers.remove(item)
        # Layers.remove(items)
        Layers.append(items)
    def contact():
        nonlocal i
        i = i + 1
        text = Label(root1, text="----------------------"
                                 "------- 网络已经拼接 ---"
                                 "----------------------")
        text.place(x=30,y= 30*i)
        items=[]
        for item in Layers:
            items.append(item)
        Layers.clear()
        Layers.append(items)
    def dense():
        nonlocal i
        i = i + 1
        text = Label(root1, text="----------------------"
                                 "---------- 全连接 ------"
                                 "----------------------")

        text.place(x=30, y=30 * i)
        Layers.append({"dense":1})
        pass
    addOne = Button(root1,text="添加卷积",command=addCNN)
    addTwo = Button(root1, text="添加池化",command=addPool)
    addThree = Button(root1, text="完成网络",command=finishNetwork)
    addFour = Button(root1,text="连接网络",command=contact)
    addFive = Button(root1, text=" 全连接 ", command=dense)

    addOne.place(x=0,y=0)
    addTwo.place(x=60,y=0)
    addThree.place(x=120,y=0)
    addFour.place(x=180,y=0)
    addFive.place(x=240,y=0)

    def getNetWork(NetWo):
        print(type(NetWo))
        layeritems = []
        for net in NetWo:
            key=list(net.keys())[0]
            print(key)
            items=net[key]
            if type(items).__name__ == 'list':
                keyitems = []
                for mi in items:
                    if not isinstance(mi, ttk.Combobox):
                        keyitems.append(int(mi.get()))
                    else:
                        keyitems.append(mi.get())
                    print(keyitems)
                layeritems.append({key: keyitems})
            else:
                layeritems.append(net)
        # print("layeritems",layeritems)
        return layeritems
    def Ensure():
        print(Layers)
        # for item in Layers:
        #     if type(item).__name__ == 'list' and type(item[0]).__name__ == 'list':
        #         # listitems=[]
        #         for it in item:
        #             for li in it:
        #                 layeritems = []
        #                 layeritem = []
        #                 for key in list(li.keys()):
        #                     items=li[key]
        #                     for mi in items:
        #                         if not isinstance(mi,ttk.Combobox):
        #                             layeritem.append(int(mi.get()))
        #                         else:
        #                             layeritem.append(mi.get())
        #                     layeritems.append({key:layeritem})
                    # listitems.append(layeritems)
        NiN = []
        for item in Layers:
            if type(item).__name__ == 'list' and type(item[0]).__name__ == 'list':
                for itemn in item:
                    layeritems=getNetWork(itemn)
                    NiN.append(layeritems)
                CNNList.append(NiN)
            if type(item).__name__ == 'list' and type(item[0]).__name__ != 'list':
                layeritems = getNetWork(item)
                CNNList.append(layeritems)
            if type(item).__name__ != 'list':
                CNNList.append(item)
        # print("CNNList",CNNList)
        # print("--------")
        root1.destroy()
        root1.quit()

    Ensure = Button(root1, text="  确定  ", command=Ensure)
    Ensure.place(x=300, y=0)
    root1.mainloop()
def anotherWindow():

    root1 = Tk()
    Listlayers=[]
    layers.clear()
    root1.geometry('300x400')
    frame = Frame(root1, width=300, height=400, bg='pink')
    def addList():

        if Listlayers.__len__()<6:
            comvalue = IntVar()
            comboxlist = ttk.Combobox(root1,textvariable=comvalue)
            comboxlist["values"] = [1, 2, 3, 4, 5, 6]
            comboxlist.current(0)
            Listlayers.append(comboxlist)
            comboxlist.pack()

    def getListlayers():
        for i in range(Listlayers.__len__()):
            layers.append(int(Listlayers[i].get()))
        root1.destroy()
        root1.quit()
    button = Button(frame, text="添加",height=1,command=addList)
    button.pack()
    button = Button(frame, text="确定", height=1,command=getListlayers)
    button.pack()
    frame.pack()
    root1.mainloop()


#生成容器
root=Tk()
#设置窗体大小
root.geometry('1200x1200')



"""该部分主要是关于菜单"""
#生成菜单栏
menubar=Menu(root)
#生成下拉菜单1   tearoff代表下拉菜单，0代表一级下拉菜单 ；1代表二级下拉菜单
submenu1=Menu(menubar,tearoff=0)
#往submenu1菜单中添加命令,command后面是响应函数,不需要()，否则界面生成就会执行该函数
submenu1.add_command(label='Open',command=FileOpen)
submenu1.add_command(label='Save',command=FileSave)
#submenu1.add_command(label='Close',command=FileClose)
#将submenu1菜单添加到menubar菜单栏中
menubar.add_cascade(label='File',menu=submenu1)
#将Menubar（菜单栏）添加到root容器中
root.config(menu=menubar)

"""该部分主要是窗口组件部分"""
#Frame组件
# frame1=Frame(root,width=200,height=600,bg='pink')

#pack_propagate(0)表示该frame不会随里面放的组件的大小而改变大小
# frame1.pack_propagate(0)
#pack()是将组件放在主容器上,side属性表示将组件要放的位置，有top,left,bottom,right
# frame1.pack(side='left')

commomIsShow=False
def commomToggle():
    global commomIsShow
    if commomIsShow:
        commonframe.place_forget()
        commomIsShow=False
        definiedButton.place(x=0, y=30)
        if definiedIsShow:
            definiedFrame.place(x=0, y=60)
            parameterButton.place(x=0, y=570)
            if parameterIsShow:
                parameterFrame.place(x=0,y=600)
                trainButton.place(x=0, y=690)
            else:
                trainButton.place(x=0, y=600)
        else:
            parameterButton.place(x=0, y=60)
            if parameterIsShow:
                parameterFrame.place(x=0, y=90)
                trainButton.place(x=0, y=180)
            else:
                trainButton.place(x=0, y=90)
    else:
        commonframe.place(x=0,y=30)
        definiedButton.place(x=0, y=60)
        if definiedIsShow:
            definiedFrame.place(x=0, y=90)
            parameterButton.place(x=0, y=600)
            if parameterIsShow:
                parameterFrame.place(x=0,y=630)
                trainButton.place(x=0, y=720)
            else:
                trainButton.place(x=0, y=630)

        else:
            parameterButton.place(x=0, y=90)
            if parameterIsShow:
                parameterFrame.place(x=0, y=120)
                trainButton.place(x=0, y=208)
            else:
                trainButton.place(x=0, y=120)
        commomIsShow = True

definiedIsShow=False
def definiedToggle():
    global definiedIsShow
    if definiedIsShow:
        definiedFrame.place_forget()
        definiedIsShow=False
        if commomIsShow:
            parameterButton.place(x=0, y=90)
            if parameterIsShow:
                parameterFrame.place(x=0,y=120)
                trainButton.place(x=0,y=210)
            else:
                trainButton.place(x=0, y=120)
        else:
            parameterButton.place(x=0, y=60)
            if parameterIsShow:
                parameterFrame.place(x=0, y=90)
                trainButton.place(x=0, y=180)
            else:
                trainButton.place(x=0, y=90)
    else:
        if commomIsShow:
            definiedFrame.place(x=0,y=90)
            parameterButton.place(x=0, y=600)
            if parameterIsShow:
                parameterFrame.place(x=0, y=630)
                trainButton.place(x=0, y=720)
            else:
                trainButton.place(x=0, y=630)
        else:
            definiedFrame.place(x=0, y=60)
            parameterButton.place(x=0, y=570)
            if parameterIsShow:
                parameterFrame.place(x=0, y=595)
                trainButton.place(x=0, y=685)
            else:
                trainButton.place(x=0, y=600)
        definiedIsShow=True

# def drawCNN():
#     # CNNList.clear()
#     # canvas1.Circle.clear()
#     canvas1.clearAll()
#     CNN()
#     print("CNNList:",CNNList)
#     canvas1.state=""
#
#     # canvas1.CircleNum=0
#     canvas1.CNNList=CNNList
#     canvas1.drawCNN()
#
#     # print("CNNList:",CNNList)
def CNNToggle():


    pass
MLPClick=False
def MLP():
    canvas1.setWidth_Height(1100, 700)
    # canvas1.CircleNum=0
    canvas1.clearAll()
    canvas1.state = ""
    # canvas1.Circle.clear()
    # for item in canvas1.CircleS:
    #     item.clear()
    # for item in canvas1.NodeCollectionS:
    #     item.node_collection.clear()
    # canvas1.Circle.clear()
    global MLPClick
    if not MLPClick:
        button3 = MyButton(frame2, "执行训练", canvas1, num=1)
        button3.button.pack()
        MLPClick=True
    # button3 = MyButton(frame2, "+",canvas1,num=1)
    # button3.button.place_configure(x=100, y=50)
    # button4 = MyButton(frame2, "-",canvas1,num=2)
    # button4.button.place(x=130, y=50)

    # button3 = MyButton(frame2, "+",canvas1,num=3)
    # button3.button.place_configure(x=250, y=50)
    # button4 = MyButton(frame2, "-",canvas1,num=4)
    # button4.button.place(x=280, y=50)
    #
    # button3 = MyButton(frame2, "+",canvas1,num=5)
    # button3.button.place_configure(x=400, y=50)
    # button4 = MyButton(frame2, "-",canvas1,num=6)
    # button4.button.place(x=430, y=50)
    #
    # button3 = MyButton(frame2, "+",canvas1,num=7)
    # button3.button.place_configure(x=550, y=50)
    # button4 = MyButton(frame2, "-",canvas1,num=8)
    # button4.button.place(x=580, y=50)
    #
    # button3 = MyButton(frame2, "+",canvas1,num=9)
    # button3.button.place_configure(x=700, y=50)
    # button4 = MyButton(frame2, "-",canvas1,num=10)
    # button4.button.place(x=730, y=50)
    #
    # button3 = MyButton(frame2,"+",canvas1,num=11)
    # button3.button.place_configure(x=850, y=50)
    # button4 = MyButton(frame2,"-",canvas1,num=12)
    # button4.button.place(x=880, y=50)

    anotherWindow()
    print(layers)
    canvas1.layers=layers
    canvas1.DenseInit(layers)

# def drawSVM():
#
#     pass

commonframe=Frame(root,width=60,height=30,bg='red')
commonButton=Button(root,text="常规模型",command=commomToggle)
commonButton.place(x=0,y=0)
commonframe.place(x=0,y=30)
MLPButton=Button(commonframe,text="MLP",command=MLP)
MLPButton.place(x=12,y=0)
# CNNButton=Button(commonframe,text="CNN",command=CNNToggle)
# CNNButton.place(x=12,y=30)
# SVMButton=Button(commonframe,text="LIBSVM",command=drawSVM)
# SVMButton.place(x=12,y=60)


definiedFrame=Frame(root,width=200,height=500,bg='green')
definiedButton=Button(root,text="定义模型",command=definiedToggle)
definiedButton.place(x=0,y=30)
definiedFrame.place(x=0,y=90)

# frame1.place(x=0,y=200)
frame2=Frame(root,width=1100,height=800)
frame2.place(x=200,y=0)
canvas1=MyCanvas(frame2)

button1=MyButton(definiedFrame,'箭头',canvas1,frame2)
button2=MyButton(definiedFrame,'圆形',canvas1,frame2)
button2=MyButton(definiedFrame,'箭头起始圆',canvas1,frame2)
button2=MyButton(definiedFrame,'箭头终止圆',canvas1,frame2)
button2=MyButton(definiedFrame,'画笔颜色',canvas1,frame2)
parameterIsShow=False


def parameterToggle():
    global parameterIsShow
    if parameterIsShow:
        parameterFrame.place_forget()
        parameterIsShow=False
        if commomIsShow:
            if definiedIsShow:
                trainButton.place(x=0,y=630)
            else:
                trainButton.place(x=0,y=120)
        else:
            if definiedIsShow:
                trainButton.place(x=0,y=600)
            else:
                trainButton.place(x=0,y=90)
    else:
        if commomIsShow:
            if definiedIsShow:
                parameterFrame.place(x=0,y=630)
                trainButton.place(x=0, y=730)
            else:
                parameterFrame.place(x=0,y=120)
                trainButton.place(x=0, y=210)
        else:
            if definiedIsShow:
                parameterFrame.place(x=0,y=600)
                trainButton.place(x=0, y=690)
            else:
                parameterFrame.place(x=0,y=90)
                trainButton.place(x=0, y=178)
        parameterIsShow=True


parameterButton=Button(root,text="求解配置",command=parameterToggle)
parameterButton.place(x=0,y=60)
parameterFrame=Frame(root,width=200,height=200,bg="blue")
parameterFrame.place(x=0,y=270)


label1=Label(parameterFrame,text = "学习率")
label1.pack()
Learning_rateList=MyComboxList(parameterFrame,[0.00001,0.0001,0.001,
                                       0.003,0.01,0.03,0.1,0.3,1,3,10])


# label2=Label(parameterFrame,text = "激活函数")
# label2.pack()
# ActivationList=MyComboxList(parameterFrame,["ReLu","Tanh","Sigmoid","Linear"])


label3=Label(parameterFrame,text = "问题类型")
label3.pack()
ProblemtypeList=MyComboxList(parameterFrame,["Classification","Regression"])

def start_train():
    net=Network(canvas1.NodeCollection,
                canvas1.DoubleNodeList,
                canvas1.ConnectionS,
                ActivationList.getCurrentValue(),
                ProblemtypeList.getCurrentValue(),
                5,
                Learning_rateList.getCurrentValue()
                )
    net.train()

commonframe.place_forget()
definiedFrame.place_forget()
parameterFrame.place_forget()
TWOClick=False
# CNNFrame=Frame(root,width=200,height=600,bg='pink')

# CNNButton=Button(root,text="CNN",command=CNNToggle)
# CNNButton.place(x=0,y=90)

button1=MyButton(definiedFrame,'一维胶囊',canvas1,frame2)
two_dense=MyButton(definiedFrame,'二维胶囊',canvas1,frame2)

button2=MyButton(definiedFrame,'连接胶囊',canvas1,frame2)
button2=MyButton(definiedFrame,'池化胶囊',canvas1,frame2)
button2=MyButton(definiedFrame,'padding胶囊',canvas1,frame2)


button2=MyButton(definiedFrame,'卷积连接',canvas1,frame2)
button2=MyButton(definiedFrame,'恒等传输',canvas1,frame2)
button2=MyButton(definiedFrame,'Reshape连接',canvas1,frame2)
button2=MyButton(definiedFrame,'全连接',canvas1,frame2)
button2=MyButton(definiedFrame,' 拼接 ',canvas1,frame2)
button2=MyButton(definiedFrame,'调整参数',canvas1,frame2)
button2=MyButton(definiedFrame,'CNN训练',canvas1,frame2)
# button2=MyButton(definiedFrame,'调整参数',canvas1,frame2)
# CNNFrame.place_forget()

trainButton=Button(root,text="执行训练",command=start_train)
trainButton.place(x=0,y=90)
root.mainloop()



