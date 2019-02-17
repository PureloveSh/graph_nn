from tkinter import *
from tkinter import filedialog
from tkinter import colorchooser
import math
from node import Node
from NodeCollection import NodeCollection
from tkinter import ttk
import fc
import bp
from connection import Connection
#定义一个画布类
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
        self.NodeCollection=NodeCollection()
        self.DoubleNodeList=[]
        self.NodeCollectionS=[[],[],[],[],[],[]]
        self.CircleS = [[], [], [], [], [], []]
        self.ConnectionS=[]
        self.layers=[]
        for i in range(6):
            self.NodeCollectionS[i]=NodeCollection()
        #press函数表示当鼠标在画布上按下时初始化鼠标按下位置
    def DenseInit(self,layers):
        for i in range(layers.__len__()):
            x=115+i*150
            for j in range(layers[i]):
                self.add_circle(i,x)

    def add_circle(self,i,x):
        if self.NodeCollectionS[i].node_list().__len__()>=6:
            return
        j=self.NodeCollectionS[i].node_list().__len__()
        y=(100+j*90)
        num = self.canvas.create_oval(x-30,y-30,x+30,y+30,
        outline=self.color, tags='line' + str(self.CircleNum), width=2)
        # print(num)
        self.CircleS[i].append(Circle(x, y, num, self.CircleNum))
        # self.canvas.create_text(x, y, text=self.CircleNum, )
        node = Node(self.CircleNum)
        self.NodeCollectionS[i].add_node(node)
        self.CircleNum += 1
        self.canvas.delete('line')
        for n in range(6):
            if self.CircleS[n].__len__()>0:
                # print(self.CircleS[n].__len__())
                for j in range(n+1,6):
                    if self.CircleS[j].__len__()>0:
                        self.drawAllLine(self.CircleS[n],self.CircleS[j])
                        break




    def delete_circle(self):


        pass
    def drawAllLine(self,startCircle,endCircle):

        for c3 in startCircle:
            for c4 in endCircle:
                a = c3.x - c4.x
                b = c3.y - c4.y
                if b==0:
                    b=c3.y
                if a < 0 and b < 0:
                    self.canvas.create_line(c3.x + 20, c3.y + 20, c4.x - 20, c4.y - 20,
                                            fill=self.color,
                                            tags='line',
                                            width=2,
                                            arrow="last",
                                            arrowshape='8 10 3')
                elif a < 0 and b > 0:
                    self.canvas.create_line(c3.x + 20, c3.y - 20, c4.x - 20, c4.y + 20,
                                            fill=self.color,
                                            tags='line',
                                            width=2,
                                            arrow="last",
                                            arrowshape='8 10 3')

                elif a > 0 and b > 0:
                    self.canvas.create_line(c3.x - 20, c3.y - 20, c4.x + 20, c4.y + 20,
                                            fill=self.color,
                                            tags='line',
                                            width=2,
                                            arrow="last",
                                            arrowshape='8 10 3')
                elif a > 0 and b < 0:
                    self.canvas.create_line(c3.x - 20, c3.y + 20, c4.x + 20, c4.y - 20,
                                            fill=self.color,
                                            tags='line',
                                            width=2,
                                            arrow="last",
                                            arrowshape='8 10 3')
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
        if self.state == 'oval':
            num=self.canvas.create_oval(self.x-30, self.y-30, self.x+30, self.y+30,
                                        outline=self.color,tags='line'+str(self.CircleNum),width = 2)
            # print(num)
            self.Circle.append(Circle(self.x, self.y, num,self.CircleNum))
            self.canvas.create_text(self.x, self.y, text=self.CircleNum)
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
                    if a<0 and b<0:
                        self.canvas.create_line(c3.x+20, c3.y+20, c4.x-20, c4.y-20,
                                                fill=self.color,
                                                tags='line',
                                                width=2,
                                                arrow="last",
                                                arrowshape='8 10 3')
                    elif a<0 and b>0:
                        self.canvas.create_line(c3.x + 20, c3.y - 20, c4.x - 20, c4.y + 20,
                                                fill=self.color,
                                                tags='line',
                                                width=2,
                                                arrow="last",
                                                arrowshape='8 10 3')

                    elif a>0 and b>0:
                        self.canvas.create_line(c3.x - 20, c3.y - 20, c4.x + 20, c4.y + 20,
                                                fill=self.color,
                                                tags='line',
                                                width=2,
                                                arrow="last",
                                                arrowshape='8 10 3')
                    elif a>0 and b<0:
                        self.canvas.create_line(c3.x - 20, c3.y + 20, c4.x + 20, c4.y - 20,
                                                    fill=self.color,
                                                    tags='line',
                                                    width=2,
                                                    arrow="last",
                                                    arrowshape='8 10 3')
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
            # for f in V:
            #     print(f.node_index)
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
    def setStates(self,state):
        self.state=state

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



#定义一个按钮类
class MyButton:
    def __init__(self,root,label,canvas=None,canvasFrame=None,num=None):
        self.root=root
        self.label=label
        self.canvas=canvas
        self.canvasFrame=canvasFrame
        self.num=num
        if label == '直线':
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
        self.button.pack()

    def exe(self):
        net = fc.Network(self.canvas.layers, 1e-2, 10000)
        print(type(self.canvas.layers[0]))
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
def anotherWindow():

    root1 = Tk()
    Listlayers=[]

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
        print(layers)
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
root.geometry('1200x800')



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
frame1=Frame(root,width=100,height=800,bg='pink')
#pack_propagate(0)表示该frame不会随里面放的组件的大小而改变大小
frame1.pack_propagate(0)
#pack()是将组件放在主容器上,side属性表示将组件要放的位置，有top,left,bottom,right
frame1.pack(side='left')

frame2=Frame(root,width=1100,height=800)
frame2.pack_propagate(0)
frame2.pack()
canvas1=MyCanvas(frame2)


def MLP():
    canvas1.setWidth_Height(1100, 700)
    canvas1.CircleNum=0
    button3 = MyButton(frame2, "执行训练", canvas1, num=1)
    button3.button.pack()
    button3 = MyButton(frame2, "+",canvas1,num=1)
    button3.button.place_configure(x=100, y=50)
    button4 = MyButton(frame2, "-",canvas1,num=2)
    button4.button.place(x=130, y=50)

    button3 = MyButton(frame2, "+",canvas1,num=3)
    button3.button.place_configure(x=250, y=50)
    button4 = MyButton(frame2, "-",canvas1,num=4)
    button4.button.place(x=280, y=50)

    button3 = MyButton(frame2, "+",canvas1,num=5)
    button3.button.place_configure(x=400, y=50)
    button4 = MyButton(frame2, "-",canvas1,num=6)
    button4.button.place(x=430, y=50)

    button3 = MyButton(frame2, "+",canvas1,num=7)
    button3.button.place_configure(x=550, y=50)
    button4 = MyButton(frame2, "-",canvas1,num=8)
    button4.button.place(x=580, y=50)

    button3 = MyButton(frame2, "+",canvas1,num=9)
    button3.button.place_configure(x=700, y=50)
    button4 = MyButton(frame2, "-",canvas1,num=10)
    button4.button.place(x=730, y=50)

    button3 = MyButton(frame2,"+",canvas1,num=11)
    button3.button.place_configure(x=850, y=50)
    button4 = MyButton(frame2,"-",canvas1,num=12)
    button4.button.place(x=880, y=50)
    anotherWindow()
    canvas1.layers=layers
    canvas1.DenseInit(layers)



MLPButton=Button(frame1,text="MLP",command=MLP)
MLPButton.pack()

button1=MyButton(frame1,'直线',canvas1,frame2)
button2=MyButton(frame1,'圆形',canvas1,frame2)
button2=MyButton(frame1,'箭头起始圆',canvas1,frame2)
button2=MyButton(frame1,'箭头终止圆',canvas1,frame2)
button2=MyButton(frame1,'画笔颜色',canvas1,frame2)

label1=Label(frame1,text = "学习率")
label1.pack()
Learning_rateList=MyComboxList(frame1,[0.00001,0.0001,0.001,
                                       0.003,0.01,0.03,0.1,0.3,1,3,10])


label2=Label(frame1,text = "激活函数")
label2.pack()
ActivationList=MyComboxList(frame1,["ReLu","Tanh","Sigmoid","Linear"])


label3=Label(frame1,text = "问题类型")
label3.pack()
ProblemtypeList=MyComboxList(frame1,["Classification","Regression"])

def start_train():
    data, label = generator_point.Generator_Point().simple_two_class()
    net=bp.Network(canvas1.NodeCollection,
                canvas1.DoubleNodeList,
                ActivationList.getCurrentValue(),
                ProblemtypeList.getCurrentValue(),
                10000,
                Learning_rateList.getCurrentValue()
                )
    net.train(data, label)


trainButton=Button(frame1,text="执行训练",command=start_train)
trainButton.pack()
root.mainloop()




