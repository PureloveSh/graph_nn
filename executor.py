class Executor:
    def __init__(self, v, e):
        self.v = v
        self.e = e

    def indegree0(self):
        if self.v == []:
            return None
        tmp = self.v[:]
        for i in self.e:
            if i[1] in tmp:
                tmp.remove(i[1])
        if tmp == []:
            return -1

        for t in tmp:
            for i in range(len(self.e)):
                if t in self.e[i]:
                    self.e[i] = ()  # 站位，之后删掉

        if self.e:
            eset = set(self.e)
            eset.remove(())
            self.e[:] = list(eset)

        if self.v:
            for t in tmp:
                self.v.remove(t)
        return tmp

    def topoSort(self):
        result = []
        while True:
            nodes = self.indegree0()
            if nodes is None:
                break
            if nodes == -1:
                print("there's a circle.")
                return None
            result.extend(nodes)
        return result


if __name__ == "__main__":
    # v = ['a', 'b', 'c', 'd', 'e']
    # e = [('a', 'b'), ('a', 'd'), ('b', 'c'), ('d', 'c'), ('d', 'e'), ('e', 'c')]

    v = [1, 2, 3, 4, 5, 6]
    e = [(1, 3), (1, 4), (2, 3), (2, 4), (2, 5), (1, 6), (3, 6), (4, 6)]
    res = Executor(v, e).topoSort()
    print(res)
