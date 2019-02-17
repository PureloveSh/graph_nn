import node


class NodeCollection:

    def __init__(self):
        self.node_collection = []

    def add_node(self, node):
        self.node_collection.append(node)

    def node_list(self):
        return self.node_collection

    def get_Node(self,node_index):
        for node in self.node_collection:
            if node.node_index == node_index:
                return node

    def conn_to_output(self):
        node_list = []
        for node in self.node_collection:
            if node.get_upstream() == []:
                node_list.append(node)
        return node_list

    def get_input_node(self):
        node_input_list = []
        for node in self.node_collection:
            if node.get_downstream() == []:
                node_input_list.append(node)
        return node_input_list


    def get_output(self):
        output = []
        for node in self.node_collection:
            if node.get_upstream() == []:
                output.append(node.get_output())
        return output