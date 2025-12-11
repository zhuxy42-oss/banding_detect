from graphviz import Digraph

def visualize_gine_conv():
    dot = Digraph(comment='GINEConv Module')
    dot.attr(rankdir='TB')
    
    # 输入
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs')
        c.node('nodes', 'Node Features\n[N, node_feat_dim]', shape='box')
        c.node('edges', 'Edge Features\n[E, edge_feat_dim]', shape='box')
        c.node('indices', 'Edge Indices\n[2, E]', shape='box')
    
    # GINEConv 过程
    dot.node('msg', 'Message Passing\n(Edge-aware)', shape='ellipse')
    dot.node('agg', 'Aggregation\n(Sum over neighbors)', shape='ellipse')
    
    # MLP 组件
    with dot.subgraph(name='cluster_mlp') as c:
        c.attr(label='Node Update MLP')
        c.node('linear1', 'Linear\nnode_feat_dim → hidden_dim', shape='box')
        c.node('norm', 'LayerNorm\n(hidden_dim)', shape='box')
        c.node('act', 'Tanh Activation', shape='box')
        c.node('drop', 'Dropout\n(p=0.3)', shape='box')
        c.node('linear2', 'Linear\nhidden_dim → hidden_dim', shape='box')
    
    # 输出
    dot.node('output', 'Output Nodes\n[N, hidden_dim]', shape='box')
    
    # 连接
    dot.edges([
        ('nodes', 'msg'), ('edges', 'msg'), ('indices', 'msg'),
        ('msg', 'agg'),
        ('agg', 'linear1'),
        ('linear1', 'norm'), ('norm', 'act'), ('act', 'drop'), 
        ('drop', 'linear2'), ('linear2', 'output')
    ])
    
    return dot

# 生成图像
graph = visualize_gine_conv()
graph.render('gineconv_architecture', format='png', cleanup=True)