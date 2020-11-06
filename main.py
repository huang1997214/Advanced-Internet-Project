import torch
import networkx
from Model import GCN_Net
from Internet_Generation import Internet_Generation, build_false_batch
import random


feature_list, edge_list, node_type_list = Internet_Generation()
node_num = len(node_type_list)
model = GCN_Net([15, 15, 15])
edges_torch = torch.t(torch.LongTensor(edge_list))
edges_torch.cuda()
feature_torch = torch.FloatTensor(feature_list)
feature_torch.cuda()
split = int(0.9 * len(edge_list))
train_edge_index = edge_list[0:split]
test_edge_index = edge_list[split:len(edge_list)]
epoch = 100
lr = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                                         lr=lr)
for eps in range(epoch):
    random.shuffle(train_edge_index)
    truth_train_batch = train_edge_index[0:16]
    false_train_batch = build_false_batch(node_num, edge_list)
    train_batch = truth_train_batch + false_train_batch
    random.shuffle(train_batch)
    all_result_torch = None
    sum_loss = 0
    for batch in train_batch:
        output = model(feature_torch, edges_torch, batch[0], batch[1])
        if batch in truth_train_batch:
            label = [0]
        else:
            label = [1]
        label = torch.tensor(label).cuda()
        loss = criterion(output.unsqueeze(0), label)
        sum_loss += loss
    print(sum_loss)
    optimizer.zero_grad()
    sum_loss.backward()
    optimizer.step()
    test_batch = test_edge_index[0:16]
    prediction = []
    truth = []
    for batch in test_batch:
        output = model(feature_torch, edges_torch, batch[0], batch[1])
        if output[0] > output[1]:
            prediction.append(1)
        else:
            prediction.append(0)
        truth.append(1)
        acc = sum(prediction) / sum(truth)
    print(acc)

