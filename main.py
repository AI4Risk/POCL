from models import *
from tool import *
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt

online_dataset = get_online_dataset()
contrast_dataset = get_contrast_dataset()
model_mas = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrain_model = ADDer(32, 32).to(device)
optimizer_pretrain = torch.optim.Adam(pretrain_model.parameters(), lr=0.003)
classifier = Classifier(32, 2).to(device)
train_optimizer = torch.optim.Adam([{'params': pretrain_model.parameters(), 'lr': 0.001},
                                    {'params': classifier.parameters(), 'lr': 0.003}], lr=0.01)


def pretrain(model, pretrain_date, epoch, optimizer):
    model.train()
    shuffled_list = list(range(pretrain_date))
    np.random.shuffle(shuffled_list)
    for j in range(epoch):
        loss_sum = 0
        for i in shuffled_list:
            optimizer.zero_grad()
            loss = model([contrast_dataset[i * 2 + 1][0].to(device),
                          contrast_dataset[i * 2][0].to(device)],
                         [contrast_dataset[i * 2 + 1][1].to(device),
                          contrast_dataset[i * 2][1].to(device)
                          ], None, True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print("Pretrain Epoch:", j, "Loss:", loss_sum)


def iterate_offline_model(x, xedge, y, optimizer):
    pretrain_model.train()
    classifier.train()
    optimizer.zero_grad()
    loss_pre, data_en = pretrain_model(x, xedge, y, False)
    loss_mas = 0
    out = classifier(data_en, xedge)
    if model_mas is not None:
        loss_mas = model_mas.penalty(pretrain_model.encoder, classifier)
    loss = F.cross_entropy(out, y.long()) + loss_pre * 0.1 + loss_mas * 10
    loss.backward()
    optimizer.step()
    return loss.item()


def train_offline_model(epoch, pretrain_date, optimizer):
    pretrain_data = []
    for j in range(epoch):
        loss_sum = 0
        for i in range(pretrain_date):
            data_now = torch.Tensor(online_dataset[i][0]).to(device)
            edge_now = torch.LongTensor(online_dataset[i][1]).long().to(device)
            y_now = torch.Tensor(online_dataset[i][2]).bool().to(device)
            pretrain_data.append([data_now, edge_now])
            loss = iterate_offline_model(data_now, edge_now, y_now, optimizer)
            loss_sum += loss
        print("Offline Epoch:", j, "Loss:", loss_sum)


pretrain_days = 14
pretrain(pretrain_model, pretrain_days, 300, optimizer_pretrain)
pretrain_data = train_offline_model(100, pretrain_days, train_optimizer)
model_mas = MAS(pretrain_model.encoder, classifier, pretrain_data, [])

num = 0
aucu = []
f1u = []
accu = []

print("Online Training")

for i in range(pretrain_days, len(online_dataset)):
    classifier.eval()
    pretrain_model.eval()
    data_now = torch.Tensor(online_dataset[i][0]).to(device)
    edge_now = torch.LongTensor(online_dataset[i][1]).long().to(device)
    y_now = torch.Tensor(online_dataset[i][2]).bool().to(device)

    data_en = pretrain_model.encoder(data_now, edge_now)
    out = classifier(data_en, edge_now)
    out = F.softmax(out, dim=1)
    _, pred = out.max(dim=1)
    correct = pred.eq(y_now).sum().item()
    acc = correct / (len(y_now))
    accu.append(acc)
    class_probs = out[:, 1].detach().cpu().numpy()
    auc = roc_auc_score(y_now.cpu(), class_probs)
    aucu.append(auc)

    preds_np = pred.cpu().detach().numpy()
    y_true_np = y_now.cpu().detach().numpy()

    f1 = f1_score(y_true_np, preds_np, labels=[1], average='binary')
    f1u.append(f1)

    if num % 2 == 1:
        for numbers in range(10):
            iterate_offline_model(data_now, edge_now, y_now, train_optimizer)
        classifier.eval()
        pretrain_model.eval()
        importance = model_mas.get_history_importance()
        model_mas = MAS(pretrain_model.encoder, classifier, [[data_now, edge_now]], importance)
    num += 1

print("Online AUC:", np.mean(aucu))
print("Online F1:", np.mean(f1u))
print("Online ACC:", np.mean(accu))

plt.plot(get_avg(accu), 'o-', label='POCL')
plt.xlabel('t')
plt.ylabel('acc')
plt.legend()
plt.savefig('acc.png')