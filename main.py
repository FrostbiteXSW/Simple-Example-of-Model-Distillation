import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm


class Student(nn.Module):
	def __init__(self):
		super(Student, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(3, 32, 3, 2),
			nn.ReLU(),
			nn.Conv2d(32, 32, 3, 2),
			nn.ReLU(),
			nn.Conv2d(32, 32, 2, 1),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(2)
		)

		self.fc = nn.Linear(128, 10)

	def forward(self, x):
		out = self.conv(x)
		out = out.view(x.shape[0], -1)
		out = self.fc(out)
		return out


class Teacher(nn.Module):
	def __init__(self):
		super(Teacher, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(3, 128, 3, 2),
			nn.ReLU(),
			nn.Conv2d(128, 128, 3, 2),
			nn.ReLU(),
			nn.Conv2d(128, 128, 2, 1),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d(2)
		)

		self.fc = nn.Linear(512, 10)

	def forward(self, x):
		out = self.conv(x)
		out = out.view(x.shape[0], -1)
		out = self.fc(out)
		return out


def cross_entropy_loss(output, target):
	return -torch.sum(output.log() * target) / output.shape[0]


batch_size = 64
num_epochs = 5
temperature = 3
loss_lambda = 0.5

input_transform = transforms.Compose([
	transforms.Resize(28),
	transforms.Grayscale(3),
	transforms.ToTensor()
])

train_data = MNIST('./data', train=True, transform=input_transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = MNIST('./data', train=False, transform=input_transform, download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
len_train_data = len(train_data)
len_test_data = len(test_data)

# ------------------------------------------------------------------------------------------------------#

print('\n\nTraining student without teacher...')
student_without_teacher = Student().cuda()
optimizer = torch.optim.Adam(student_without_teacher.parameters())
criterion = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
	print(f'Epoch {epoch + 1}/{num_epochs}:')

	train_loss = .0
	train_acc = .0
	student_without_teacher.train()
	for batch_x, batch_y in tqdm(train_loader):
		batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

		out = torch.softmax(student_without_teacher(batch_x), dim=1)
		loss = criterion(out, batch_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss += loss.data.item()
		train_acc += (torch.max(out, 1)[1] == batch_y).sum().data.item()
	print('Train loss: {:.6f}, acc: {:.6f}'.format(train_loss / len_train_data, train_acc / len_train_data))

	eval_loss = .0
	eval_acc = .0
	student_without_teacher.eval()
	with torch.no_grad():
		for batch_x, batch_y in tqdm(test_loader):
			batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

			out = torch.softmax(student_without_teacher(batch_x), dim=1)
			loss = criterion(out, batch_y)

			eval_loss += loss.data.item()
			eval_acc += (torch.max(out, 1)[1] == batch_y).sum().data.item()
		print('Eval loss: {:.6f}, acc: {:.6f}'.format(eval_loss / len_test_data, eval_acc / len_test_data))

# ------------------------------------------------------------------------------------------------------#

print('\n\nTraining teacher...')
teacher = Teacher().cuda()
optimizer = torch.optim.Adam(teacher.parameters())
criterion = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
	print(f'Epoch {epoch + 1}/{num_epochs}:')

	train_loss = .0
	train_acc = .0
	teacher.train()
	for batch_x, batch_y in tqdm(train_loader):
		batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

		out = torch.softmax(teacher(batch_x), dim=1)
		loss = criterion(out, batch_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss += loss.data.item()
		train_acc += (torch.max(out, 1)[1] == batch_y).sum().data.item()
	print('Train loss: {:.6f}, acc: {:.6f}'.format(train_loss / len_train_data, train_acc / len_train_data))

	eval_loss = .0
	eval_acc = .0
	teacher.eval()
	with torch.no_grad():
		for batch_x, batch_y in tqdm(test_loader):
			batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

			out = torch.softmax(teacher(batch_x), dim=1)
			loss = criterion(out, batch_y)

			eval_loss += loss.data.item()
			eval_acc += (torch.max(out, 1)[1] == batch_y).sum().data.item()
		print('Eval loss: {:.6f}, acc: {:.6f}'.format(eval_loss / len_test_data, eval_acc / len_test_data))

# ------------------------------------------------------------------------------------------------------#

print('\n\nTraining student with teacher...')
student_with_teacher = Student().cuda()
optimizer = torch.optim.Adam(student_with_teacher.parameters())
criterion = [nn.CrossEntropyLoss(), cross_entropy_loss]
for epoch in range(num_epochs):
	print(f'Epoch {epoch + 1}/{num_epochs}:')

	train_loss = .0
	train_acc = .0
	student_with_teacher.train()
	for batch_x, batch_y in tqdm(train_loader):
		batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

		out = student_with_teacher(batch_x)
		batch_y_teacher = torch.softmax(teacher(batch_x) / temperature, dim=1).detach()
		out_result = torch.softmax(out, dim=1)
		loss = (1 - loss_lambda) * criterion[0](out_result, batch_y) + \
		       loss_lambda * temperature**2 * criterion[1](torch.softmax(out / temperature, dim=1), batch_y_teacher)
		assert not torch.isnan(loss)

		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(student_with_teacher.parameters(), 1)
		optimizer.step()

		train_loss += loss.data.item()
		train_acc += (torch.max(out_result, 1)[1] == batch_y).sum().data.item()
	print('Train loss: {:.6f}, acc: {:.6f}'.format(train_loss / len_train_data, train_acc / len_train_data))

	eval_loss = .0
	eval_acc = .0
	student_with_teacher.eval()
	with torch.no_grad():
		for batch_x, batch_y in tqdm(test_loader):
			batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

			out = student_with_teacher(batch_x)
			batch_y_teacher = torch.softmax(teacher(batch_x) / temperature, dim=1).detach()
			out_result = torch.softmax(out, dim=1)
			loss = (1 - loss_lambda) * criterion[0](out_result, batch_y) + \
			       loss_lambda * temperature ** 2 * criterion[1](torch.softmax(out / temperature, dim=1), batch_y_teacher)

			eval_loss += loss.data.item()
			eval_acc += (torch.max(out_result, 1)[1] == batch_y).sum().data.item()
		print('Eval loss: {:.6f}, acc: {:.6f}'.format(eval_loss / len_test_data, eval_acc / len_test_data))
