# Notes from Appendix A

## Basics

Source: Appendix A Introduction to PyTorch (Build LLM from Scratch - Sebastian Raschka)

![image.png](Notes%20from%20Appendix%20A%20179345517e4b8062b4f8f115bc6c7858/image.png)

Example that computes the gradient of a loss function w.r.t the weight and bias for a logstic regression model.

```python
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b 
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True) # store gradients after calculation.
grad_L_b = grad(loss, b, retain_graph=True)
```

An example neural net code in PyTorch

```python
class NeuralNetExample(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            torch.nn.Linear(30, 20),
            torch.nn.ReLU()

            torch.nn.Linear(20, num_outputs)
        )
    
    def forward(self, x):
        logits = self.layers(x)
        return logits

```

At inference time, the gradients are not required. Hence it may be handled using a `no_grad` context manager as shown:

```python
with torch.no_grad():
    out = model(X)
print(out)
```

## Data Loader

![image.png](Notes%20from%20Appendix%20A%20179345517e4b8062b4f8f115bc6c7858/image%201.png)

**Dataset class**: Describes the data records, defines the Xs and ys, how they are retrieved, etc.

```python
from torch.utils.data import Dataset

class ToyDataset(Dataset):
	def __init__(self, X, y):
		self.X = X
		self.y = y
		
	def getitem(self, index):
		one_X = self.X[index]
		one_y = self.y[index]
		
		return one_X, one_y
	
	def len(self):
		return self.X.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

```

**Dataloader class**: Defines the operations on the records defined using the *Dataset* object, how they are loaded, etc.

```python
from torch.utils.data import DataLoader

torch.manual.seed(42)

train_loader = DataLoader(
	dataset=train_ds,
	batch_size=2,
	shuffle=True,
	num_workers=0 # Parallelization of data load (and pre-processing)
)

test_loader = DataLoader(
	dataset=test_ds,
	batch_size=2,
	shuffle=False,
	num_workers=0 # `0` uses the main process for data loading.
)

for idx, X_sample, y_sample in enumerate(train_loader):
	print(f"Batch {idx+1}:", X_sample, y_sample)
```

The random number generator using `torch.manual_seed(42)`  should ensure the exact same shuffling order of training examples. However, if iterated over the dataset for a second time, the shuffling order will change. This prevents deep neural networks from getting caught in repetitive update cycles during training.

In practice, having a substantially smaller batch as the last batch in a training epoch can disturb the convergence during training. To prevent this, setting `drop_last=True` will drop the last batch in each epoch.

```python
train_loader = DataLoader(
	dataset=train_ds,
	batch_size=2,
	shuffle=True,
	num_workers=0, # `> 0` uses system resources efficiently, good for larger data.
	drop_last=True
)
# `num_workers` > 0 is not suitable for Jupyter notebooks.
```

## Training loop

```python
import torch.nn.functional as F

torch.manual_seed(42)

model = NeuralNetExample(num_inputs=2, num_outputs=2)
opt = torch.optimizer.SGD(model.parameters(), lr=0.5) # Pass params to optimize.

num_epochs=5

for epoch in range(num_epochs):
	model.train() # Enables settings for training.
	
	for idx, (features, labels) in enumerate(train_loader):
		logits = model(features)
		loss = F.cross_entropy(logits, labels)
		
		opt.zero_grad() # Resets gradients from previous iteration to 0.
		loss.backward() # Computes the gradients from the loss.
		opt.step() # Use the new gradients to update model parameters.
		
		print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")
              
  model.eval() # Enables settings for inference.

	with torch.no_grad():
		outputs = model(X_train)
	print(outputs)
	
	# To get probabilities of predictions.
	torch.set_printoptions(sci_mode=False)
	probas = torch.softmax(outputs, dim=1)
	print(probas)
```

## Saving and Loading a model

```python
torch.save(model.state_dict(), 'path/to/model_file.pt') # .pt or .pth are conventional extension for PyTorch models.

# model = NeuralNetExample(2, 2)
model.load_state_dict(torch.load('path/to/model_file.pt'))
```

## Running PyTorch operations on a GPU

```python
# Assign `cuda` as the device if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For Mac OS.
device = torch.device("mps" torch.backends.mps.is_available() else "cpu")

# Usage

t1 = tensor_1.to(device)
t2 = tensor_2.to(device)

t1 + t2 # Both tensors should be on the same device.

model.to(device) # Set device before running the training loop.
```

For parallelization using multi-GPU support (ideal for faster training), a DDP strategy (Distributed Data Panel) may be used. More information here: https://github.com/rasbt/LLMs-from-scratch

Alternative: Fabric library. Refer [https://mng.bz/jXle](https://mng.bz/jXle)

## Summary

- **PyTorch** is an open source library with three core components: a tensor library, automatic differentiation functions, and deep learning utilities.
- PyTorch’s tensor library is similar to array libraries like NumPy.
- In the context of PyTorch, tensors are array-like data structures representing *scalars*, *vectors*, *matrices*, and higher-dimensional arrays.
- PyTorch tensors can be executed on the CPU, but one major advantage of PyTorch’s tensor format is its GPU support to accelerate computations.
- The automatic differentiation (*autograd*) capabilities in PyTorch allow us to conveniently train neural networks using backpropagation without manually deriving gradients.
- The deep learning utilities in PyTorch provide building blocks for creating custom deep neural networks.
- PyTorch includes `Dataset` and `DataLoader` classes to set up efficient data-loading pipelines.
- It’s easiest to train models on a CPU or single GPU.
- Using `DistributedDataParallel` is the simplest way in PyTorch to accelerate the training if multiple GPUs are available.