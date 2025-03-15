import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import math

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import torch.nn.functional as F

USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')  
print_every = 100
print('using device:', device)  

'''
The project is split into two parts, first of which is an implementation
of the module of neural ODE, and second is the concrete implementation with
a neural ODE
'''


def ODE_solve(t0, t1, z0, func):
    
    '''
    Here we adopt the simplest Euler method to solve the differential equation,
    the design of this function here considers the batch processing and determins
    the steps.
    
    Our inputs, 
    t0, t1:[batch_size, 1]
    z0:[batch_size, features] (The original form of the data)
    func: The concrete function that can handle the derivative
    
    Our wanted outputs,
    the final result of z[batch_size, features] evolved from z0 from t0 to t1
    
    '''
    
    h_max = 0.05 # The largest interval of a step
    
    n_steps = math.ceil(abs(t1 - t0).max().item() / h_max) 
    # The n_steps depends on the largest interval of t0 and t1
    
    h = (t1 - t0) / n_steps
    t = t0
    z = z0
    
    for i in range(n_steps):
        z += h * func(z, t) #Euler step in
        t += h
        
    return z

class ODEF(nn.Module):
    '''
    The defined class ODEF here models the evolution i.e. the dynamic function.
    It doesn't require the init method, as it's a abstract base class that
    needs inheriting, the concrete implementation are to finish by its subclass
    '''
    def forward_with_grad(self, z, t, grad_outputs):
        
        '''
        z:[batch_size, features]
        t:[batch_size, 1]
        grad_outputs[batch_size, features] 
        (grad_outputs are calculated by partial L over partial z(t), and are in
        the same shape as z)
        '''
        
        batch_size = z.shape[0]
        
        out = self.forward(z, t) 
        # The forward function are to implement by subclass, 
        # ODEF simply focus on the interface
        
        a = grad_outputs
        
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a,),
            allow_unused=True, retain_graph=True
        )
        '''
        Let me briefly clarify the design here, out is actually retrived
        through the function dz/dt = f(z, t), which is modeled by the subclass.
        out is of the same shape as z. a is the upper stream of derivatives.
        the allow_unused guarantees that the stability when there is no explicit reliance
        between outputs and inputs
        the retain_graph promises that the computing graph can be invoked multiple times
        for backprapagate.        
        '''
        '''
        An issue here is that the grad method automatically sum along all the data points
        in a single batch, so we need to average them and give out, expand it to match batch size
        '''
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp if p_grad is not None]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size

        return out, adfdz, adfdt, adfdp
    
    def flatten_parameters(self):
        p_shapes = []
        flatten_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flatten_parameters.append(p.flatten())
        return torch.cat(flatten_parameters)
    
class neuralODEAdjoint(torch.autograd.Function):
    '''
    This class implements the forward pass and backward pass of
    the neural ODE using adjoint method
    '''
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        '''
        ctx represents the context, 
        z0:[batch_size, features]
        t[time_len],
        flat_parameters(the model parameters flattened)
        func(an instance of ODEF)
        
        Our wanted output, z of [time_len, batch_size, features]
        The solutions at every time point
        '''     
        
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)
        
        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ODE_solve(t[i_t], t[i_t + 1], z0, func)
                z[i_t + 1] = z0
        
        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z
    
    @staticmethod
    def backward(ctx, dLdz):
        '''
        This is where the ctx func and tensors from forward static method
        are retrieved by the backward method
        '''
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)
        
        def augmented_dynamics(aug_z_i, t_i):
            '''
            t_i, the time point now [bs, 1]
            aug_z_i, an augmented form of [bs, 2 * n_dim + n_params + 1]
            
            Wanted output, the derivative of the augmented matrix
            '''
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim: 2 * n_dim]
            
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                '''
                This is where we do a truncation, the t_i and
                z_i are truncated to prevent potential wrong calculation
                of grads
                '''
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)
                # ensure the gradients exist
                
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim)
            
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)
        
        dLdz = dLdz.view(time_len, bs, n_dim)
        
        with torch.no_grad():
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            adj_t = torch.zeros(bs, 1).to(dLdz)
            
            for i_t in range(time_len - 1, 0, -1):
                
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)
                
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]
                
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i
                
                #pack the augmented matrix [z, a, p, t]
                aug_z = torch.cat(z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t], dim = -1)
                
                aug_ans = ODE_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)
                
                #unpack the answer
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]  
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]  
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:] 
                
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None
                
                
class NeuralODE(nn.Module):
    '''
    Encapsulate the functions as a pytorch class
    '''
    def __init__(self, func):
        super().__init__()
        assert isinstance(func, ODEF)
        self.func = func
        
    def forward(self, z0, t=torch.Tensor([0., 1.]), return_whole_sequence=False):
        t = t.to(z0)
        z = neuralODEAdjoint.apply(z0, t, self.func.flatten_parameters, self.func)
        
        if return_whole_sequence:
            return z
        else:
            return z[-1]

class ConvODEF(ODEF):
    def __init__(self):
        super().__init__()
        # 使用简单的卷积网络计算状态变化率
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
    def forward(self, z, t):
        # z: [batch_size, 3, 32, 32]
        # t: [batch_size, 1]
        x = F.relu(self.conv1(z))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

NUM_TRAIN = 49000
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,   
                                transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64,
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                                transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64,
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
                                transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()
            
def flatten(x):
    N = x.shape[0] 
    return x.view(N, -1)
def transform(x):
    N = x.shape[0]
    return x.view(N, 32, 32, 32)
def concat(x1, x2, x3, x4):
    return torch.cat((x1, x2, x3, x4), 1)
class fcNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 32 * 3, 10)
    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

class iKunNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.Conv2d1 = nn.Conv2d(in_channels=3, out_channels=32, 
                                 kernel_size=1, stride=1, padding=0)
        self.Conv2d2 = nn.Conv2d(in_channels=3, out_channels=32,
                                 kernel_size=3, stride=1, padding = 1)
        self.Conv2d3 = nn.Conv2d(in_channels=3, out_channels=32,
                                 kernel_size=5, stride=1, padding=2)
        self.FClayer = nn.Linear(in_features=3*32*32, 
                                 out_features= 32*32*32)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.FCfinal_Layer = nn.Linear(in_features=32*32*32*4, out_features=num_classes)
    def forward(self, x):
        x1 = self.Conv2d1(x)
        x2 = self.Conv2d2(x)
        x3 = self.Conv2d3(x)
        x4 = self.FClayer(flatten(x))
        x1 = self.relu1(x1)
        x2 = self.relu2(x2)
        x3 = self.relu3(x3)
        x4 = self.relu4(x4)
        x4 = transform(x4)
        x = concat(x1, x2, x3, x4)
        x = self.FCfinal_Layer(flatten(x))
        return x


learning_rate = 1e-2 #waiting for fine-tuning

# yet to done for the KunNet
dynamic = ConvODEF()
model1 = NeuralODE(dynamic)
model2 = fcNet()
model = nn.Sequential(model1, model2)

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

anothermodel = iKunNet()
anotheroptim = optim.SGD(anothermodel.parameters(), lr=learning_rate)
newmodel = fcNet()
newoptimizer = optim.SGD(newmodel.parameters(), lr=learning_rate)
train_part34(model, optimizer)
train_part34(anothermodel, anotheroptim)
train_part34(newmodel, newoptimizer)