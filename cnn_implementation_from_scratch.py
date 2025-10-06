'''
Usage Examples
Mini workflow (quick debug, random data):
python cnn_from_scratch.py --mode mini --model lenet --epochs 3

Full workflow (real CIFAR-10, subset for speed):
python cnn_from_scratch.py --mode full --model vgg --epochs 10 --batch_size 64 --lr 0.01

'''



import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.ndimage import rotate
import seaborn as sns

# ==============================
# Base Layer
# ==============================
class Layer:
    def __init__(self):
        self.trainable = True
        self.params = {}
        self.grads = {}
        self.cache = {}

    def forward(self, x, training=True):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError


# ==============================
# Layers
# ==============================
class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding="same"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding

        kh, kw = self.kernel_size
        scale = np.sqrt(2.0 / (in_channels * kh * kw))
        self.params["W"] = np.random.randn(out_channels, in_channels, kh, kw) * scale
        self.params["b"] = np.zeros((out_channels, 1))

    def _get_pad_width(self, input_shape):
        _, _, H, W = input_shape
        kh, kw = self.kernel_size
        if self.padding == "same":
            pad_h = max((np.ceil(H / self.stride) - 1) * self.stride + kh - H, 0)
            pad_w = max((np.ceil(W / self.stride) - 1) * self.stride + kw - W, 0)
            return int(pad_h // 2), int(pad_w // 2)
        elif self.padding == "valid":
            return 0, 0
        else:
            raise ValueError("Invalid padding")

    def _pad_input(self, x, pad_h, pad_w):
        return np.pad(x, ((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)), mode="constant")

    def forward(self, x, training=True):
        batch_size, _, H, W = x.shape
        kh, kw = self.kernel_size
        stride = self.stride
        pad_h, pad_w = self._get_pad_width(x.shape)

        x_padded = self._pad_input(x, pad_h, pad_w)
        H_out = (H + 2*pad_h - kh) // stride + 1
        W_out = (W + 2*pad_w - kw) // stride + 1
        out = np.zeros((batch_size, self.out_channels, H_out, W_out))

        W, b = self.params["W"], self.params["b"]

        for n in range(batch_size):
            for f in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start, w_start = i*stride, j*stride
                        patch = x_padded[n, :, h_start:h_start+kh, w_start:w_start+kw]
                        out[n,f,i,j] = np.sum(patch * W[f]) + float(b[f])

        self.cache = {"x": x, "x_padded": x_padded, "pad_h": pad_h, "pad_w": pad_w}
        return out

    def backward(self, grad_output):
        x, x_padded = self.cache["x"], self.cache["x_padded"]
        pad_h, pad_w = self.cache["pad_h"], self.cache["pad_w"]
        batch_size, _, H, W = x.shape
        kh, kw = self.kernel_size
        stride = self.stride
        _, _, H_out, W_out = grad_output.shape

        dW, db = np.zeros_like(self.params["W"]), np.zeros_like(self.params["b"])
        dx_padded = np.zeros_like(x_padded)

        for n in range(batch_size):
            for f in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start, w_start = i*stride, j*stride
                        patch = x_padded[n, :, h_start:h_start+kh, w_start:w_start+kw]
                        dW[f] += grad_output[n,f,i,j] * patch
                        db[f] += grad_output[n,f,i,j]
                        dx_padded[n,:,h_start:h_start+kh,w_start:w_start+kw] += grad_output[n,f,i,j] * self.params["W"][f]

        dx = dx_padded[:,:,pad_h:H+pad_h, pad_w:W+pad_w]
        self.grads["W"], self.grads["b"] = dW, db
        return dx


class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=None):
        super().__init__()
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size,int) else pool_size
        self.stride = stride or pool_size
        self.trainable = False

    def forward(self, x, training=True):
        N, C, H, W = x.shape
        ph, pw = self.pool_size
        stride = self.stride
        H_out, W_out = (H-ph)//stride+1, (W-pw)//stride+1
        out = np.zeros((N,C,H_out,W_out))
        self.cache["mask"] = {}

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h, w = i*stride, j*stride
                        region = x[n,c,h:h+ph,w:w+pw]
                        out[n,c,i,j] = np.max(region)
                        idx = np.unravel_index(np.argmax(region), region.shape)
                        self.cache["mask"][(n,c,i,j)] = (h+idx[0], w+idx[1])
        self.cache["input_shape"] = x.shape
        return out

    def backward(self, grad_output):
        dx = np.zeros(self.cache["input_shape"])
        for (n,c,i,j),(h,w) in self.cache["mask"].items():
            dx[n,c,h,w] += grad_output[n,c,i,j]
        return dx


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, x, training=True):
        self.cache["input_shape"] = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.cache["input_shape"])


class Dense(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        scale = np.sqrt(2.0 / in_features)
        self.params["W"] = np.random.randn(out_features,in_features)*scale
        self.params["b"] = np.zeros((out_features,1))

    def forward(self, x, training=True):
        self.cache["x"] = x
        return x.dot(self.params["W"].T) + self.params["b"].T

    def backward(self, grad_output):
        x = self.cache["x"]
        self.grads["W"] = grad_output.T.dot(x)
        self.grads["b"] = np.sum(grad_output, axis=0, keepdims=True).T
        return grad_output.dot(self.params["W"])


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, x, training=True):
        self.cache["mask"] = (x>0).astype(float)
        return np.maximum(0,x)

    def backward(self, grad_output):
        return grad_output * self.cache["mask"]


class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.trainable = False

    def forward(self, x, training=True):
        if training:
            mask = (np.random.rand(*x.shape) > self.p).astype(float)
            self.cache["mask"] = mask
            return x*mask/(1-self.p)
        return x

    def backward(self, grad_output):
        return grad_output * self.cache.get("mask",1)/(1-self.p)


class BatchNorm2D(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.params["gamma"] = np.ones((num_features,1))
        self.params["beta"] = np.zeros((num_features,1))
        self.running_mean = np.zeros((num_features,1))
        self.running_var = np.ones((num_features,1))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x, training=True):
        N,C,H,W = x.shape
        if training:
            mean = np.mean(x, axis=(0,2,3), keepdims=True)
            var = np.var(x, axis=(0,2,3), keepdims=True)
            x_hat = (x-mean)/np.sqrt(var+self.eps)
            self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*mean.reshape(C,1)
            self.running_var = self.momentum*self.running_var + (1-self.momentum)*var.reshape(C,1)
            self.cache={"x":x,"x_hat":x_hat,"mean":mean,"var":var}
        else:
            mean = self.running_mean.reshape(1,C,1,1)
            var = self.running_var.reshape(1,C,1,1)
            x_hat = (x-mean)/np.sqrt(var+self.eps)
        return self.params["gamma"].reshape(1,C,1,1)*x_hat + self.params["beta"].reshape(1,C,1,1)

    def backward(self, grad_output):
        x, x_hat, mean, var = self.cache["x"], self.cache["x_hat"], self.cache["mean"], self.cache["var"]
        N,C,H,W = x.shape
        m = N*H*W
        self.grads["gamma"] = np.sum(grad_output*x_hat, axis=(0,2,3), keepdims=True).reshape(C,1)
        self.grads["beta"] = np.sum(grad_output, axis=(0,2,3), keepdims=True).reshape(C,1)
        dx_hat = grad_output*self.params["gamma"].reshape(1,C,1,1)
        dvar = np.sum(dx_hat*(x-mean)*-0.5*(var+self.eps)**(-1.5), axis=(0,2,3), keepdims=True)
        dmean = np.sum(dx_hat*-1/np.sqrt(var+self.eps), axis=(0,2,3), keepdims=True) + dvar*np.sum(-2*(x-mean), axis=(0,2,3), keepdims=True)/m
        return dx_hat/np.sqrt(var+self.eps) + dvar*2*(x-mean)/m + dmean/m


# ==============================
# Models
# ==============================
class LeNet5:
    def __init__(self, num_classes=10):
        self.layers=[
            Conv2D(3,6,5,padding="valid"), BatchNorm2D(6), ReLU(), MaxPool2D(2),
            Conv2D(6,16,5,padding="valid"), BatchNorm2D(16), ReLU(), MaxPool2D(2),
            Flatten(),
            Dense(16*5*5,120), ReLU(),
            Dense(120,84), ReLU(),
            Dense(84,num_classes)
        ]
    def forward(self,x,training=True):
        for l in self.layers: x=l.forward(x,training)
        return x
    def backward(self,g):
        for l in reversed(self.layers): g=l.backward(g)
        return g
    def get_params(self):
        return {f"layer{i}_{k}":v for i,l in enumerate(self.layers) if l.trainable for k,v in l.params.items()}
    def get_grads(self):
        return {f"layer{i}_{k}":v for i,l in enumerate(self.layers) if l.trainable for k,v in l.grads.items()}


class MiniVGG:
    def __init__(self,num_classes=10):
        self.layers=[
            Conv2D(3,32,3,"same"), BatchNorm2D(32), ReLU(),
            Conv2D(32,32,3,"same"), BatchNorm2D(32), ReLU(), MaxPool2D(2),
            Conv2D(32,64,3,"same"), BatchNorm2D(64), ReLU(),
            Conv2D(64,64,3,"same"), BatchNorm2D(64), ReLU(), MaxPool2D(2),
            Conv2D(64,128,3,"same"), BatchNorm2D(128), ReLU(),
            Conv2D(128,128,3,"same"), BatchNorm2D(128), ReLU(), MaxPool2D(2),
            Flatten(),
            Dense(128*4*4,256), ReLU(), Dropout(0.5),
            Dense(256,num_classes)
        ]
    def forward(self,x,training=True):
        for l in self.layers: x=l.forward(x,training)
        return x
    def backward(self,g):
        for l in reversed(self.layers): g=l.backward(g)
        return g
    def get_params(self):
        return {f"layer{i}_{k}":v for i,l in enumerate(self.layers) if l.trainable for k,v in l.params.items()}
    def get_grads(self):
        return {f"layer{i}_{k}":v for i,l in enumerate(self.layers) if l.trainable for k,v in l.grads.items()}


# ==============================
# Loss + Optimizer
# ==============================
class CrossEntropyLoss:
    def forward(self, logits, labels):
        shifted = logits-np.max(logits,axis=1,keepdims=True)
        exp_logits=np.exp(shifted)
        probs=exp_logits/np.sum(exp_logits,axis=1,keepdims=True)
        N=logits.shape[0]
        loss=-np.mean(np.log(probs[np.arange(N),labels]))
        self.cache={"probs":probs,"labels":labels}
        return loss
    def backward(self):
        probs,labels=self.cache["probs"],self.cache["labels"]
        N=probs.shape[0]
        grad=probs.copy()
        grad[np.arange(N),labels]-=1
        return grad/N


class SGD:
    def __init__(self, model, lr=0.01):
        self.model=model; self.lr=lr
    def step(self):
        params=self.model.get_params(); grads=self.model.get_grads()
        for k in params: params[k]-=self.lr*grads[k]


class LRScheduler:
    def __init__(self, optimizer, step_size=10, gamma=0.1):
        self.optimizer=optimizer; self.step_size=step_size; self.gamma=gamma
    def step(self, epoch):
        if (epoch+1)%self.step_size==0:
            self.optimizer.lr*=self.gamma
            print(f"[Scheduler] LR adjusted to {self.optimizer.lr:.6f}")


# ==============================
# Training + Augmentation
# ==============================
class DataAugmentation:
    def __init__(self,horizontal_flip=True,rotation_range=15,shift_range=0.1):
        self.hflip=horizontal_flip; self.rot=rotation_range; self.shift=shift_range
    def augment_batch(self,X):
        X_aug=np.empty_like(X); N,C,H,W=X.shape
        for i in range(N):
            img=X[i].transpose(1,2,0)
            if self.hflip and np.random.rand()<0.5: img=np.fliplr(img)
            if self.rot>0:
                angle=np.random.uniform(-self.rot,self.rot)
                img=rotate(img,angle,reshape=False,mode="reflect")
            if self.shift>0:
                sh=int(np.random.uniform(-self.shift,self.shift)*H)
                sw=int(np.random.uniform(-self.shift,self.shift)*W)
                shifted=np.zeros_like(img)
                h_start=max(0,sh); h_end=H+min(0,sh)
                w_start=max(0,sw); w_end=W+min(0,sw)
                shifted[h_start:h_end,w_start:w_end]=img[max(0,-sh):H-max(0,sh), max(0,-sw):W-max(0,sw)]
                img=shifted
            X_aug[i]=img.transpose(2,0,1)
        return X_aug


def accuracy(preds,labels):
    return np.mean(np.argmax(preds,axis=1)==labels)


def train(model,X_train,y_train,X_val,y_val,epochs=3,batch_size=32,lr=0.01,augment=False,scheduler=None):
    loss_fn=CrossEntropyLoss(); opt=SGD(model,lr=lr)
    augmenter=DataAugmentation() if augment else None
    num_batches=int(np.ceil(X_train.shape[0]/batch_size))
    for epoch in range(epochs):
        epoch_loss,epoch_acc=0,0
        idx=np.arange(X_train.shape[0]); np.random.shuffle(idx)
        for i in range(num_batches):
            batch=idx[i*batch_size:(i+1)*batch_size]
            Xb,yb=X_train[batch],y_train[batch]
            if augment: Xb=augmenter.augment_batch(Xb)
            logits=model.forward(Xb,True); loss=loss_fn.forward(logits,yb)
            grad=loss_fn.backward(); model.backward(grad); opt.step()
            epoch_loss+=loss; epoch_acc+=accuracy(logits,yb)
        val_logits=model.forward(X_val,False)
        val_loss=loss_fn.forward(val_logits,y_val); val_acc=accuracy(val_logits,y_val)
        if scheduler: scheduler.step(epoch)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss {epoch_loss/num_batches:.4f}, "
              f"Train Acc {epoch_acc/num_batches:.4f} | Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")


# ==============================
# Evaluation Tools
# ==============================
def confusion_matrix(preds,labels,num_classes=10):
    if preds.ndim>1: preds=np.argmax(preds,axis=1)
    cm=np.zeros((num_classes,num_classes),dtype=int)
    for t,p in zip(labels,preds): cm[t,p]+=1
    return cm

def per_class_accuracy(cm):
    return cm.diagonal()/cm.sum(axis=1,where=(cm.sum(axis=1)!=0))

def plot_confusion_matrix(cm,class_names=None):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",
                xticklabels=class_names,yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.show()


# ==============================
# Visualization
# ==============================
def visualize_filters(conv_layer):
    W=conv_layer.params["W"]; F,C,kh,kw=W.shape
    W=(W-W.min())/(W.max()-W.min()+1e-8)
    cols=8; rows=int(np.ceil(F/cols)); plt.figure(figsize=(cols,rows))
    for i in range(F):
        f=W[i]
        if C==3: f_img=np.transpose(f,(1,2,0))
        else: f_img=f[0]
        plt.subplot(rows,cols,i+1); plt.imshow(f_img); plt.axis("off")
    plt.show()

def visualize_feature_maps(model,X_sample,layer_indices):
    x=X_sample[np.newaxis,...]; acts=[]
    for idx,l in enumerate(model.layers):
        x=l.forward(x,False)
        if idx in layer_indices: acts.append((idx,x.copy()))
    for idx,act in acts:
        N,C,H,W=act.shape; cols=8; rows=int(np.ceil(C/cols))
        plt.figure(figsize=(cols,rows))
        for i in range(C):
            plt.subplot(rows,cols,i+1); plt.imshow(act[0,i],cmap="gray"); plt.axis("off")
        plt.suptitle(f"Feature Maps at Layer {idx}"); plt.show()


def load_cifar10(normalize=True):
    """Load CIFAR-10 using Keras dataset loader."""
    from tensorflow.keras.datasets import cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Convert to (N, C, H, W)
    X_train = X_train.transpose(0, 3, 1, 2).astype(np.float32)
    X_test = X_test.transpose(0, 3, 1, 2).astype(np.float32)
    y_train, y_test = y_train.flatten(), y_test.flatten()

    if normalize:
        X_train /= 255.0
        X_test /= 255.0
    return X_train, y_train, X_test, y_test



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN from scratch")
    parser.add_argument("--mode", type=str, default="mini",
                        choices=["mini", "full"],
                        help="Choose run mode: mini (quick test) or full (CIFAR-10)")
    parser.add_argument("--model", type=str, default="lenet",
                        choices=["lenet", "vgg"],
                        help="Choose model architecture")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    # Select model
    if args.model == "lenet":
        model = LeNet5(num_classes=10)
    else:
        model = MiniVGG(num_classes=10)

    if args.mode == "mini":
        print("🔹 Running mini workflow (random data)")
        # Random data
        X = np.random.randn(50, 3, 32, 32)
        y = np.random.randint(0, 10, 50)

        history = train(model, X[:40], y[:40], X[40:], y[40:],
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, augment=True)

        plot_history(history)

        logits = model.forward(X[40:], training=False)
        cm = confusion_matrix(logits, y[40:], num_classes=10)
        print("\nPer-class accuracy (mini run):")
        for i, acc in enumerate(per_class_accuracy(cm)):
            print(f"Class {i}: {acc:.2f}")
        plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])
        print("✅ Mini run finished")

    elif args.mode == "full":
        print("🔹 Running full workflow (CIFAR-10)")
        X_train, y_train, X_test, y_test = load_cifar10()

        # Validation split
        X_val, y_val = X_train[:5000], y_train[:5000]
        X_train, y_train = X_train[5000:], y_train[5000:]

        history = train(model, X_train[:5000], y_train[:5000],  # use subset for speed
                        X_val[:1000], y_val[:1000],
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, augment=True)

        plot_history(history)

        logits = model.forward(X_test[:2000], training=False)  # subset test for speed
        cm = confusion_matrix(logits, y_test[:2000], num_classes=10)
        print("\nPer-class accuracy (CIFAR-10):")
        for i, acc in enumerate(per_class_accuracy(cm)):
            print(f"Class {i}: {acc:.2f}")
        plot_confusion_matrix(cm, class_names=[str(i) for i in range(10)])
        print("✅ Full CIFAR-10 run finished")