from keras.applications import vgg19
from keras import backend as K
from utils.loaders import preprocess_image, load_img, img_to_array, deprocess_image
import numpy as np
import PIL
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
img_ncols=128
img_nrows=128
base_img_path = 'NST/'
style_img_path = 'NST/'


class Evaluator(object):
    def __init__(self, f, shp): self.f, self.shp = f, shp

    def loss(self, x):
        loss_, self.grad_values = self.f([x.reshape(self.shp)])
        return loss_.astype(np.float64)

    def grads(self, x): return self.grad_values.flatten().astype(np.float64)

content_weight=0.01


base_img=K.variable(preprocess_image(base_img_path,"base.jpg"))
style_img=K.variable(preprocess_image(style_img_path,"stylp.jpg"))
combination_img=K.placeholder((1,128,128,3))
style_shape = style_img.shape
'''
print(style_img.shape)
print(base_img.shape)
'''

input_tensor=K.concatenate([base_img,style_img,combination_img], axis=0)
print(input_tensor.shape)
model=vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

#content loss
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
layer_features = outputs_dict['block5_conv2']

base_img_features = layer_features[0, :, :, :]
comb_features = layer_features[2, :, :, :]

def content_loss(content, gen):
    return K.sum(K.square(gen - content))

content_loss = content_weight*content_loss(base_img_features, comb_features)


#style loss
style_weight=100
style_loss_val=0.0

def gram_matrix(x):
    features=K.batch_flatten(K.permute_dimensions(x, (2,0,1)))
    gram=K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S=gram_matrix(style)
    C=gram_matrix(combination)
    channels=3
    size=img_nrows*img_ncols
    return K.sum(K.square(S-C)) / (4.0*(channels**2)*(size**2))

feature_layers=['block1_conv1', 'block2_conv1',
                'block3_conv1', 'block4_conv1',
                'block5_conv1']

for layer_name in feature_layers:
    layer_features=outputs_dict[layer_name]
    style_reference_features=layer_features[1, :, :, :]
    combination_features=layer_features[2, :, :, :]
    sl=style_loss(style_reference_features, combination_features)
    style_loss_val += (style_weight/len(feature_layers)) * sl

#Total Variance Loss
tot_var_weight=20
def tot_var_loss(x):
    a=K.square(
        x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :]
    )
    b = K.square(
        x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :]
    )
    return K.sum(K.pow(a+b,1.25))

tv_loss=tot_var_weight*tot_var_loss(input_tensor)


#total loss
loss = tv_loss + style_loss_val + content_loss

'''
grads = K.gradients(loss, model.input)
fn = K.function([model.input], [loss] + grads)
evaluator = Evaluator(fn, style_shape)

from scipy.optimize import fmin_l_bfgs_b

iterations=1000
x=preprocess_image(base_img_path,'base.jpg')

for i in range(iterations):
    x, min_val, info = fmin_l_bfgs_b(
        evaluator.loss,
        x.flatten(),
        fprime=evaluator.grads,
        maxfun=20
    )
'''
# Get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_img)[0]

# Function to fetch the values of the current loss and the current gradients
fetch_loss_and_grads = K.function([combination_img], [loss, grads])


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_nrows, img_ncols, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

from scipy.misc import imsave
import time
from scipy.optimize import fmin_l_bfgs_b

result_prefix = 'style_transfer_result'
iterations = 200

# Run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss.
# This is our initial state: the target image.
# Note that `scipy.optimize.fmin_l_bfgs_b` can only process flat vectors.
x = preprocess_image(base_img_path,'base.jpg')
x = x.flatten()
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                     fprime=evaluator.grads, maxfun=20)
    if i%10==0:
        print('Current loss value:', min_val)
        # Save current generated image
        img = x.copy().reshape((img_nrows, img_ncols, 3))
        img = deprocess_image(img)
        fname = result_prefix + '_at_iteration_%d.png' % i
        imsave(fname, img)
        end_time = time.time()
        print('Image saved as', fname)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))

'''
# Content image
plt.imshow(load_img(target_image_path, target_size=(img_height, img_width)))
plt.figure()

# Style image
plt.imshow(load_img(style_reference_image_path, target_size=(img_height, img_width)))
plt.figure()

# Generate image
plt.imshow(img)
plt.show()
'''