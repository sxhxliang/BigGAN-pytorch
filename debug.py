from model_resnet import *
from demo import *
from utils import *

dim_z = 120
vocab_size = 1000


num_samples = 12 #@param {type:"slider", min:1, max:20, step:1}
truncation = 0.32 #@param {type:"slider", min:0.02, max:1, step:0.02}
noise_seed = 0 #@param {type:"slider", min:0, max:100, step:1}
category = "951"


z = truncated_z_sample(num_samples, truncation, noise_seed)
y = int(951)
# print(z)


feed_dict = sample(z, y, truncation=truncation)
# print(feed_dict['input_y'].shape)


model = Generator(code_dim=120, n_class=1000, chn=6, debug=True)
# inputs = torch.from_numpy(feed_dict['input_z']).float()
# labels = torch.from_numpy(feed_dict['input_y']).float()
# out = model(inputs,labels)

# print(out.size())
# model.apply(weights_init)


print('0,1,2,3'.split(','))
# torch.save(model.state_dict(),'test_model.pth')



