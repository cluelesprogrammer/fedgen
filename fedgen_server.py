import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import copy
import time
from torch.utils.data import ConcatDataset
MIN_SAMPLES_PER_LABEL=1
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fedgen_user import UserpFedGen
import yaml
import wandb
from main import evaluator

MIN_SAMPLES_PER_LABEL=1
RUNCONFIGS = {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 10,
            'num_pretrain_iters': 10,
            'ensemble_alpha': 1,  # teacher loss (server side)
            'ensemble_beta': 0,  # adversarial student loss
            'ensemble_eta': 1,
            'unique_labels': 19,
            'generative_alpha': 10,
            'generative_beta': 10,
            'weight_decay': 1e-2
        }

class FedGen():
    def __init__(self, id_client_name: dict, config, train_loaders, test_loader, model, device, val_loader=None):
        total_users = len(id_client_name)
        self.test_loader = test_loader
        self.config = config
        with open('config/fedgen_config.yaml') as file:
            fedgen_dict = yaml.safe_load(file)
        self.config.update(fedgen_dict)

        self.model = model
        self.student_model = copy.deepcopy(self.model)
        generator_layers = (512, 2048,  19, 32)
        self.generative_model = Generator(generator_layers)
        self.generative_model.to(device)
        #-1 when layer is the final linear layer
        self.latent_layer_idx = self.generative_model.latent_layer_idx
        self.ensemble_lr = RUNCONFIGS['ensemble_lr']
        self.ensemble_batch_size = RUNCONFIGS['ensemble_batch_size']
        self.ensemble_epochs = RUNCONFIGS['ensemble_epochs']
        self.num_pretrain_iters = RUNCONFIGS['num_pretrain_iters']
        self.temperature = 1
        self.unique_labels = RUNCONFIGS['unique_labels']
        self.ensemble_alpha = RUNCONFIGS['ensemble_alpha']
        self.ensemble_beta = RUNCONFIGS['ensemble_beta']
        self.ensemble_eta = RUNCONFIGS['ensemble_eta']
        self.weight_decay = RUNCONFIGS['weight_decay']
        self.generative_alpha = RUNCONFIGS['generative_alpha']
        self.generative_beta = RUNCONFIGS['generative_beta']
        self.ensemble_train_loss = []
        self.n_teacher_iters = 5
        self.n_student_iters = 1
        self.batch_size = config['batch_size']
        self.gen_batch_size = config['gen_batch_size']
        self.epochs = config['epochs']
        self.generative_optimizer = torch.optim.AdamW(
            params=self.generative_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)
        with open('label_frequencies.yaml') as file:
            self.label_frequency = yaml.full_load(file)

        #### creating users ####
        self.users = []
        self.total_train_samples = 0
        for t in train_loaders:
            self.total_train_samples += config['batch_size'] * len(train_loaders[t])

        self.available_labels = {
            "Urban fabric": 0,
            "Industrial or commercial units": 1,
            "Arable land": 2,
            "Permanent crops": 3,
            "Pastures": 4,
            "Complex cultivation patterns": 5,
            "Land principally occupied by agriculture, with significant areas of natural vegetation": 6,
            "Agro-forestry areas": 7,
            "Broad-leaved forest": 8,
            "Coniferous forest": 9,
            "Mixed forest": 10,
            "Natural grassland and sparsely vegetated areas": 11,
            "Moors, heathland and sclerophyllous vegetation": 12,
            "Transitional woodland, shrub": 13,
            "Beaches, dunes, sands": 14,
            "Inland wetlands": 15,
            "Coastal wetlands": 16,
            "Inland waters": 17,
            "Marine waters": 18
        }
        self.device = device
        self.user_idx = list(id_client_name.keys())
        for id in self.user_idx:
            user=UserpFedGen(config, id, model, self.generative_model,
                 train_loaders[id], self.available_labels, self.latent_layer_idx, self.label_frequency[id_client_name[id]], device)
            self.users.append(user)
        print("Number of Train/Test samples:", self.total_train_samples)
        print("Data from {} users in total.".format(len(self.user_idx)))
        print("Finished creating FedAvg server.")
        self.init_loss_fn()

    def init_loss_fn(self):
        self.loss = nn.BCELoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")  # ,log_target=True)
        self.ce_loss = nn.CrossEntropyLoss()

    def train_test(self):
        #### pretraining
        for epoch in range(self.epochs):
            print("\n\n-------------Round number: ",epoch, " -------------\n\n")
            # broadcast averaged prediction model
            self.send_parameters()
            self.timestamp = time.time() # log user-training start time
            for user_id, user in zip(self.user_idx, self.users): # allow selected users to train
                user.train(
                    epoch,
                    regularization= epoch > 3)
            self.timestamp = time.time() # log server-agg start time
            self.train_generator(
                self.batch_size,
                epoches=self.ensemble_epochs // self.n_teacher_iters,
                latent_layer_idx=self.latent_layer_idx,
                verbose=True
            )
            #fedavg
            self.aggregate_parameters()

            # Validation
        metrics = evaluator(self.model, self.test_loader)
        # add _country as a prefix to the metrics dict
        for k, v in __metrics.get().items():
            _logs["server_".format(k)] = v

        _wandb_run.log(_logs, step=_round)

    def send_parameters(self, beta=1):
        users = self.users
        for user in self.users:
            user.set_parameters(self.model)

    def add_parameters(self, user, ratio):
        for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def aggregate_parameters(self):
        #assert (self.selected_users is not None and len(self.selected_users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.users:
            total_train += user.train_samples
        for user in self.users:
            self.add_parameters(user, user.train_samples / total_train)

    def evaluate(self, save=True, selected=False):
        # override evaluate function to log vae-loss.
        test_ids, test_samples, test_accs, test_losses = self.test(selected=selected)
        glob_acc = np.sum(test_accs) * 1.0 / np.sum(test_samples)
        glob_loss = np.sum([x * y for (x, y) in zip(test_samples, test_losses)]).item() / np.sum(test_samples)
        """
        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)
        """
        print("Average Global Accurancy = {:.4f}, Loss = {:.2f}.".format(glob_acc, glob_loss))

    def test(self, selected=False):
        for samples in tqdm(self.test_loader):
            with torch.no_grad():
                output = self.model(samples['X'])['output']
                self.metrics.update(output, samples['y'])


    def train_generator(self, batch_size, epoches=1, latent_layer_idx=-1, verbose=False):
        """
        Learn a generator that find a consensus latent representation z, given a label 'y'.
        :param batch_size:
        :param epoches:
        :param latent_layer_idx: if set to -1 (-2), get latent representation of the last (or 2nd to last) layer.
        :param verbose: print loss information.
        :return: Do not return anything.
        """
        #self.generative_regularizer.train()
        self.label_weights, self.qualified_labels = self.get_label_weights()
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0

        def update_generator_(n_iters, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            self.generative_model.train()
            student_model.eval()
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                y_sampled = sample_multi_label_distribution(len(self.available_labels), self.gen_batch_size)
                #which labels have been selected for every sampled y
                labels_in_sample = [np.where(y_i == 1)[0] for y_i in y_sampled]
                #y=np.random.choice(self.qualified_labels, batch_size)
                y_sampled = torch.from_numpy(y_sampled).float().to(self.device)
                ## feed to generator
                gen_result=self.generative_model(y_sampled, self.device, latent_layer_idx=latent_layer_idx)
                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps=gen_result['output'], gen_result['eps']
                ##### get losses ####
                # decoded = self.generative_regularizer(gen_output)
                # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
                diversity_loss=self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs
                ######### get teacher loss ############
                teacher_loss=0
                teacher_logit=0
                for user_id, user in enumerate(self.users):
                    user.model.eval()
                    #every labels' weights in the user's data
                    label_weights=self.label_weights[:, user_id].reshape(-1, 1)
                    #because of multi label case, sample's label weights is averaged accordingly
                    weight = [[label_weights[i] for i in ls] for ls in labels_in_sample]
                    weight = torch.tensor([sum(w)/len(w) for w in weight]).to(self.device)
                    expand_weight=np.tile(weight.cpu().numpy(), (1, self.unique_labels))
                    expand_weight = torch.from_numpy(expand_weight).to(self.device)
                    user_result_given_gen=user.model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                    teacher_loss_=torch.mean( \
                        self.loss(user_result_given_gen['output'], y_sampled) * \
                        torch.tensor(weight, dtype=torch.float32))
                    teacher_loss+=teacher_loss_
                    wandb.log({'generator_teacher_loss_user_{}'.format(user_id): teacher_loss_})
                    teacher_logit+=user_result_given_gen['logit'] * torch.tensor(expand_weight, dtype=torch.float32)

                ######### get student loss ############
                student_output=student_model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                student_loss=F.kl_div(F.logsigmoid(student_output['logit']), F.sigmoid(teacher_logit))
                wandb.log({'generator_student_loss_total': student_loss})
                if self.ensemble_beta > 0:
                    loss=self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss
                else:
                    loss=self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += self.ensemble_alpha * teacher_loss#(torch.mean(TEACHER_LOSS.double())).item()
                STUDENT_LOSS += self.ensemble_beta * student_loss#(torch.mean(student_loss.double())).item()
                DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#(torch.mean(diversity_loss.double())).item()
                wandb.log({'total_teacher_loss' : TEACHER_LOSS,
                        'total_student_loss' : STUDENT_LOSS,
                        'total_diversity_loss' : DIVERSITY_LOSS,
                })

            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS

        for i in range(epoches):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS=update_generator_(
                self.n_teacher_iters, self.model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)

        TEACHER_LOSS = TEACHER_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        STUDENT_LOSS = STUDENT_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS.detach().cpu().\
                             numpy() / (self.n_teacher_iters * epoches)
        info="Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        if verbose:
            print(info)
        self.generative_lr_scheduler.step()

    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        print(self.unique_labels, ' are the number of unique labels')
        for label in range(self.unique_labels):
            weights = []
            for user in self.users:
                weights.append(user.label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append( np.array(weights) / np.sum(weights) )
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        #label weight is the weight of every label in every user
        return label_weights, qualified_labels

def sample_multi_label_distribution(total_labels, batch_size, qualified_labels='all'):
    #labels are drawn with a probability higher than it iwereif it were uniformly distributed
    #to encourage multi label samples
    while True:
        if qualified_labels=='all':
            prior_label_distribution = np.tile(np.array(2.5/total_labels), total_labels)
            sampled_y = np.random.binomial(1, np.tile(prior_label_distribution, batch_size)).reshape(
                batch_size, total_labels)
        elif (type(qualified_labels) == list):
            label_p = 2.5/len(qualified_labels)
            prior_label_distribution = np.zeros(total_labels)
            prior_label_distribution[qualified_labels] = label_p
            prior_label_distribution = np.tile(prior_label_distribution, batch_size)
            sampled_y = np.random.binomial(1, prior_label_distribution).reshape(batch_size, total_labels)
        else:
            print('wrong input')
        #to ensure that at least one labels has been sampled
        least_one_label_bool = np.array([np.any(y == 1) for y in sampled_y])
        least_one_label_per_sample = np.all(least_one_label_bool)
        if (least_one_label_per_sample):
            break
    return sampled_y

class Generator(nn.Module):
    def __init__(self, layers, device=torch.device("cpu"), latent_layer_idx=-1):
        super(Generator, self).__init__()
        #self.model=model
        self.latent_layer_idx = latent_layer_idx
        #input channel is an unnecessary variable as the n_class and embedding determines the input channel of generator
        self.hidden_dim, self.latent_dim, self.n_class, self.noise_dim = layers
        input_dim = self.noise_dim + self.n_class
        self.fc_configs = [input_dim, self.hidden_dim]
        self.build_network()
        self.init_loss_fn()

    def get_number_of_parameters(self):
        pytorch_total_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def init_loss_fn(self):
        self.crossentropy_loss=nn.NLLLoss(reduce=False) # same as above
        self.diversity_loss = DiversityLoss(metric='l1')
        self.dist_loss = nn.MSELoss()

    def build_network(self):
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        ### Representation layer
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, y_input, device, latent_layer_idx=-1):
        result = {}
        batch_size = y_input.shape[0]
        eps = torch.rand((batch_size, self.noise_dim)).to(device) # sampling from Gaussian
        result['eps'] = eps
        #y_input is n-hot vector
        z = torch.cat((eps, y_input), dim=1)
        ### FC layers
        for layer in self.fc_layers:
            z = layer(z)
        z = self.representation_layer(z)
        result['output'] = z
        return result

    @staticmethod
    def normalize_images(layer):
        """
        Normalize images into zero-mean and unit-variance.
        """
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1)) \
            .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std

class DivLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward2(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        chunk_size = layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer=layer.view((layer.size(0), -1))
        chunk_size=layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))





