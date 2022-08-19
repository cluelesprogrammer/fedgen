from userpFedGen import UserpFedGen
from generator import create_generative_model
import sys
from data_utils import sample_multi_label_distribution
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import copy
import time
from torch.utils.data import ConcatDataset
import wandb
from tqdm import tqdm
from model_utils import evaluator


class FedGen():
    def __init__(self, config, data, model, metrics, seed, debug=False):
        self.config = config
        self.init_loss_fn()
        self.init_ensemble_config()
        self.algorithm = 'fedgen'
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.mode='partial' if 'partial' in self.algorithm.lower() else 'all'
        self.num_glob_iters = config['num_glob_iters']
        self.seed = seed
        self.deviations = {}
        self.qualified_labels, self.available_labels, self.label_weights = None, config['unique_labels'], None
        # Initialize data for all users
        #data = read_data(config[dataset)
        self.metrics = metrics
        self.debug=debug
        # data contains: clients, train_loaders, test_loader
        clients = data[0]
        total_users = copy.copy(clients)
        if config['num_users'] == 'all':
            self.num_users = len(total_users)
        else:
            if int(self.num_users) > len(total_users):
                self.num_users = len(total_users)
        self.train_loaders, self.test_loader = data[1], data[2]
        self.total_test_samples = len(self.test_loader) * config['batch_size']
        self.local = 'local' in self.algorithm.lower()
        self.student_model = copy.deepcopy(self.model)
        self.generative_model = create_generative_model(config['dataset'], config['algorithm'], config['backbone'], config['embedding'], device=config['device'])

        if not config['train']:
            print('number of generator parameteres: [{}]'.format(self.generative_model.get_number_of_parameters()))
            print('nu [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = self.generative_model.latent_layer_idx
        self.available_labels = config['labels']
        self.gen_batch_size = config['gen_batch_size']
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
        self.min_samples_per_labels = 1

        #getting prior label distribution
        all_clients = config['clients']
        label_p = {l: 0 for l in range(19)}
        label_acc = {l: 0 for l in range(19)}

        for c in all_clients:
            country_label_freq = config['label_frequencies'][c]
            for label in country_label_freq:
                label_acc[label] += country_label_freq[label]

        label_p = {c: label_acc[c]/sum(label_acc.values()) for c in label_acc}
        self.config['label_p'] = label_p

        #### creating users and selected users later used for training####
        self.users = []
        self.selected_users = []

        print(total_users)
        for id in total_users:
            self.total_train_samples+=len(self.train_loaders[id]) * config['batch_size']
            user=UserpFedGen(
                config, id, model, self.generative_model,
                self.train_loaders[id],
                self.available_labels, self.latent_layer_idx, config['label_frequencies'][config['id_countries'][id]], config['id_countries'][id], #contains id to country mapping
                use_adam=True)
            self.users.append(user)

        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedAvg server.")

    def init_loss_fn(self):
        self.loss=nn.BCELoss()

    def init_ensemble_config(self):
        #### used for ensemble learning, parameters for generative model####
        self.embedding = int(self.config['embedding'])
        self.n_teacher_iters = int(self.config['n_teacher_iters'])
        self.n_student_iters = int(self.config['n_student_iters'])
        self.ensemble_lr = float(self.config['ensemble_lr'])
        self.ensemble_batch_size = int(self.config['ensemble_batch_size'])
        self.ensemble_epochs = int(self.config['ensemble_epochs'])
        self.num_pretrain_iters = int(self.config['num_pretrain_iters'])
        self.unique_labels = int(self.config['unique_labels'])
        self.ensemble_alpha = float(self.config['ensemble_alpha'])
        self.ensemble_beta = float(self.config['ensemble_beta'])
        self.ensemble_eta = float(self.config['ensemble_eta'])
        self.weight_decay = float(self.config['weight_decay'])
        self.generative_alpha = float(self.config['generative_alpha'])
        self.generative_beta = float(self.config['generative_beta'])
        self.ensemble_train_loss = []

    def aggregate_and_broadcast(self):
        # Count the number of clients
        with torch.no_grad():
            for key in self.model.state_dict().keys():
                temp = torch.zeros_like(self.model.state_dict()[key], dtype=torch.float32)
                for id, user in enumerate(self.selected_users):
                    temp += (1/len(self.selected_users)) * user.model.state_dict()[key]
                self.model.state_dict()[key].data.copy_(temp)
                for id, user in enumerate(self.selected_users):
                   user.model.state_dict()[key].data.copy_(self.model.state_dict()[key])

    def train_test(self):
        test_n_user_every_iter = 1
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            if (self.debug and glob_iter > 2):
                break
            self.selected_users, self.user_idxs=self.select_users(glob_iter, self.num_users, return_idx=True)
            #broadcasting server model's parameters on to user models
            #self.evaluate()
            chosen_verbose_user = np.random.randint(0, len(self.users))
            self.timestamp = time.time() # log user-training start time
            user_tested = 0
            for user_id, user in zip(self.user_idxs, self.selected_users): # allow selected users to train
                user.train(
                    glob_iter,
                    self.qualified_labels, self.label_weights,
                    verbose=user_id == chosen_verbose_user and glob_iter > 0,
                    regularization=glob_iter>self.config['distill_start'] and glob_iter<self.config['distill_stop'],debug=self.debug)

            self.train_generator(
                self.config['batch_size'],
                epoches=self.ensemble_epochs // self.n_teacher_iters,
                latent_layer_idx=self.latent_layer_idx,
                verbose=True
            )
            self.aggregate_and_broadcast()
            server_metrics = evaluator(self.model, self.test_loader, self.config['device']).get()
            server_metrics_log = {}
            for k in server_metrics:
                server_metrics_log['server' + ' ' + k] = server_metrics[k]
            server_metrics_log['communication_round'] = glob_iter
            wandb.log(server_metrics_log)

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
                #this function samples at least one label for every sample; in average 2.5 labels per sample
                y_sampled = sample_multi_label_distribution(self.gen_batch_size, self.config['label_p'])
                labels_in_sample = [np.where(y_i == 1)[0] for y_i in y_sampled]
                y_sampled = torch.from_numpy(y_sampled).float().to(self.config['device'])
                ## feed to generator
                gen_result=self.generative_model(y_sampled, latent_layer_idx=latent_layer_idx, verbose=True)
                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps=gen_result['output'], gen_result['eps']
                ##### get losses ####
                diversity_loss=self.generative_model.diversity_loss(eps, gen_output)  # encourage different outputs
                ######### get teacher loss ############
                teacher_loss=0
                teacher_logit=0
                for user_idx, user in enumerate(self.selected_users):
                    user.model.eval()
                    #every labels' weights in the user's data
                    label_weights=self.label_weights[:, user_idx].reshape(-1, 1)
                    #sample's label weights is averaged accordingly
                    weight = [[label_weights[i] for i in ls] for ls in labels_in_sample]
                    #talking average of every label's weight in the sample to obtain label weight for multi label case
                    weight = torch.tensor(np.array([sum(w)/len(w) for w in weight]))
                    expand_weight=np.tile(weight, (1, self.unique_labels))
                    weight =  weight.to(self.config['device'])
                    expand_weight = torch.tensor(expand_weight, dtype=torch.float32)
                    expand_weight =expand_weight.to(self.config['device'])
                    user_result_given_gen=user.model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                    teacher_loss_=torch.mean( self.loss(user_result_given_gen['output'], y_sampled) * weight)
                        #torch.tensor(weight, dtype=torch.float32))
                    teacher_loss+=teacher_loss_
                    teacher_logit+=user_result_given_gen['logit'] * expand_weight 

                ######### get student loss ############
                student_output=student_model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                student_loss=self.loss(F.sigmoid(student_output['logit']), F.sigmoid(teacher_logit))
                if self.ensemble_beta > 0:
                    loss=self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss
                else:
                    loss=self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += self.ensemble_alpha * teacher_loss#(torch.mean(TEACHER_LOSS.double())).item()
                STUDENT_LOSS += self.ensemble_beta * student_loss#(torch.mean(student_loss.double())).item()
                DIVERSITY_LOSS += self.ensemble_eta * diversity_loss#(torch.mean(diversity_loss.double())).item()
                wandb.log({'teacher_loss' : teacher_loss,
                        'student_loss' : student_loss,
                        'diversity_loss': diversity_loss,
                        'ensemble_step': epoch * self.n_teacher_iters + i
                })

            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS

        print('training generator....')
        #for logging
        epoch = 0
        for _epoch in tqdm(range(epoches)):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS=update_generator_(
                self.n_teacher_iters, self.model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)

        TEACHER_LOSS = TEACHER_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        STUDENT_LOSS = STUDENT_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS.detach().cpu().numpy() / (self.n_teacher_iters * epoches)
        info="Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        if verbose:
            print(info)
        self.generative_lr_scheduler.step()

    def select_users(self, round, num_users, return_idx=False):
        '''selects num_clients clients weighted by number of samples from possible_clients
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            return self.users, range(len(self.users))

        num_users = min(num_users, len(self.users))
        if return_idx:
            user_idxs = np.random.choice(range(len(self.users)), num_users, replace=False)  # , p=pk)
            return [self.users[i] for i in user_idxs], user_idxs
        else:
            return np.random.choice(self.users, num_users, replace=False)

    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])
            if np.max(weights) > self.min_samples_per_labels:
                qualified_labels.append(label)
            w, ws = np.array(weights).astype(np.float64), np.sum(weights).astype(np.float64)
            to_append = np.divide(w, ws, out=np.zeros_like(w), where=ws!=0.0)
            label_weights.append(to_append)
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        #label weight is the weight of every label in every user
        return label_weights, qualified_labels

