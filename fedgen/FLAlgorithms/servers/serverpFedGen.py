from fedgen.FLAlgorithms.users.userpFedGen import UserpFedGen
from .serverbase import Server
import sys
from utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model, sample_multi_label_distribution
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
import copy
import time
from torch.utils.data import ConcatDataset
MIN_SAMPLES_PER_LABEL=1
from torchmetrics import Accuracy
import wandb
from tqdm import tqdm


class FedGen(Server):
    def __init__(self, config, data, model, metrics, seed, debug=False)
        self.config = config
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.mode='partial' if 'partial' in self.algorithm.lower() else 'all'
        self.seed = seed
        self.deviations = {}
        # Initialize data for all users
        #data = read_data(config[dataset)
        self.metrics = metrics
        self.debug=debug
        self.init_loss_fn()
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
        self.use_adam = 'adam' in self.algorithm.lower()
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
                use_adam=self.use_adam)
            self.users.append(user)

        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedAvg server.")

   def init_ensemble_configs(self):
        #### used for ensemble learning, parameters for generative model####
        self.embedding = self.config['embedding']
        self.n_teacher_iters = self.config['n_teacher_iters']
        self.n_student_iters = self.config['n_student_iters']

        self.ensemble_lr = self.config['ensemble_lr']
        self.ensemble_batch_size = self.config['ensemble_batch_size']
        self.ensemble_epochs = self.config['ensemble_epochs']
        self.num_pretrain_iters = self.config['num_pretrain_iters']
        self.unique_labels = self.config['unique_labels']
        self.ensemble_alpha = self.config['ensemble_alpha']
        self.ensemble_beta = self.config['ensemble_beta']
        self.ensemble_eta = self.config['ensemble_eta']
        self.weight_decay = self.config['weight_decay']
        self.generative_alpha = self.config['generative_alpha']
        self.generative_beta = self.config['generative_beta']
        self.ensemble_train_loss = []

    def train_test(self):
        for glob_iter in range(config['num_glob_iters']):
            print("\n\n-------------Round number: ",glob_iter, " -------------\n\n")
            if (self.debug and glob_iter > 2):
                break
            self.selected_users, self.user_idxs=self.select_users(glob_iter, self.num_users, return_idx=True)
            self.send_parameters()
            if not self.local:
                self.send_parameters(mode=self.mode) # broadcast averaged prediction model
            #self.evaluate()
            chosen_verbose_user = np.random.randint(0, len(self.users))
            self.timestamp = time.time() # log user-training start time
            for user_id, user in zip(self.user_idxs, self.selected_users): # allow selected users to train
                user.train(
                    glob_iter,
                    personalized=self.personalized,
                    verbose=user_id == chosen_verbose_user and glob_iter > 0,
                    regularization=glob_iter>3 and glob_iter<10,debug=self.debug)

            curr_timestamp = time.time() # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
            #self.metrics['user_train_time'].append(train_time)

            #if self.personalized:
            #    self.evaluate_personalized_model()
            self.timestamp = time.time() # log server-agg start time
            self.train_generator(
                self.config['batch_size'],
                epoches=self.ensemble_epochs // self.n_teacher_iters,
                latent_layer_idx=self.latent_layer_idx,
                verbose=True
            )
            self.aggregate_parameters()
            server_metrics = self.test()
            server_metrics['communication_round'] = glob_iter
            wandb.log(server_metrics)
            #self.metrics['server_agg_time'].append(agg_time)
            #if glob_iter  > 0 and glob_iter % 20 == 0 and self.latent_layer_idx == 0:
            #    self.visualize_images(self.generative_model, glob_iter, repeats=10)

        #self.save_results(args)
        #self.save_model()

    def test(self, user_model=False):
        self.model.eval()  # set model to evaluation mode
        self.metrics.reset() #flushing all values from preivous epoch
        for _data, _target in tqdm(self.test_loader, total=len(self.test_loader), desc="Evaluator", unit="Batch", dynamic_ncols=True):
            _data = _data.type(torch.float32).to(device)
            _target = _target.type(torch.int8).to(device)
            with torch.no_grad():
                _preds = self.model(_data)
            self.metrics.update(_preds, _target)
            return self.metrics.get()

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
            for i in tqdm(range(n_iters)):
                if (self.debug and i > 2):
                    break
                self.generative_optimizer.zero_grad()
                #this function samples at least one label for every sample; in average 2.5 labels per sample
                y_sampled = sample_multi_label_distribution(len(self.available_labels), self.gen_batch_size)
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
                for user_idx, user in enumerate(tqdm(self.selected_users)):
                    user.model.eval()
                    #every labels' weights in the user's data
                    label_weights=self.label_weights[:, user_idx].reshape(-1, 1)
                    #sample's label weights is averaged accordingly
                    weight = [[label_weights[i] for i in ls] for ls in labels_in_sample]
                    weight = torch.tensor([sum(w)/len(w) for w in weight])
                    expand_weight=np.tile(weight, (1, self.unique_labels))
                    weight =  weight.to(self.config['device'])
                    expand_weight = torch.tensor(expand_weight, dtype=torch.float32)
                    expand_weight = expand_weight.to(self.config['device'])
                    user_result_given_gen=user.model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                    teacher_loss_=torch.mean( self.loss(user_result_given_gen['output'], y_sampled) * weight)
                        #torch.tensor(weight, dtype=torch.float32))
                    teacher_loss+=teacher_loss_
                    teacher_logit+=user_result_given_gen['logit'] * expand_weight 

                ######### get student loss ############
                student_output=student_model(gen_output, start_layer_idx=latent_layer_idx, logit=True)
                student_loss=F.kl_div(F.log_sigmoid(student_output['logit'], dim=1), F.sigmoid(teacher_logit, dim=1))
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
            if (self.debug and i > 2):
                break
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

    def send_parameters(self, mode='all', beta=1, selected=False):
        users = self.users
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.selected_users
        for user in users:
            user.set_parameters(self.model,beta=beta)


    def add_parameters(self, user, ratio, partial=False):
        if partial:
            for server_param, user_param in zip(self.model.get_shared_parameters(), user.model.get_sh>                server_param.data = server_param.data + user_param.data.clone() * ratio
        else:
            for server_param, user_param in zip(self.model.parameters(), user.model.parameters()):
                server_param.data = server_param.data + user_param.data.clone() * ratio


    def aggregate_parameters(self,partial=False):
        assert (self.selected_users is not None and len(self.selected_users) > 0)
        if partial:
            for param in self.model.get_shared_parameters():
                param.data = torch.zeros_like(param.data)
        else:
            for param in self.model.parameters():
                param.data = torch.zeros_like(param.data)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train,partial=partial)

    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        print(self.unique_labels, ' are the number of unique labels')
        for label in range(self.unique_labels):
            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append( np.array(weights) / np.sum(weights) )
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        #label weight is the weight of every label in every user
        return label_weights, qualified_labels
