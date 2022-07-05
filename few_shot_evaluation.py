import random
import json
import torch
import math
import numpy as np
import os 
from args import args

f = open(args.dataset_path + "datasets.json")    
all_datasets = json.loads(f.read())
f.close()

class EpisodicGenerator():
    def __init__(self, datasetName, max_classes=50, num_elements_per_class=None, balanced_queries=True, verbose=False):
        assert datasetName != None or num_elements_per_class!=None, "datasetName and num_elements_per_class can't both be None"
        
        self.verbose = verbose
        self.datasetName = datasetName
        self.balanced_queries = balanced_queries
        if datasetName != None and datasetName in all_datasets.keys():
            self.dataset = all_datasets[datasetName]
        if num_elements_per_class == None:
            self.num_elements_per_class = self.dataset["num_elements_per_class"]
        else:
            self.num_elements_per_class = num_elements_per_class
        self.max_classes = min(len(self.num_elements_per_class), 50)
        
                
    def select_classes(self, ways):
        # number of ways for this episode
        n_ways = ways if ways!=0 else random.randint(5, self.max_classes)

        # get n_ways classes randomly
        choices = torch.randperm(len(self.num_elements_per_class))[:n_ways]
        return choices 
    
    def get_query_size(self, choice_classes, n_queries):
        min_queries = n_queries if n_queries != 0 else 10
        query_size = min([int(0.5*self.num_elements_per_class[c]) for c in choice_classes]) 
        query_size = min(min_queries, query_size)
        return query_size

    def get_support_size(self, choice_classes, query_size, n_shots):
        # sample beta uniformly from (0,1]
        if n_shots == 0:
            beta = 0.
            while beta == 0.:
                beta = torch.rand(1).item()
            support_size = sum([math.ceil(beta*min(100, self.num_elements_per_class[c]-query_size)) for c in choice_classes])
            support_size = min(500, support_size)
        else:
            support_size = len(choice_classes)*n_shots
        return support_size
    
    def get_number_of_shots(self, choice_classes, support_size, query_size, n_shots):
        if n_shots == 0: 
            n_ways = len(choice_classes)
            alphas = torch.Tensor(np.random.rand(n_ways)*(np.log(2)-np.log(0.5))+np.log(0.5)) # sample uniformly between log(0.5) and log(2)
            proportions = torch.exp(alphas)*torch.cat([torch.Tensor([self.num_elements_per_class[c]]) for c in choice_classes])
            proportions /= proportions.sum() # make sum to 1
            n_shots_per_class = ((proportions*(support_size-n_ways)).int()+1).tolist()
            n_shots_per_class = [min(n_shots_per_class[i], self.num_elements_per_class[c]-query_size) for i,c in enumerate(choice_classes)]
        else:
            n_shots_per_class = [n_shots]*len(choice_classes)
        return n_shots_per_class

    def get_number_of_queries(self, choice_classes, query_size):
        if self.balanced_queries:
            n_queries_per_class = [query_size]*len(choice_classes)
        else:
            n_queries_per_class = [random.randint(1, query_size) for _ in range(len(choice_classes))]

        return n_queries_per_class

    def sample_indices(self, num_elements_per_chosen_classes, n_shots_per_class, n_queries_per_class):
        shots_idx = []
        queries_idx = []
        for k, q, elements_per_class in zip(n_shots_per_class, n_queries_per_class, num_elements_per_chosen_classes):
            choices = torch.randperm(elements_per_class)
            shots_idx.append(choices[:k].tolist())
            queries_idx.append(choices[k:k+q].tolist())
        return shots_idx, queries_idx

    def sample_episode(self, ways=0, n_shots=0, n_queries=0):
        """
        Sample an episode
        """
        # get n_ways classes randomly
        choice_classes = self.select_classes(ways=ways)
        
        query_size = self.get_query_size(choice_classes, n_queries)
        support_size = self.get_support_size(choice_classes, query_size, n_shots)

        n_shots_per_class = self.get_number_of_shots(choice_classes, support_size, query_size, n_shots)
        n_queries_per_class = self.get_number_of_queries(choice_classes, query_size)
        shots_idx, queries_idx = self.sample_indices([self.num_elements_per_class[c] for c in choice_classes], n_shots_per_class, n_queries_per_class)

        if self.verbose:
            print(f'chosen class: {choice_classes}')
            print(f'n_ways={len(choice_classes)}, q={query_size}, S={support_size}, n_shots_per_class={n_shots_per_class}')
            print(f'queries per class:{n_queries_per_class}')
            print(f'shots_idx: {shots_idx}')
            print(f'queries_idx: {queries_idx}')

        return {'choice_classes':choice_classes, 'shots_idx':shots_idx, 'queries_idx':queries_idx}

    def get_features_from_indices(self, features, episode):
        """
        Get features from a list of all features and from a dictonnary describing an episode
        """
        choice_classes, shots_idx, queries_idx = episode['choice_classes'], episode['shots_idx'], episode['queries_idx']
        shots, queries = [], []
        for i, c in enumerate(choice_classes):
            shots.append(features[c]['features'][shots_idx[i]])
            queries.append(features[c]['features'][queries_idx[i]])
        return shots, queries

class ImageNetGenerator(EpisodicGenerator):
    """
    """
    def __init__(self, datasetName, **kwargs):
        super().__init__(datasetName, **kwargs)
        # Read ImageNet graph

        split = {'train':'TRAIN', 'test':'TEST', 'validation': 'VALID'}[datasetName.split('_')[-1]]
        with open(os.path.join('datasets', 'ilsvrc_2012_dataset_spec.json'), 'r') as file:
            self.graph = json.load(file)['split_subgraphs'][split]

        self.graph_map = {node['wn_id']:node['children_ids'] for node in self.graph}
        self.node_candidates = [node for node in self.graph_map.keys() if 5<=len(self.get_spanning_leaves(node))<=392]
        self.classIdx = self.dataset["classIdx"]
    def get_spanning_leaves(self, node):
        """
        Given a graph and a node return the list of all leaves spanning from the node 
        """
        if len(self.graph_map[node]) == 0:
            return [node]
        else:
            names = []
            for children in self.graph_map[node]:
                names += self.get_spanning_leaves(children)
            return names

    def select_classes(self, ways):
        """
        Different protocol for ImageNet
        """
        # Sample a node in the graph
        node = self.node_candidates[random.randint(0, len(self.node_candidates)-1)]

        leaves_candidates = self.get_spanning_leaves(node)
        max_classes = min(len(leaves_candidates), 50)
        # Sample a number of ways
        n_ways = ways if ways!=0 else random.randint(5, max_classes)

        # get n_ways classes randomly from the subgraph
        choices_idx = torch.randperm(len(leaves_candidates))[:n_ways]
        choices_names = [leaves_candidates[idx] for idx in choices_idx]
        choices = torch.Tensor([self.classIdx[leaf] for leaf in choices_names]).int()
        return choices 
class OmniglotGenerator(EpisodicGenerator):
    """
    """
    def __init__(self, datasetName, **kwargs):
        super().__init__(datasetName, **kwargs)

    def select_classes(self, ways):
        """
        Different protocol for Omniglot
        """
        pass
if __name__=='__main__':
    print('Test')
    print(args.dataset)

    for suffix in ['_validation', '_test']:
        if not (args.dataset == 'mnist' and suffix == '_validation'):
            for _ in range(1):
                print(f'\n---------------Generating episodes for {args.dataset+suffix}--------------------')
                generator = ImageNetGenerator(args.dataset+suffix, verbose=True, balanced_queries=False)
                _= generator.sample_episode(n_queries=args.few_shot_queries, ways=args.few_shot_ways, n_shots=args.few_shot_shots)
  