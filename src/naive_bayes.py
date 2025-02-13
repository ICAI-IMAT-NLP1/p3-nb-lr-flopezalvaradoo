import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.shape[1]
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # Count number of samples for each output class and divide by total of samples
        classes, counts = torch.unique(labels, return_counts = True)
        total_counts = labels.size(0)
        class_priors: Dict[int, torch.Tensor] = {int(cl): torch.Tensor(count/total_counts) for cl, count in zip(classes, counts)}
        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # Estimate conditional probabilities for the words in features and apply smoothing
        unique_classes = torch.unique(labels)
        num_classes = unique_classes.size(0)  
        vocab_size = features.shape[1]  

        class_word_counts = torch.zeros((num_classes, vocab_size))

        for c in unique_classes:
            class_mask = labels == c  
            class_word_counts[c] = features[class_mask].sum(dim=0) 


        class_word_totals = class_word_counts.sum(dim=1, keepdim=True)
        conditional_probabilities = (class_word_counts + delta) / (class_word_totals + delta * vocab_size)
        return {int(c): conditional_probabilities[c] for c in unique_classes}

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # Calculate posterior based on priors and conditional probabilities of the words
        
        log_priors = torch.log(torch.tensor([self.class_priors[c] for c in self.class_priors.keys()]))
        log_conditional_probs = torch.log(torch.stack([self.conditional_probabilities[c] for c in self.class_priors.keys()]))
        
        log_likelihood = (feature * log_conditional_probs).sum(dim = 1) 
        log_posteriors: torch.Tensor = log_priors + log_likelihood
        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        # Calculate log posteriors and obtain the class of maximum likelihood 
        log_posteriors = self.estimate_class_posteriors(feature)
        pred: int = torch.argmax(log_posteriors).item()
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # Calculate log posteriors and transform them to probabilities (softmax)
        log_posteriors = self.estimate_class_posteriors(feature)
        exp_posteriors = torch.exp(log_posteriors)
        sum_exp_posteriors = exp_posteriors.sum(dim = 0) 
        probs: torch.Tensor = exp_posteriors / sum_exp_posteriors
        return probs
