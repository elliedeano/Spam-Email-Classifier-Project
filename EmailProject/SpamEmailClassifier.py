import numpy as np

# Create sample data directly in the script
training_spam = np.array([
    [1, 1, 0, 1],  # Label + 3 features
    [0, 0, 1, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 1]
])

testing_spam = np.array([
    [1, 1, 0, 0],
    [0, 1, 1, 0]
])

print("Shape of the spam training data set:", training_spam.shape)
print("Training spam data:")
print(training_spam)

print("Shape of the spam testing data set:", testing_spam.shape)
print("Testing spam data:")
print(testing_spam)

class SpamClassifier:
    # Initializes the object
    def __init__(self, k, training_spam, testing_spam):
        self.k = k
        # Data attribute is the training spam and testing spam concatenated together
        self.data = np.concatenate([training_spam, testing_spam])
        # Trains the classifier after initializing the data
        self.train()

    # Estimates the log of the class priors for spam and ham
    def estimate_log_class_priors(self):
        # Extracts labels from the first column
        spamOrHam = self.data[:, 0]
        all_data = len(spamOrHam)
        # Calculates the frequency of each class
        one_freq = np.count_nonzero(spamOrHam == 1) / all_data
        zero_freq = np.count_nonzero(spamOrHam == 0) / all_data
        # Handle zero frequency case to avoid log(0)
        epsilon = 1e-10
        one_freq = max(one_freq, epsilon)
        zero_freq = max(zero_freq, epsilon)
        # Calculate the logs
        log_one = np.log(one_freq)
        log_zero = np.log(zero_freq)
        # Put it into an array
        log_class_priors = np.array([log_zero, log_one])
        # Return the array
        return log_class_priors

    # Estimates the log of the conditional likelihoods of each feature given the class
    def estimate_log_class_conditional_likelihoods(self):
        # Take the first column (spam or ham)
        spamOrHam = self.data[:, 0]
        features = self.data[:, 1:]
        # For each class, it counts the occurrences of each feature given whether it's a 1 or 0
        count_numbers = np.zeros((2, features.shape[1]))
        count_numbers[0] = np.count_nonzero(features[spamOrHam == 0], axis=0)
        count_numbers[1] = np.count_nonzero(features[spamOrHam == 1], axis=0)
        # Apply Laplace smoothing
        alpha = 1
        if alpha > 0:
            count_numbers += alpha
        # Divide count by total count
        total_count = np.sum(count_numbers, axis=1)
        total_count = np.maximum(total_count, alpha * features.shape[1])  # Avoid zero in the denominator
        log_calc = count_numbers / total_count[:, np.newaxis]
        # Take log of the result
        log_likelihoods = np.log(log_calc)
        # Return that log
        return log_likelihoods

    # Calls the two functions that train the classifier
    def train(self):
        self.log_class_priors = self.estimate_log_class_priors()
        self.log_class_conditional_likelihoods = self.estimate_log_class_conditional_likelihoods()

    # Predicts the classes of the new data as spam or ham
    def predict(self, new_data):
        # Ensure new_data has the correct number of features
        num_features = self.log_class_conditional_likelihoods.shape[1]
        if new_data.shape[1] != num_features:
            raise ValueError(
                "Feature dimension mismatch: new_data has {} features, but training data has {} features.".format(
                    new_data.shape[1], num_features
                )
            )
        # Calculates log likelihood of the new data belonging to each class based on the class priors and conditional likelihoods estimated during training
        log_likelihoods = np.dot(new_data, self.log_class_conditional_likelihoods.T) + self.log_class_priors
        # Class with highest log likelihood is predicted
        class_predictions = np.argmax(log_likelihoods, axis=1)
        return class_predictions

# Create a SpamClassifier object with specified parameters
def create_classifier():
    classifier = SpamClassifier(k=1, training_spam=training_spam, testing_spam=testing_spam)
    return classifier

# Instance is created by calling create_classifier
classifier = create_classifier()

# Sample new data to predict
new_data = np.array([
    [1, 1, 0],  # New sample, features should match the training data feature count
    [0, 1, 1]   # New sample, features should match the training data feature count
])

# Predict classes for new data
predictions = classifier.predict(new_data)

# Print the predictions
print("Predictions for new data:", predictions)


