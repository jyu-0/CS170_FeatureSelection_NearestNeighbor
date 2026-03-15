import math
import time

def load_data(filename):
    """
    The first column is the class (1 or 2)[cite: 32].
    The other columns contain the continuous features[cite: 33].
    """
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # splits by whitespace and converts strings to floats
            data.append([float(x) for x in line.split()])
    return data

def leave_one_out_cross_validation(data, current_set_of_features):
    """
    calculate the accuracy using Nearest Neighbor and leave-one-out eval
    """
    number_correctly_classified = 0
    
    for i in range(len(data)):
        label_object_to_classify = data[i][0]
        
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_label = 0
        
        for k in range(len(data)):
            if k != i:
                distance = 0
                # calc Euclidean distance using only the features in our current subset
                for feature_idx in current_set_of_features:
                    # feature 1 is at ind 1, feature 2 is at ind 2, etc
                    diff = data[i][feature_idx] - data[k][feature_idx]
                    distance += diff ** 2
                distance = math.sqrt(distance)
                
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_label = data[k][0]
        
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1
            
    return number_correctly_classified / len(data)

def forward_selection(data):
    # total col minus 1 for the class label
    num_features = len(data[0]) - 1 
    current_set_of_features = []
    
    best_overall_accuracy = 0
    best_overall_features = []

    print(f"This dataset has {num_features} features (not including the class attribute), with {len(data)} instances.")
    print("Beginning search.\n")

    # early stopping vars
    previous_level_accuracy = 0
    decrease_counter = 0

    for i in range(num_features):
        feature_to_add_at_this_level = None
        best_so_far_accuracy = 0

        # test every feature to see which one is the best to add
        for k in range(1, num_features + 1):
            if k not in current_set_of_features:
                features_to_try = current_set_of_features + [k]
                accuracy = leave_one_out_cross_validation(data, features_to_try)
                
                print(f"\tUsing feature(s) {features_to_try} accuracy is {accuracy * 100:.1f}%")

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k

        # add best feature found at this level to our current set
        current_set_of_features.append(feature_to_add_at_this_level)
        print(f"\nFeature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy * 100:.1f}%\n")
        
        # track all-time best feature set
        if best_so_far_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_so_far_accuracy
            best_overall_features = current_set_of_features.copy()

        # early stop logic: check if accuracy dropped compared to the previous level
        if best_so_far_accuracy < previous_level_accuracy:
            decrease_counter += 1
            print(f"Warning: Accuracy decreased. (Decrease count: {decrease_counter})")
            if decrease_counter >= 2:
                print("Accuracy has decreased at two consecutive levels. Halting search early to save time.")
                break # Stop the search
        else:
            # reset counter if accuracy improved/stayed the same
            decrease_counter = 0 

        previous_level_accuracy = best_so_far_accuracy

    print(f"Finished. The best feature subset is {best_overall_features}, which has an accuracy of {best_overall_accuracy * 100:.1f}%")
    return best_overall_features

def backward_elimination(data):
    num_features = len(data[0]) - 1 
    # start with list of all features [1, 2, 3... num_features]
    current_set_of_features = list(range(1, num_features + 1))
    
    # calc the baseline accuracy with all features
    baseline_accuracy = leave_one_out_cross_validation(data, current_set_of_features)
    best_overall_accuracy = baseline_accuracy
    best_overall_features = current_set_of_features.copy()

    print(f"\nBeginning Backward Elimination search.")
    print(f"Using all features {current_set_of_features} accuracy is {baseline_accuracy * 100:.1f}%\n")

    # Early stopping variables
    previous_level_accuracy = baseline_accuracy
    decrease_counter = 0

    for i in range(num_features - 1):
        feature_to_remove_at_this_level = None
        best_so_far_accuracy = 0

        for k in current_set_of_features:
            features_to_try = current_set_of_features.copy()
            features_to_try.remove(k)
            
            accuracy = leave_one_out_cross_validation(data, features_to_try)
            
            print(f"\tRemoving feature {k}, using feature(s) {features_to_try} accuracy is {accuracy * 100:.1f}%")

            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_remove_at_this_level = k

        # remove worst feature found at this level
        current_set_of_features.remove(feature_to_remove_at_this_level)
        print(f"\nFeature set {current_set_of_features} was best, accuracy is {best_so_far_accuracy * 100:.1f}%\n")
        
        # remove worst feature found at this level
        if best_so_far_accuracy > best_overall_accuracy:
            best_overall_accuracy = best_so_far_accuracy
            best_overall_features = current_set_of_features.copy()

        # Early Stopping Logic: Check if accuracy dropped compared to the previous level
        if best_so_far_accuracy < previous_level_accuracy:
            decrease_counter += 1
            print(f"Warning: Accuracy decreased. (Decrease count: {decrease_counter})")
            if decrease_counter >= 2:
                print("Accuracy has decreased at two consecutive levels. Halting search early to save time.")
                break # Stop the search
        else:
            decrease_counter = 0 

        previous_level_accuracy = best_so_far_accuracy

    print(f"Finished search. The best feature subset is {best_overall_features}, which has an accuracy of {best_overall_accuracy * 100:.1f}%")
    return best_overall_features

# --- TESTING ---
if __name__ == "__main__":
    try:
        print("Welcome to the Feature Selection Algorithm.")
        file_name = 'CS170_Small_DataSet__41.txt' 
        print(f"Type in the name of the file to test: {file_name}")
        
        data = load_data(file_name)
        
        # print("\n--- Running Forward Selection ---")
        # forward_selection(data)
        
        # print("\n--- Running Backward Elimination ---")
        # backward_elimination(data)

        print("\n--- Running Forward Selection ---")
        start_time_forward = time.time()
        forward_selection(data)
        end_time_forward = time.time()
        print(f"Forward Selection took {(end_time_forward - start_time_forward) / 60:.2f} minutes.")
        
        print("\n--- Running Backward Elimination ---")
        start_time_backward = time.time()
        backward_elimination(data)
        end_time_backward = time.time()
        print(f"Backward Elimination took {(end_time_backward - start_time_backward) / 60:.2f} minutes.")
        
    except FileNotFoundError:
        print(f"Please make sure '{file_name}' is in your workspace.")