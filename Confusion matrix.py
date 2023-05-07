import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# User inputs data
num_classes = int(input("Enter the number of classes: "))
class_labels = []
for i in range(num_classes):
    class_label = input(f"Enter label for class {i+1}: ")
    class_labels.append(class_label)

actual_labels = input(f"Enter the actual labels separated by a space: ")
actual_labels = actual_labels.split()

predicted_labels = input(f"Enter the predicted labels separated by a space: ")
predicted_labels = predicted_labels.split()

# Create confusion matrix
labels = np.array(class_labels)
cm = confusion_matrix(actual_labels, predicted_labels, labels=labels)

# Plot confusion matrix
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print confusion matrix
print('Confusion Matrix:')
print(df_cm)

Output
Enter the number of classes: 3
Enter label for class 1: bird
Enter label for class 2: cat
Enter label for class 3: dog
Enter the actual labels separated by a space: cat cat dog dog cat bird bird dog cat dog
Enter the predicted labels separated by a space: cat dog dog dog cat bird bird dog cat cat
