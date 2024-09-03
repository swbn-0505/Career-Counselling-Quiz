import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset with the added 'Career' column
df = pd.read_csv('Career_Dataset.csv')

input_data = df.drop("ROLE" , axis = "columns")
target_data =  df["ROLE"]

le_MATHS = LabelEncoder()
le_PHYSICS_CHEMISTRY = LabelEncoder()
le_COMPUTER = LabelEncoder()
le_BIOLOGY = LabelEncoder()
le_LANGUAGE = LabelEncoder()
le_HUMANITIES = LabelEncoder()
le_PERSONALITY = LabelEncoder()
le_HOBBIES = LabelEncoder()

input_data["MATHS"] = le_MATHS.fit_transform(input_data["MATHS"])
input_data["PHYSICS/ CHEMISTRY"] = le_PHYSICS_CHEMISTRY.fit_transform(input_data["PHYSICS/ CHEMISTRY"])
input_data["COMPUTER"] = le_COMPUTER.fit_transform(input_data["COMPUTER"])
input_data["BIOLOGY"] = le_BIOLOGY.fit_transform(input_data["BIOLOGY"])
input_data["LANGUAGE"] = le_LANGUAGE.fit_transform(input_data["LANGUAGE"])
input_data["HUMANITIES"] = le_HUMANITIES.fit_transform(input_data["HUMANITIES"])
input_data["PERSONALITY "] = le_PERSONALITY.fit_transform(input_data["PERSONALITY "])
input_data["HOBBIES"] = le_HOBBIES.fit_transform(input_data["HOBBIES"])

# Split the dataset into training and testing sets
X_trainset , X_testset , y_trainset , y_testset = train_test_split(input_data , target_data , test_size = 0.1 , random_state = 3)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(input_data, target_data)

# Save the trained model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model trained and saved as 'model.pkl'.")
